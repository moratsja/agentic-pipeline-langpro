"""Agentic generate-prove-refine loop for LangPro NLI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kbprojection.filtering import pipeline_filter_kb_injections
from kbprojection.langpro import langpro_api_call
from kbprojection.llm import LLMGenerationError
from kbprojection.models import (
    ExperimentResult,
    ExperimentStatus,
    ExperimentStepStatus,
    LangProResult,
    NLILabel,
    NLIProblem,
)

from .agent_llm import generate_kb, generate_text
from .critic import CriticResult, parse_critic_output
from .langpro_trace import build_langpro_trace, format_critic_tableau_context
from .llm_config import ResolvedLLM, resolve_llm
from .prompts import fill_critic_prompt, fill_kb_prompt, fill_retry_prompt
from .types import AgenticSkipReason, AgenticStopReason, BaselineOutcome

_SKIP_OUTCOMES = {
    AgenticSkipReason.PROVER_ERROR: BaselineOutcome.PROVER_ERROR,
    AgenticSkipReason.BASELINE_SOLVED: BaselineOutcome.ALREADY_CORRECT,
    AgenticSkipReason.WRONG_NON_NEUTRAL: BaselineOutcome.WRONG_NON_NEUTRAL,
}


@dataclass
class AgenticIteration:
    attempt: int
    prompt_kind: str = "initial"
    kb_raw: List[str] = field(default_factory=list)
    kb_filtered: List[str] = field(default_factory=list)
    pred_with_kb: Optional[NLILabel] = None
    langpro_error: Optional[str] = None
    langpro_trace: Optional[Dict[str, Any]] = None
    llm_output_raw: Optional[str] = None
    llm_error: Optional[str] = None
    critic: Optional[CriticResult] = None


@dataclass
class AgenticRunMetadata:
    baseline_outcome: BaselineOutcome
    skip_reason: Optional[AgenticSkipReason] = None
    stop_reason: Optional[AgenticStopReason] = None
    baseline_pred: Optional[str] = None
    baseline_error: Optional[str] = None
    iterations: List[AgenticIteration] = field(default_factory=list)
    llm: Optional[ResolvedLLM] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def baseline_from_problem_input(problem: NLIProblem) -> Optional[LangProResult]:
    """Reuse pred_baseline from JSONL input instead of calling LangPro."""
    original = problem.original_data or {}
    pred_raw = original.get("pred_baseline")
    if pred_raw is None or not str(pred_raw).strip():
        return None

    error = original.get("baseline_error")
    error = str(error).strip() if error is not None else None
    if error == "":
        error = None

    try:
        label = NLILabel(str(pred_raw).strip().lower())
    except ValueError:
        label = NLILabel.UNKNOWN
    return LangProResult(label=label, error=error)


def _baseline_skip(problem: NLIProblem, baseline: LangProResult) -> Optional[AgenticSkipReason]:
    if baseline.error:
        return AgenticSkipReason.PROVER_ERROR
    if baseline.label == problem.gold_label:
        return AgenticSkipReason.BASELINE_SOLVED
    if baseline.label != NLILabel.NEUTRAL:
        return AgenticSkipReason.WRONG_NON_NEUTRAL
    return None


def _build_experiment_result(
    problem: NLIProblem,
    baseline: Optional[LangProResult],
    iteration: AgenticIteration,
    *,
    extra_prover_call: Optional[LangProResult] = None,
) -> ExperimentResult:
    calls: List[Optional[LangProResult]] = [baseline] if baseline else []
    if extra_prover_call is not None:
        calls.append(extra_prover_call)

    exp = ExperimentResult(problem=problem, prover_calls=[c for c in calls if c is not None])
    exp.pred_no_kb = baseline.label if baseline else None
    if baseline is None:
        exp.status_no_kb = ExperimentStepStatus.SKIPPED
    else:
        exp.status_no_kb = (
            ExperimentStepStatus.ERROR if baseline.error else ExperimentStepStatus.SUCCESS
        )

    exp.kb_raw = iteration.kb_raw
    exp.kb_filtered = iteration.kb_filtered
    exp.llm_output_raw = iteration.llm_output_raw
    exp.llm_error = iteration.llm_error
    exp.pred_with_kb = iteration.pred_with_kb

    if iteration.langpro_error:
        exp.status_with_kb = ExperimentStepStatus.ERROR
        exp.final_status = ExperimentStatus.NORMALISED_KB_PROVER_FAILED
    elif iteration.pred_with_kb == problem.gold_label:
        exp.status_with_kb = ExperimentStepStatus.SUCCESS
        exp.final_status = ExperimentStatus.NORMALISED_KB_SOLVED
        exp.fixed_by = "normalised_kb"
    elif not iteration.kb_filtered:
        exp.final_status = ExperimentStatus.KB_NORMALISATION_EMPTY
    elif not iteration.kb_raw:
        exp.final_status = ExperimentStatus.KB_GENERATION_EMPTY
    else:
        exp.status_with_kb = ExperimentStepStatus.SUCCESS
        exp.final_status = ExperimentStatus.KB_NOT_SOLVED
    return exp


def _baseline_skip_result(
    problem: NLIProblem,
    baseline: LangProResult,
    skip: AgenticSkipReason,
    llm: ResolvedLLM,
    *,
    from_input: bool,
    ) -> tuple[ExperimentResult, AgenticRunMetadata]:
    exp = ExperimentResult(problem=problem, prover_calls=[baseline])
    exp.pred_no_kb = baseline.label
    exp.status_no_kb = (
        ExperimentStepStatus.ERROR if baseline.error else ExperimentStepStatus.SUCCESS
    )
    if skip == AgenticSkipReason.BASELINE_SOLVED:
        exp.final_status = ExperimentStatus.BASELINE_SOLVED
    elif skip == AgenticSkipReason.PROVER_ERROR:
        exp.final_status = ExperimentStatus.BASELINE_PROVER_FAILED
    else:
        exp.final_status = ExperimentStatus.KB_NOT_SOLVED

    meta = AgenticRunMetadata(
        baseline_outcome=_SKIP_OUTCOMES[skip],
        skip_reason=skip,
        baseline_pred=baseline.label.value,
        baseline_error=baseline.error,
        llm=llm,
        extra={"baseline_from_input": from_input},
    )
    return exp, meta


def _agent_exit(
    problem: NLIProblem,
    baseline: LangProResult,
    iteration: AgenticIteration,
    meta: AgenticRunMetadata,
    *,
    stop: AgenticStopReason,
    solved: bool,
    prover_with_kb: Optional[LangProResult] = None,
) -> tuple[ExperimentResult, AgenticRunMetadata]:
    meta.iterations.append(iteration)
    meta.stop_reason = stop
    meta.extra["agentic_solved"] = solved
    exp = _build_experiment_result(
        problem,
        baseline,
        iteration,
        extra_prover_call=prover_with_kb if solved else None,
    )
    return exp, meta


async def arun_agentic_problem(
    problem: NLIProblem,
    *,
    model: Optional[str] = None,
    max_iterations: int = 3,
    post_process: bool = True,
    verbose: bool = True,
    context: Any = None,
) -> tuple[ExperimentResult, AgenticRunMetadata]:
    """
    1. LangPro without KB (or reuse pred_baseline from input).
    2. If neutral and wrong: LLM proposes KB -> filter -> LangPro with KB.
    3. On failure: critic analyses -> retry until solved or max_iterations.
    """
    llm = resolve_llm(model=model)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\n[agentic] {problem.id} | gold={problem.gold_label.value}")

    cached = baseline_from_problem_input(problem)
    if cached is not None:
        baseline = cached
        baseline_trace = None
        baseline_trace_note = (
            "baseline label reused from input pred_baseline; LangPro was not called"
        )
        log(f"  [agentic] baseline from input: pred={baseline.label.value}")
    else:
        baseline = await langpro_api_call(
            problem.premises,
            problem.hypothesis,
            report=False,
            context=context,
        )
        baseline_trace = build_langpro_trace(baseline, kb_sent=[])
        baseline_trace_note = None
        log(f"  [agentic] baseline LangPro: pred={baseline.label.value}")

    skip = _baseline_skip(problem, baseline)
    if skip is not None:
        if skip == AgenticSkipReason.BASELINE_SOLVED:
            log("  [agentic] baseline correct -> done")
        elif skip == AgenticSkipReason.PROVER_ERROR:
            log("  [agentic] baseline prover error -> stop")
        else:
            log("  [agentic] baseline wrong (non-neutral) -> skip KB")
        return _baseline_skip_result(
            problem, baseline, skip, llm, from_input=cached is not None
        )

    premise = "\n".join(problem.premises)
    hypothesis = problem.hypothesis
    fallback = baseline.label

    meta = AgenticRunMetadata(
        baseline_outcome=BaselineOutcome.NEUTRAL_BASELINE,
        baseline_pred=baseline.label.value,
        baseline_error=baseline.error,
        llm=llm,
        extra={"baseline_from_input": cached is not None},
    )
    critic_notes: List[str] = []
    last: Optional[AgenticIteration] = None

    log("  [agentic] neutral baseline wrong -> start agent loop")

    for attempt in range(1, max_iterations + 1):
        log(f"  [agentic] attempt {attempt}/{max_iterations}")

        if attempt == 1:
            prompt = fill_kb_prompt(premise, hypothesis)
            kind = "initial"
        else:
            assert last is not None and last.critic is not None
            prompt = fill_retry_prompt(
                premise,
                hypothesis,
                analysis=last.critic.analysis,
                previous_kb="\n".join(last.kb_raw),
                kb_label=last.pred_with_kb.value if last.pred_with_kb else "error",
            )
            kind = "retry"

        it = AgenticIteration(attempt=attempt, prompt_kind=kind)
        lp_with_kb: Optional[LangProResult] = None

        log(f"  [agentic] LLM KB generation ({kind})...")
        try:
            kb_raw, raw = await generate_kb(model=llm.model, prompt=prompt, context=context)
            it.kb_raw = kb_raw
            it.llm_output_raw = raw
        except (LLMGenerationError, Exception) as exc:
            it.llm_error = str(exc)
            return _agent_exit(
                problem, baseline, it, meta, stop=AgenticStopReason.LLM_ERROR, solved=False
            )

        log(f"  [agentic] proposed KB: {kb_raw}")

        if not kb_raw:
            it.pred_with_kb = fallback
            log("  [agentic] empty KB — LangPro not called")
        else:
            kb_filtered = [
                r.relation
                for r in pipeline_filter_kb_injections(
                    kb_raw,
                    problem.premises,
                    problem.hypothesis,
                    post_process=post_process,
                )
            ]
            it.kb_filtered = kb_filtered
            log(f"  [agentic] filtered KB: {kb_filtered}")

            if not kb_filtered:
                it.pred_with_kb = fallback
                log("  [agentic] all KB removed during filtering")
            else:
                lp_with_kb = await langpro_api_call(
                    problem.premises,
                    problem.hypothesis,
                    kb=kb_filtered,
                    context=context,
                )
                it.langpro_trace = build_langpro_trace(lp_with_kb, kb_sent=kb_filtered)

                if lp_with_kb.error:
                    it.langpro_error = lp_with_kb.error
                    return _agent_exit(
                        problem,
                        baseline,
                        it,
                        meta,
                        stop=AgenticStopReason.LANGPRO_ERROR,
                        solved=False,
                    )

                it.pred_with_kb = lp_with_kb.label
                verified = it.langpro_trace["verification"]["all_sent_found_in_response"]
                log(
                    f"  [agentic] LangPro with KB: {lp_with_kb.label.value}"
                    f" | kb verified: {verified}"
                )

        if it.pred_with_kb == problem.gold_label:
            return _agent_exit(
                problem,
                baseline,
                it,
                meta,
                stop=AgenticStopReason.SOLVED,
                solved=True,
                prover_with_kb=lp_with_kb,
            )

        if attempt >= max_iterations:
            stop = (
                AgenticStopReason.EMPTY_KB if not it.kb_raw else AgenticStopReason.MAX_ITERATIONS
            )
            return _agent_exit(problem, baseline, it, meta, stop=stop, solved=False)

        with_kb_note: Optional[str] = None
        if it.langpro_trace is None:
            if not it.kb_raw:
                with_kb_note = "LLM returned empty KB (no relations proposed)"
            elif not it.kb_filtered:
                with_kb_note = "all KB removed during filtering"

        tableau_ctx = format_critic_tableau_context(
            baseline_trace,
            it.langpro_trace,
            baseline_unavailable_reason=baseline_trace_note,
            with_kb_unavailable_reason=with_kb_note,
        )
        critic_prompt = fill_critic_prompt(
            premise,
            hypothesis,
            baseline_label=baseline.label.value,
            attempted_kb="\n".join(it.kb_filtered or it.kb_raw),
            kb_label=it.pred_with_kb.value if it.pred_with_kb else fallback.value,
            closure_pattern=tableau_ctx["closure_pattern"],
            proof_info=tableau_ctx["proof_info"],
            kb_verification=tableau_ctx["kb_verification"],
            tableau_comparison=tableau_ctx["tableau_comparison"],
            previous_notes="\n---\n".join(critic_notes),
        )

        try:
            critic_raw = await generate_text(
                model=llm.model, prompt=critic_prompt, context=context
            )
            it.critic = parse_critic_output(critic_raw)
            critic_notes.append(it.critic.analysis)
            log(f"  [agentic] critic: {it.critic.analysis[:120]}")
        except Exception as exc:
            it.llm_error = f"critic failed: {exc}"
            return _agent_exit(
                problem, baseline, it, meta, stop=AgenticStopReason.LLM_ERROR, solved=False
            )

        meta.iterations.append(it)
        last = it

    raise RuntimeError("agent loop ended without a result")


def run_agentic_problem(*args: Any, **kwargs: Any) -> tuple[ExperimentResult, AgenticRunMetadata]:
    return asyncio.run(arun_agentic_problem(*args, **kwargs))


async def arun_agentic_batch(
    problems: List[NLIProblem],
    *,
    concurrency: int = 4,
    **kwargs: Any,
) -> List[tuple[ExperimentResult, AgenticRunMetadata]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(p: NLIProblem) -> tuple[ExperimentResult, AgenticRunMetadata]:
        async with semaphore:
            return await arun_agentic_problem(p, **kwargs)

    return await asyncio.gather(*[one(p) for p in problems])
