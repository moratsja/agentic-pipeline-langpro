"""Build JSONL run records from pipeline results."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from kbprojection.models import ExperimentResult, NLIProblem

from .pipeline import AgenticIteration, AgenticRunMetadata
from .types import AgenticSkipReason, AgenticStopReason


def _val(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _step(step: int, phase: str, **fields: Any) -> Dict[str, Any]:
    return {"step": step, "phase": phase, **fields}


def _baseline_narrative(pred: Optional[str], gold: str, skip: Optional[str]) -> str:
    if skip == AgenticSkipReason.BASELINE_SOLVED.value:
        return f"Baseline: {pred} = gold ({gold}) -> done."
    if skip == AgenticSkipReason.WRONG_NON_NEUTRAL.value:
        return f"Baseline: {pred} wrong (non-neutral) vs gold {gold} -> skip KB."
    if skip == AgenticSkipReason.PROVER_ERROR.value:
        return "Baseline: prover error -> stop."
    return f"Baseline: {pred} neutral but wrong (gold {gold}) -> start agent."


def _baseline_step(result: ExperimentResult, meta: AgenticRunMetadata) -> Dict[str, Any]:
    pred = _val(result.pred_no_kb) or meta.baseline_pred
    gold = result.problem.gold_label.value
    skip = _val(meta.skip_reason)
    from_input = bool(meta.extra.get("baseline_from_input"))
    return _step(
        1,
        "baseline_langpro",
        title="LangPro baseline (no KB)",
        output={
            "pred": pred,
            "gold": gold,
            "matches_gold": pred == gold if pred else None,
            "prover_error": meta.baseline_error,
            "from_input": from_input,
        },
        decision={
            "baseline_outcome": meta.baseline_outcome.value,
            "skip_reason": skip,
            "continue_to_agent": skip is None,
        },
        narrative=_baseline_narrative(pred, gold, skip),
    )


def _langpro_output(it: AgenticIteration, gold: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pred": _val(it.pred_with_kb),
        "gold": gold,
        "matches_gold": _val(it.pred_with_kb) == gold if it.pred_with_kb else None,
        "prover_error": it.langpro_error,
    }
    if it.langpro_trace:
        out["kb_verification"] = it.langpro_trace.get("verification")
        out["proof_excerpts"] = it.langpro_trace.get("proof_excerpts")
    return out


def _critic_step(step: int, it: AgenticIteration) -> Dict[str, Any]:
    assert it.critic is not None
    return _step(
        step,
        "critic",
        title="Critic",
        attempt=it.attempt,
        output={"analysis": it.critic.analysis},
        status="analysis",
    )


def _attempt_timeline(
    it: AgenticIteration,
    *,
    step: int,
    gold: str,
    stop: Optional[AgenticStopReason],
    is_last: bool,
) -> tuple[List[Dict[str, Any]], List[str]]:
    steps: List[Dict[str, Any]] = []
    lines = [f"--- Attempt {it.attempt} ---"]

    if it.llm_error and not it.kb_raw:
        steps.append(
            _step(
                step,
                "kb_generation",
                title="KB generation",
                attempt=it.attempt,
                status="error",
                output={"llm_output": it.llm_output_raw},
                error=it.llm_error,
            )
        )
        lines.append(f"  KB generation: ERROR — {it.llm_error}")
        return steps, lines

    gen_status = "empty" if not it.kb_raw else "success"
    steps.append(
        _step(
            step,
            "kb_generation",
            title="KB generation",
            attempt=it.attempt,
            prompt_kind=it.prompt_kind,
            status=gen_status,
            output={"kb_raw": it.kb_raw, "llm_output": it.llm_output_raw},
        )
    )
    step += 1
    if not it.kb_raw:
        lines.append("  KB generation: empty")
    else:
        lines.append(f"  KB generation ({it.prompt_kind}): {it.kb_raw}")

    if not it.kb_raw:
        if it.critic:
            steps.append(_critic_step(step, it))
            lines.append(f"  Critic: {it.critic.analysis[:200]}")
        elif is_last and stop == AgenticStopReason.EMPTY_KB:
            lines.append("  >> Run ended: empty_kb.")
        return steps, lines

    steps.append(
        _step(
            step,
            "kb_filtering",
            title="KB filtering",
            attempt=it.attempt,
            status="empty" if not it.kb_filtered else "success",
            input={"kb_raw": it.kb_raw},
            output={"kb_filtered": it.kb_filtered},
        )
    )
    step += 1
    lines.append(
        f"  Filtering: kept {it.kb_filtered}" if it.kb_filtered else f"  Filtering: removed all"
    )

    lp_status = "skipped" if not it.kb_filtered else ("error" if it.langpro_error else "success")
    steps.append(
        _step(
            step,
            "langpro_with_kb",
            title="LangPro with KB",
            attempt=it.attempt,
            status=lp_status,
            input={"kb_sent": it.kb_filtered},
            output=_langpro_output(it, gold),
            error=it.langpro_error,
        )
    )
    step += 1

    pred = _val(it.pred_with_kb)
    if not it.kb_filtered:
        lines.append("  LangPro+KB: skipped")
    elif it.langpro_error:
        lines.append(f"  LangPro+KB: ERROR — {it.langpro_error}")
    else:
        ok = "SOLVED" if pred == gold else "not solved"
        lines.append(f"  LangPro+KB: pred={pred}, gold={gold} -> {ok}")

    if it.critic:
        steps.append(_critic_step(step, it))
        lines.append(f"  Critic: {it.critic.analysis[:200]}")
        step += 1

    if stop == AgenticStopReason.SOLVED and pred == gold:
        lines.append(f"  >> Attempt {it.attempt} solved.")
    elif is_last and stop == AgenticStopReason.MAX_ITERATIONS:
        lines.append(f"  >> Max iterations after attempt {it.attempt}.")

    return steps, lines


def build_run_record(
    problem: NLIProblem,
    result: ExperimentResult,
    meta: AgenticRunMetadata,
) -> Dict[str, Any]:
    """One JSON object per problem run (written as JSONL by the CLI)."""
    gold = problem.gold_label.value
    pred_final = _val(result.pred_with_kb) or _val(result.pred_no_kb)
    solved = meta.extra.get("agentic_solved")
    if solved is None:
        solved = pred_final == gold

    timeline: List[Dict[str, Any]] = [_baseline_step(result, meta)]
    narrative = [timeline[0]["narrative"]]
    step_num = 2

    if meta.skip_reason is not None:
        timeline.append(
            _step(
                step_num,
                "agent_stopped",
                title="Agent stopped",
                status="skipped",
                output={"skip_reason": meta.skip_reason.value},
            )
        )
        narrative.append(f"Agent skipped: {meta.skip_reason.value}.")
    else:
        for i, it in enumerate(meta.iterations):
            attempt_steps, attempt_lines = _attempt_timeline(
                it,
                step=step_num,
                gold=gold,
                stop=meta.stop_reason,
                is_last=i == len(meta.iterations) - 1,
            )
            timeline.extend(attempt_steps)
            narrative.extend(attempt_lines)
            step_num += len(attempt_steps)
        if meta.stop_reason is not None:
            narrative.append(f"Run ended: {meta.stop_reason.value}.")

    record: Dict[str, Any] = {
        "problem_id": problem.id,
        "dataset": problem.dataset,
        "problem": {
            "id": problem.id,
            "premise": problem.premises,
            "hypothesis": problem.hypothesis,
            "gold_label": gold,
        },
        "outcome": {
            "solved": bool(solved),
            "gold_label": gold,
            "pred_baseline": _val(result.pred_no_kb) or meta.baseline_pred,
            "pred_final": pred_final,
            "final_status": _val(result.final_status),
            "baseline_outcome": meta.baseline_outcome.value,
            "skip_reason": _val(meta.skip_reason),
            "stop_reason": _val(meta.stop_reason),
            "attempt_count": len(meta.iterations),
        },
        "timeline": timeline,
        "narrative_lines": narrative,
    }

    if meta.llm is not None:
        record["llm"] = {"provider": "openrouter", "model": meta.llm.model}

    return record


def format_run_report(record: Dict[str, Any]) -> str:
    """Short markdown summary of one run record."""
    outcome = record["outcome"]
    problem = record["problem"]
    stop = outcome.get("stop_reason") or outcome.get("skip_reason") or "—"
    lines = [
        f"# {record['problem_id']}",
        "",
        f"Gold: {problem['gold_label']} | Solved: {outcome['solved']} | Stop: {stop}",
        "",
        f"Premise: {problem['premise'][0]}",
        f"Hypothesis: {problem['hypothesis']}",
        "",
        "Narrative:",
    ]
    for line in record.get("narrative_lines", []):
        lines.append(f"- {line}")

    lines.extend(["", "Timeline:"])
    for step in record.get("timeline", []):
        title = step.get("title") or step.get("phase")
        lines.append(f"- Step {step.get('step')}: {title} [{step.get('status', '')}]")
        if step.get("output"):
            blob = json.dumps(step["output"], ensure_ascii=False)
            lines.append(f"  {blob[:300]}{'...' if len(blob) > 300 else ''}")
        if step.get("error"):
            lines.append(f"  error: {step['error']}")

    return "\n".join(lines)
