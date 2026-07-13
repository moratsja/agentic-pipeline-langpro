"""Load and fill the three pipeline prompt templates from <repo root>/prompts/."""

from __future__ import annotations

from .paths import PROMPTS_DIR

KB_PROMPT_FILE = "knowledge_generation.txt"
CRITIC_PROMPT_FILE = "failure_analysis.txt"
RETRY_PROMPT_FILE = "knowledge_refinement.txt"


def _read(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _rules_block() -> str:
    """Rules + final check + examples from the KB prompt (shared by retry prompt)."""
    full = _read(KB_PROMPT_FILE)
    start = full.find("## STRICT RULES")
    end = full.find("Premise: {premise}")
    if start == -1 or end == -1:
        raise ValueError(
            f"{KB_PROMPT_FILE} must contain ## STRICT RULES and Premise: {{premise}}"
        )
    return full[start:end].strip()


def fill_kb_prompt(premise: str, hypothesis: str) -> str:
    template = _read(KB_PROMPT_FILE)
    return template.format(premise=premise, hypothesis=hypothesis)


def fill_retry_prompt(
    premise: str,
    hypothesis: str,
    *,
    analysis: str,
    previous_kb: str,
    kb_label: str,
) -> str:
    template = _read(RETRY_PROMPT_FILE)
    return template.format(
        base_rules=_rules_block(),
        analysis=analysis.strip(),
        previous_kb=previous_kb.strip() or "(empty)",
        kb_label=kb_label,
        premise=premise,
        hypothesis=hypothesis,
    )


def fill_critic_prompt(
    premise: str,
    hypothesis: str,
    *,
    baseline_label: str,
    attempted_kb: str,
    kb_label: str,
    closure_pattern: str = "",
    proof_info: str = "",
    kb_verification: str = "",
    tableau_comparison: str = "",
    previous_notes: str = "",
) -> str:
    template = _read(CRITIC_PROMPT_FILE)
    return template.format(
        premise=premise,
        hypothesis=hypothesis,
        baseline_label=baseline_label,
        attempted_kb=attempted_kb.strip() or "(empty)",
        kb_label=kb_label,
        closure_pattern=closure_pattern.strip() or "(not available)",
        proof_info=proof_info.strip() or "(not available)",
        kb_verification=kb_verification.strip() or "(not available)",
        tableau_comparison=tableau_comparison.strip() or "(not available)",
        previous_notes=previous_notes.strip() or "(none)",
    )
