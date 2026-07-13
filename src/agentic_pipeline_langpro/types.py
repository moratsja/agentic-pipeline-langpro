from __future__ import annotations

from enum import Enum


class BaselineOutcome(str, Enum):
    """How the LangPro no-KB step relates to the rest of the pipeline."""

    PROVER_ERROR = "prover_error"
    ALREADY_CORRECT = "already_correct"
    WRONG_NON_NEUTRAL = "wrong_non_neutral"
    NEUTRAL_BASELINE = "neutral_baseline"


class AgenticSkipReason(str, Enum):
    """Why agentic KB generation did not run (mirrors kbprojection early exits)."""

    PROVER_ERROR = "prover_error"
    BASELINE_SOLVED = "baseline_solved"
    WRONG_NON_NEUTRAL = "wrong_non_neutral"


class AgenticStopReason(str, Enum):
    """Why the agent loop ended."""

    SOLVED = "solved"
    MAX_ITERATIONS = "max_iterations"
    EMPTY_KB = "empty_kb"
    LLM_ERROR = "llm_error"
    LANGPRO_ERROR = "langpro_error"
