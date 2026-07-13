from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolvedLLM:
    model: str


def resolve_llm(*, model: Optional[str] = None) -> ResolvedLLM:
    """Resolve OpenRouter model for agentic pipeline.

    Model must come from --model or LLM in .env.
    """
    resolved_model = model or os.environ.get("LLM") or os.environ.get("AGENTIC_LANGPRO_MODEL")
    if not resolved_model:
        raise ValueError(
            "No model configured. Set LLM in .env, pass --model, "
            "or set AGENTIC_LANGPRO_MODEL."
        )
    return ResolvedLLM(model=resolved_model)
