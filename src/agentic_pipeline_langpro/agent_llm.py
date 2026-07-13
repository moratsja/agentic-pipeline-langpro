"""LLM generation functions."""
from __future__ import annotations

from typing import List, Optional

from kbprojection.async_runtime import AsyncRunContext, resolve_async_run_context
from kbprojection.llm import AsyncGenericAIClient, _extract_validated_kb_from_output

OPENROUTER = "openrouter"


async def generate_text(
    *,
    model: str,
    prompt: str,
    context: Optional[AsyncRunContext] = None,
) -> str:
    client = AsyncGenericAIClient(provider=OPENROUTER)
    resolved = resolve_async_run_context(context)
    async with resolved.llm_semaphore:
        output = await client.generate(prompt=prompt, model=model)
    if not isinstance(output, str):
        raise TypeError(f"Expected string LLM output, got {type(output)!r}")
    return output


async def generate_kb(
    *,
    model: str,
    prompt: str,
    context: Optional[AsyncRunContext] = None,
) -> tuple[List[str], str]:
    raw = await generate_text(model=model, prompt=prompt, context=context)
    try:
        kb = _extract_validated_kb_from_output(raw)
    except ValueError:
        from kbprojection.llm import extract_kb_from_output

        kb = extract_kb_from_output(raw)
    return kb, raw
