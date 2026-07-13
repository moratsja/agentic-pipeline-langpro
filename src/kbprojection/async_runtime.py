import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AsyncRunLimits:
    llm_concurrency: int = 2
    langpro_concurrency: int = 4
    local_langpro_concurrency: int = 2


@dataclass
class AsyncRunContext:
    limits: AsyncRunLimits
    llm_semaphore: asyncio.Semaphore
    langpro_semaphore: asyncio.Semaphore
    local_langpro_semaphore: asyncio.Semaphore


def create_async_run_context(limits: Optional[AsyncRunLimits] = None) -> AsyncRunContext:
    resolved_limits = limits or AsyncRunLimits()
    return AsyncRunContext(
        limits=resolved_limits,
        llm_semaphore=asyncio.Semaphore(resolved_limits.llm_concurrency),
        langpro_semaphore=asyncio.Semaphore(resolved_limits.langpro_concurrency),
        local_langpro_semaphore=asyncio.Semaphore(resolved_limits.local_langpro_concurrency),
    )


_DEFAULT_CONTEXTS: dict[int, AsyncRunContext] = {}


def get_default_async_run_context() -> AsyncRunContext:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    context = _DEFAULT_CONTEXTS.get(loop_id)
    if context is None:
        context = create_async_run_context()
        _DEFAULT_CONTEXTS[loop_id] = context
    return context


def resolve_async_run_context(context: Optional[AsyncRunContext] = None) -> AsyncRunContext:
    return context or get_default_async_run_context()
