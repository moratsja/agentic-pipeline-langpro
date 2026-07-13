"""
Knowledge Base PROver inJECTION (kbprojection).

Vendored subset of https://github.com/ettoc00/kbprojection (MIT license,
Copyright (c) 2025-2026 Ettore Cesari; see LICENSE in this directory).

This package provides the substrate used by the agentic pipeline:
LangPro API access and caching, the multi-provider async LLM client,
KB parsing/filtering, and the SNLI/SICK dataset loaders. The upstream
experiment-orchestration modules (``orchestration``, ``runners``,
``utils``, ``data``) are intentionally not vendored.
"""

__version__ = "0.5.0"

from .prompts import get_prompt, fill_prompt
from .models import (
    NLIProblem,
    NLILabel,
    LangProResult,
    ExperimentResult,
    ExperimentStepStatus,
)
from .loaders.base import DatasetLoader
from .loaders.snli import SNLILoader
from .loaders.sick import SICKLoader
from .easyccg_vendor import install_local_easyccg

from .langpro import (
    clear_langpro_cache,
    get_langpro_cache_backend,
    langpro_api_call,
    set_langpro_cache_backend,
)
from .llm import call_llm
from .filtering import (
    tokenize,
    remove_underscores,
    normalize_kb_args,
    drop_leading_preposition,
    parse_kb_injection,
    filter_kb_by_prem_hyp,
)
