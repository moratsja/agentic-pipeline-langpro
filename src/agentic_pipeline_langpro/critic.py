"""Critic functions"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ANALYSIS_RE = re.compile(
    r"\[ANALYSIS\]\s*(.*?)\s*\[/ANALYSIS\]",
    re.IGNORECASE | re.DOTALL,
)

@dataclass(frozen=True)
class CriticResult:
    analysis: str
    raw_output: str

def parse_critic_output(text: str) -> CriticResult:
    analysis_match = _ANALYSIS_RE.search(text)
    analysis = analysis_match.group(1).strip() if analysis_match else text.strip()
    return CriticResult(analysis=analysis, raw_output=text)
