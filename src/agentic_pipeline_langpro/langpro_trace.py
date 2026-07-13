"""LangPro trace helpers for failure analysis."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from kbprojection.models import LangProResult


def normalize_kb_relation(relation: str) -> str:
    return re.sub(r"\s+", " ", relation.strip().lower())


def serialize_langpro_kb(kb: Any) -> List[str]:
    """Turn LangPro parsed KB compounds back into relation strings."""
    if not kb:
        return []
    out: List[str] = []
    for item in kb:
        if isinstance(item, str):
            text = item.strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def verify_kb_echo(sent: List[str], received: List[str]) -> Dict[str, Any]:
    """
    Compare KB we sent to LangPro with KB echoed in the API response.

    LangPro may add WordNet-derived axioms beyond what we sent; we check that
    every sent relation appears in the response.
    """
    recv_norm = {normalize_kb_relation(r) for r in received}
    confirmed = [s for s in sent if normalize_kb_relation(s) in recv_norm]
    missing = [s for s in sent if normalize_kb_relation(s) not in recv_norm]
    sent_norm = {normalize_kb_relation(s) for s in sent}
    extra = [r for r in received if normalize_kb_relation(r) not in sent_norm]
    return {
        "kb_sent": sent,
        "kb_received": received,
        "kb_confirmed_in_response": confirmed,
        "kb_missing_from_response": missing,
        "kb_extra_in_response": extra,
        "all_sent_found_in_response": len(missing) == 0,
    }


def is_tableau_closed(info: Optional[List[Any]]) -> bool:
    """Match kbprojection label logic: tableau closed iff 'closed' in proofs[*].info."""
    if not info:
        return False
    return "closed" in info


def build_langpro_trace(result: LangProResult, *, kb_sent: List[str]) -> Dict[str, Any]:
    """Audit trail for a LangPro call: KB echo check + per-tree closure status."""
    kb_received = serialize_langpro_kb(result.kb)
    verification = verify_kb_echo(kb_sent, kb_received)
    proof_info = result.proof_info or {}

    proof_excerpts: Dict[str, Any] = {}
    for label in ("entailment", "contradiction"):
        info = proof_info.get(label)
        if info is None and (result.proofs or {}).get(label) is None:
            continue
        proof_excerpts[label] = {
            "closed": is_tableau_closed(info if isinstance(info, list) else None),
            "info": list(info) if isinstance(info, list) else [],
        }

    return {
        "verification": verification,
        "proof_excerpts": proof_excerpts,
        "pred": result.label.value if result.label is not None else None,
        "prover_error": result.error,
    }


def _tree_closed_flags(trace: Optional[Dict[str, Any]]) -> tuple[Optional[bool], Optional[bool]]:
    if not trace:
        return None, None
    excerpts = trace.get("proof_excerpts") or {}

    def _closed(label: str) -> Optional[bool]:
        entry = excerpts.get(label)
        if entry is None or "closed" not in entry:
            return None
        return bool(entry["closed"])

    return _closed("entailment"), _closed("contradiction")


def describe_closure_pattern(
    entailment_closed: Optional[bool],
    contradiction_closed: Optional[bool],
    *,
    pred: Optional[str] = None,
    ) -> str:
    if entailment_closed is None and contradiction_closed is None:
        label_note = f" (reported label: {pred})" if pred else ""
        return f"unknown — tableau info not available{label_note}"

    ent = bool(entailment_closed)
    contra = bool(contradiction_closed)
    if ent and not contra:
        return "entailment closed only → entailment"
    if contra and not ent:
        return "contradiction closed only → contradiction"
    if ent and contra:
        return "both closed → neutral (conflicting proofs)"
    return "both open → neutral (no proof)"


def _format_or_unavailable(
    trace: Optional[Dict[str, Any]],
    *,
    unavailable_reason: Optional[str],
    formatter,
) -> str:
    if unavailable_reason:
        return unavailable_reason
    if not trace:
        return "(not available)"
    return formatter(trace)


def format_closure_pattern(
    trace: Optional[Dict[str, Any]],
    *,
    unavailable_reason: Optional[str] = None,
) -> str:
    def _fmt(t: Dict[str, Any]) -> str:
        ent, contra = _tree_closed_flags(t)
        pattern = describe_closure_pattern(ent, contra, pred=t.get("pred"))
        return f"{pattern} | final label: {t.get('pred') or 'unknown'}"

    return _format_or_unavailable(trace, unavailable_reason=unavailable_reason, formatter=_fmt)


def format_proof_info_per_tree(
    trace: Optional[Dict[str, Any]],
    *,
    unavailable_reason: Optional[str] = None,
) -> str:
    def _fmt(t: Dict[str, Any]) -> str:
        excerpts = t.get("proof_excerpts") or {}
        lines: List[str] = []
        for tree_label in ("entailment", "contradiction"):
            entry = excerpts.get(tree_label)
            if entry is None:
                lines.append(f"  {tree_label}: (no proof_info)")
                continue
            info = entry.get("info", [])
            closed_txt = "closed" if entry.get("closed") else "open"
            lines.append(f"  {tree_label}: proof_info={info!r} ({closed_txt})")
        return "\n".join(lines) if lines else "(not available)"

    return _format_or_unavailable(trace, unavailable_reason=unavailable_reason, formatter=_fmt)


def format_kb_verification_block(
    trace: Optional[Dict[str, Any]],
    *,
    unavailable_reason: Optional[str] = None,
) -> str:
    def _fmt(t: Dict[str, Any]) -> str:
        verification = t.get("verification")
        if not verification:
            return "(not available — no KB sent or trace missing verification)"
        sent = verification.get("kb_sent") or []
        confirmed = verification.get("kb_confirmed_in_response") or []
        missing = verification.get("kb_missing_from_response") or []
        extra = verification.get("kb_extra_in_response") or []
        all_ok = verification.get("all_sent_found_in_response")
        lines = [
            f"  KB sent to LangPro: {sent if sent else '(none)'}",
            f"  All sent relations echoed in response: {all_ok}",
            f"  Confirmed in response: {confirmed if confirmed else '(none)'}",
        ]
        if missing:
            lines.append(f"  Missing from response: {missing}")
        if extra:
            preview = extra[:5]
            suffix = f" (+{len(extra) - 5} more)" if len(extra) > 5 else ""
            lines.append(f"  Extra axioms in response (WordNet etc.): {preview}{suffix}")
        return "\n".join(lines)

    return _format_or_unavailable(trace, unavailable_reason=unavailable_reason, formatter=_fmt)


def _tableau_snapshot_lines(label: str, trace: Optional[Dict[str, Any]]) -> List[str]:
    if trace is None:
        return [f"  {label}: tableau not available"]
    pred = trace.get("pred", "unknown")
    ent, contra = _tree_closed_flags(trace)
    lines = [
        f"  {label}, final label: {pred}",
        f"    closure pattern: {describe_closure_pattern(ent, contra, pred=str(pred))}",
    ]
    excerpts = trace.get("proof_excerpts") or {}
    for tree_label in ("entailment", "contradiction"):
        entry = excerpts.get(tree_label)
        if entry is None:
            lines.append(f"    {tree_label} proof_info: (none)")
        else:
            info = entry.get("info", [])
            closed_txt = "closed" if entry.get("closed") else "open"
            lines.append(f"    {tree_label} proof_info: {info!r} ({closed_txt})")
    return lines


def format_tableau_comparison(
    baseline_trace: Optional[Dict[str, Any]],
    with_kb_trace: Optional[Dict[str, Any]],
    *,
    baseline_unavailable_reason: Optional[str] = None,
    with_kb_unavailable_reason: Optional[str] = None,
) -> str:
    lines: List[str] = []
    if baseline_trace is None:
        reason = baseline_unavailable_reason or "LangPro baseline was not run"
        lines.append(f"Baseline (no KB): {reason}")
    else:
        lines.extend(_tableau_snapshot_lines("Baseline (no KB)", baseline_trace))
    lines.append("")
    if with_kb_trace is None and with_kb_unavailable_reason:
        lines.append(f"With KB: {with_kb_unavailable_reason}")
    else:
        lines.extend(_tableau_snapshot_lines("With KB", with_kb_trace))

    if baseline_trace and with_kb_trace:
        base_ent, base_contra = _tree_closed_flags(baseline_trace)
        kb_ent, kb_contra = _tree_closed_flags(with_kb_trace)
        changes: List[str] = []
        for name, before, after in (
            ("entailment tree", base_ent, kb_ent),
            ("contradiction tree", base_contra, kb_contra),
        ):
            if before is None or after is None:
                continue
            if before == after:
                changes.append(f"  {name}: unchanged ({'closed' if after else 'open'})")
            else:
                changes.append(
                    f"  {name}: {'closed' if before else 'open'} → {'closed' if after else 'open'}"
                )
        base_pred = baseline_trace.get("pred")
        kb_pred = with_kb_trace.get("pred")
        if base_pred != kb_pred:
            changes.append(f"  final label: {base_pred} → {kb_pred}")
        if changes:
            lines.append("")
            lines.append("Changes after KB:")
            lines.extend(changes)
    return "\n".join(lines)


def format_critic_tableau_context(
    baseline_trace: Optional[Dict[str, Any]],
    with_kb_trace: Optional[Dict[str, Any]],
    *,
    baseline_unavailable_reason: Optional[str] = None,
    with_kb_unavailable_reason: Optional[str] = None,
) -> Dict[str, str]:
    """Build the four tableau/KB blocks injected into the critic prompt."""
    return {
        "closure_pattern": format_closure_pattern(
            with_kb_trace,
            unavailable_reason=with_kb_unavailable_reason,
        ),
        "proof_info": format_proof_info_per_tree(
            with_kb_trace,
            unavailable_reason=with_kb_unavailable_reason,
        ),
        "kb_verification": format_kb_verification_block(
            with_kb_trace,
            unavailable_reason=with_kb_unavailable_reason,
        ),
        "tableau_comparison": format_tableau_comparison(
            baseline_trace,
            with_kb_trace,
            baseline_unavailable_reason=baseline_unavailable_reason,
            with_kb_unavailable_reason=with_kb_unavailable_reason,
        ),
    }
