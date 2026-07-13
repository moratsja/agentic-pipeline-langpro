"""CLI commands for the agentic LangPro pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from kbprojection.models import NLILabel, NLIProblem

from agentic_pipeline_langpro.llm_config import resolve_llm
from agentic_pipeline_langpro.paths import DATA_DIR, RESULTS_DIR, load_repo_dotenv
from agentic_pipeline_langpro.pipeline import arun_agentic_problem
from agentic_pipeline_langpro.results import build_run_record, format_run_report

DEFAULT_INPUT = DATA_DIR / "snli_train_entailment_1k.jsonl"


def load_problems(path: Path, limit: Optional[int] = None) -> List[NLIProblem]:
    rows: List[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    problems: List[NLIProblem] = []
    for row in rows:
        extra = {
            key: row[key]
            for key in ("pred_baseline", "baseline_error")
            if key in row and row[key] is not None
        }
        problems.append(
            NLIProblem(
                id=str(row["id"]),
                premises=[str(row["premise"])],
                hypothesis=str(row["hypothesis"]),
                gold_label=NLILabel(str(row["gold_label"]).lower()),
                dataset=str(row.get("dataset", "snli")),
                split=str(row.get("split", "train")),
                original_data=extra or None,
            )
        )
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic LangPro pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--langpro-builtin", choices=("on", "off"), default="on")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also write a pretty-printed .json file next to the JSONL output.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Also write a markdown .md summary next to the JSONL output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    load_repo_dotenv()
    os.environ["KBPROJECTION_LANGPRO_PROVER_CONFIG_EXTRA"] = (
        "" if args.langpro_builtin == "on" else "no_kb,no_wn"
    )

    input_path = args.input.resolve()
    problems = load_problems(input_path, args.limit)
    if not problems:
        print("No problems loaded.", file=sys.stderr)
        return 1

    out_path = args.output
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"agentic_{ts}.jsonl"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = resolve_llm(model=args.model)
    print(f"Loaded {len(problems)} problems from {input_path}")
    print(f"LLM (OpenRouter): {llm.model}")
    print(f"LangPro WordNet: {args.langpro_builtin}")
    print(f"Output: {out_path}")

    async def run_all() -> List[dict]:
        semaphore = asyncio.Semaphore(args.concurrency)
        records: List[Optional[dict]] = [None] * len(problems)
        done = 0
        start = time.monotonic()

        async def one(slot: int, problem: NLIProblem) -> None:
            nonlocal done
            async with semaphore:
                result, meta = await arun_agentic_problem(
                    problem,
                    model=llm.model,
                    max_iterations=args.max_iterations,
                )
            records[slot] = build_run_record(problem, result, meta)
            done += 1
            elapsed = int(time.monotonic() - start)
            sys.stdout.write(f"\r{done}/{len(problems)} | {elapsed}s")
            sys.stdout.flush()
            if done == len(problems):
                sys.stdout.write("\n")

        await asyncio.gather(*(one(i, p) for i, p in enumerate(problems)))
        return [r for r in records if r is not None]

    try:
        records = asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")

    if args.pretty:
        pretty_path = out_path.with_suffix(".json") if out_path.suffix.lower() != ".json" else out_path
        pretty_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote pretty JSON to {pretty_path}")

    if args.write_report:
        report_path = out_path.with_suffix(".md")
        report_path.write_text(
            "\n\n---\n\n".join(format_run_report(r) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
