from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_data import BASH_TOOLS, assistant_tool_call_count, iter_normalized_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.environ.get("RAW_DIR", "data/raw/vuljector-secbench-trajectories")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gemma4_agent_trajectories.jsonl"),
    )
    parser.add_argument("--model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--skip-token-stats", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], pct: int) -> int:
    ordered = sorted(values)
    idx = round((pct / 100) * (len(ordered) - 1))
    return ordered[idx]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_normalized_trajectories(args.raw_dir))
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    role_counts = Counter()
    task_modes = Counter()
    projects = set()
    tool_calls = 0
    message_counts: list[int] = []
    for row in rows:
        messages = row["messages"]
        message_counts.append(len(messages))
        tool_calls += assistant_tool_call_count(messages)
        task_modes[row["metadata"]["task_mode"]] += 1
        projects.add(row["metadata"]["project"])
        for message in messages:
            role_counts[message["role"]] += 1

    summary = {
        "rows": len(rows),
        "output": str(args.output),
        "projects": len(projects),
        "task_modes": dict(task_modes),
        "role_counts": dict(role_counts),
        "assistant_tool_calls": tool_calls,
        "messages_per_trajectory": {
            "min": min(message_counts),
            "mean": round(statistics.mean(message_counts), 2),
            "max": max(message_counts),
        },
    }

    if not args.skip_token_stats:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        lengths: list[int] = []
        for row in rows:
            text = tokenizer.apply_chat_template(
                row["messages"],
                tools=BASH_TOOLS,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            lengths.append(len(tokenizer(text, add_special_tokens=False).input_ids))
        summary["gemma4_full_trajectory_tokens"] = {
            "min": min(lengths),
            "mean": round(statistics.mean(lengths), 2),
            "p50": percentile(lengths, 50),
            "p75": percentile(lengths, 75),
            "p90": percentile(lengths, 90),
            "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99),
            "max": max(lengths),
            "over_32768": sum(length > 32768 for length in lengths),
            "over_65536": sum(length > 65536 for length in lengths),
        }

    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
