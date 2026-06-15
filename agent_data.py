from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable


BASH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


TOOL_CALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
GEMMA_MODEL_TURN = "<|turn>model\n"
GEMMA_PENDING_TOOL_RESPONSE = "<|tool_response>"
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    # KEY=value / KEY: value for any identifier containing api_key/token/secret/
    # password (e.g. AZURE_API_KEY, OPENAI_API_KEY, GITHUB_TOKEN). The previous
    # value class `[^\"'\s,}\\]]+` had a misplaced `]` so it required a literal
    # `]` after the value and never matched real secrets — fixed here.
    (
        re.compile(
            r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|token|password)[A-Za-z0-9_]*)"
            r"([\"']?\s*[:=]\s*[\"']?)"
            r"([^\s\"',}\]]+)"
        ),
        r"\1\2[REDACTED_SECRET]",
    ),
]


def redact_text(value: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_obj(item) for key, item in value.items()}
    return value


def load_index(raw_dir: Path) -> list[dict[str, Any]]:
    index_path = raw_dir / "data" / "index.jsonl"
    with index_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_archive(raw_dir: Path, archive_path: str) -> dict[str, Any]:
    archive = raw_dir / archive_path
    with tarfile.open(archive, "r:gz") as tf:
        trajectory = json.load(tf.extractfile("trajectory.json"))  # type: ignore[arg-type]
        task_result = json.load(tf.extractfile("task_result.json"))  # type: ignore[arg-type]
    return {"trajectory": trajectory, "task_result": task_result}


def _parse_tool_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return redact_obj(arguments)
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return redact_text(arguments)
    return redact_obj(parsed)


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "exit":
            continue
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        item: dict[str, Any] = {
            "role": role,
            "content": redact_obj(message.get("content")),
        }

        # Preserve the teacher's natural-language reasoning. The Azure Responses
        # API stores it on the assistant message at extra.reasoning_summary; the
        # Gemma 4 chat template renders a `reasoning` field as a <|channel>thought
        # block for tool-call turns after the last user message (exactly the turns
        # the trainer supervises), so it lands in the loss target. Without this the
        # reasoning is silently dropped at normalization (the original SFT bug).
        if role == "assistant":
            reasoning = (message.get("extra") or {}).get("reasoning_summary")
            if reasoning:
                item["reasoning"] = redact_text(reasoning)

        if role == "assistant" and message.get("tool_calls"):
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = dict(tool_call.get("function") or {})
                function["arguments"] = _parse_tool_arguments(function.get("arguments"))
                tool_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type", "function"),
                        "function": redact_obj(function),
                    }
                )
            item["tool_calls"] = tool_calls

        if role == "tool":
            item["tool_call_id"] = message.get("tool_call_id")

        normalized.append(item)

    return normalized


def iter_normalized_trajectories(raw_dir: Path) -> Iterable[dict[str, Any]]:
    for row in load_index(raw_dir):
        bundle = load_archive(raw_dir, row["archive_path"])
        trajectory = bundle["trajectory"]
        task_result = bundle["task_result"]
        quality = task_result.get("quality_metrics") or {}
        yield {
            "id": row["run_id"],
            "messages": normalize_messages(trajectory.get("messages", [])),
            "metadata": {
                "archive_path": row["archive_path"],
                "project": row.get("project"),
                "task_mode": row.get("task_mode"),
                "vulnerability_id": row.get("vulnerability_id"),
                "validation_status": row.get("validation_status"),
                "teacher_regime": row.get("teacher_regime"),
                "include_hints": row.get("include_hints"),
                "api_calls": quality.get("api_calls"),
                "total_steps": quality.get("total_steps"),
            },
        }


def load_processed_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gemma_format_argument(argument: Any, escape_keys: bool = True) -> str:
    if isinstance(argument, str):
        return f'<|"|>{argument}<|"|>'
    if isinstance(argument, bool):
        return "true" if argument else "false"
    if isinstance(argument, dict):
        parts = []
        for key in sorted(argument):
            rendered_key = f'<|"|>{key}<|"|>' if escape_keys else str(key)
            parts.append(f"{rendered_key}:{gemma_format_argument(argument[key], escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(argument, (list, tuple)):
        return "[" + ",".join(gemma_format_argument(item, escape_keys) for item in argument) + "]"
    if argument is None:
        return "None"
    return str(argument)


def render_gemma_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call["function"]
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        inner = ",".join(
            f"{key}:{gemma_format_argument(arguments[key], escape_keys=False)}"
            for key in sorted(arguments)
        )
    elif isinstance(arguments, str):
        inner = arguments
    else:
        inner = str(arguments)
    return f"<|tool_call>call:{function['name']}{{{inner}}}<tool_call|>"


def assistant_tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(message.get("tool_calls") or [])
        for message in messages
        if message.get("role") == "assistant"
    )


def strip_pending_tool_response(text: str) -> str:
    if text.endswith(GEMMA_PENDING_TOOL_RESPONSE):
        return text[: -len(GEMMA_PENDING_TOOL_RESPONSE)]
    return text


def find_last_model_target_span(text: str) -> tuple[int, int]:
    start = text.rfind(GEMMA_MODEL_TURN)
    if start < 0:
        raise ValueError("Could not find Gemma model turn marker in rendered chat")
    target_start = start + len(GEMMA_MODEL_TURN)
    target_end = len(text)
    if target_start >= target_end:
        raise ValueError("Rendered assistant turn is empty")
    return target_start, target_end


def find_last_assistant_action_span(
    text: str,
    tool_call_count: int,
    include_content_before_tool_call: bool = False,
) -> tuple[int, int]:
    matches = list(TOOL_CALL_RE.finditer(text))
    if tool_call_count <= 0 or len(matches) < tool_call_count:
        return find_last_model_target_span(text)

    first_target_call = matches[-tool_call_count]
    last_target_call = matches[-1]
    if not include_content_before_tool_call:
        return first_target_call.start(), last_target_call.end()

    model_boundary = text.rfind(GEMMA_MODEL_TURN, 0, first_target_call.start())
    tool_boundary = text.rfind("<tool_response|>", 0, first_target_call.start())
    if tool_boundary >= 0 and tool_boundary > model_boundary:
        target_start = tool_boundary + len("<tool_response|>")
    elif model_boundary >= 0:
        target_start = model_boundary + len(GEMMA_MODEL_TURN)
    else:
        target_start = first_target_call.start()
    return target_start, last_target_call.end()
