#!/usr/bin/env python3
"""Convert text-command trajectories into harmony TOOL-CALL format.

Each row is a multi-turn trajectory:
    system, user(task), [assistant(action), user(observation)]*

Transform:
  - system                : kept as-is
  - user[1] (the task)    : kept as-is (role user)
  - assistant             : the ```mswea_bash_command\n<cmd>\n``` text block is parsed out;
                            content -> "" and a tool_calls=[{function:{name:"bash",
                            arguments:{"command":<cmd>}}}] is attached. The THOUGHT prose is
                            DROPPED (commands-only, matching the de-chain recipe).
  - user (observation)    : any user message AFTER the first user turn is a tool result;
                            converted to role "tool" (harmony commentary/tool channel) with a
                            tool_call_id matching the preceding assistant tool call.

Usage: python build_toolfmt_data.py [input.jsonl] [output.jsonl]
"""
import json
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/trajectories_dechained.jsonl")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/trajectories_dechained_toolfmt.jsonl")

CMD_RE = re.compile(r"```mswea_bash_command\n(.*?)\n```", re.S)


def parse_cmd(content: str) -> str:
    m = CMD_RE.search(content)
    if not m:
        raise ValueError("assistant turn has no mswea_bash_command block")
    return m.group(1)


def convert_row(row: dict, row_idx: int) -> dict:
    msgs = row["messages"]
    out = []
    seen_first_user = False
    call_counter = 0
    last_call_id = None
    for m in msgs:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": m["content"]})
        elif role == "user":
            if not seen_first_user:
                # the task prompt
                seen_first_user = True
                out.append({"role": "user", "content": m["content"]})
            else:
                # an observation / tool result
                out.append({
                    "role": "tool",
                    "content": m["content"],
                    "tool_call_id": last_call_id if last_call_id is not None else "call_unknown",
                })
        elif role == "assistant":
            cmd = parse_cmd(m["content"])
            call_counter += 1
            call_id = f"call_{row_idx}_{call_counter}"
            last_call_id = call_id
            out.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "id": call_id,
                    "function": {
                        "name": "bash",
                        # DICT (not a json string): the harmony template does
                        # `tool_call.arguments|tojson`, and is_terminal_submit_action
                        # expects isinstance(arguments, dict).
                        "arguments": {"command": cmd},
                    },
                }],
            })
        else:
            raise ValueError(f"unexpected role {role}")
    new_row = dict(row)
    new_row["messages"] = out
    return new_row


def main() -> None:
    rows = [json.loads(l) for l in SRC.open()]
    converted = [convert_row(r, i) for i, r in enumerate(rows)]

    # --- verification ---
    total_asst = total_tool = bad = 0
    for r in converted:
        for m in r["messages"]:
            if m["role"] == "assistant":
                total_asst += 1
                if not m.get("tool_calls"):
                    bad += 1
                if "mswea_bash_command" in (m.get("content") or ""):
                    bad += 1
            elif m["role"] == "tool":
                total_tool += 1
    if bad:
        print(f"FATAL: {bad} malformed assistant turns", file=sys.stderr)
        sys.exit(1)

    with DST.open("w") as f:
        for r in converted:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(converted)} rows to {DST}")
    print(f"assistant tool-call turns: {total_asst}, tool(observation) turns: {total_tool}")

    # dump the first converted trajectory structure
    print("\n=== first converted trajectory (first 6 msgs) ===")
    for m in converted[0]["messages"][:6]:
        show = dict(m)
        if isinstance(show.get("content"), str) and len(show["content"]) > 120:
            show["content"] = show["content"][:120] + "...<trunc>"
        print(json.dumps(show))


if __name__ == "__main__":
    main()
