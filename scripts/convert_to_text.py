"""Convert tool-calling teacher trajectories to the TEXT-mode protocol the base model is
strong at (mini-swe-agent text mode: THOUGHT + ```mswea_bash_command``` block, observations
as role:user <returncode>/<output>).

Usage: python convert_to_text.py [input.jsonl] [output.jsonl]"""
import json

# Verbatim text-mode system prompt (from the base text-mode eval, len=544).
TEXT_SYS = (
    "You are a helpful assistant that can interact with a computer.\n\n"
    "Your response must contain exactly ONE bash code block with ONE command "
    "(or commands connected with && or ||).\n"
    "Include a THOUGHT section before your command where you explain your reasoning process.\n"
    "Format your response as shown in <format_example>.\n\n"
    "<format_example>\nYour reasoning and analysis here. Explain why you want to perform the action.\n\n"
    "```mswea_bash_command\nyour_command_here\n```\n</format_example>\n\n"
    "Failure to follow these rules will cause your response to be rejected."
)

def cmd_of(tc):
    a = (tc.get("function") or {}).get("arguments")
    if isinstance(a, str):
        try: a = json.loads(a)
        except Exception: return a
    return a.get("command", "") if isinstance(a, dict) else str(a)

def conv_obs(content):
    if isinstance(content, str):
        try:
            d = json.loads(content)
            return "<returncode>%s</returncode>\n<output>\n%s\n</output>" % (d.get("returncode", 0), d.get("output", ""))
        except Exception:
            return "<returncode>0</returncode>\n<output>\n%s\n</output>" % content
    return "<returncode>0</returncode>\n<output>\n%s\n</output>" % str(content)

import sys
inp = sys.argv[1] if len(sys.argv) > 1 else "data/processed/normalized_trajectories_v3.jsonl"
out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/normalized_trajectories_v3_text.jsonl"
ntraj = nasst = 0
with open(out, "w") as fo:
    for line in open(inp):
        d = json.loads(line)
        new = []
        for m in d.get("messages", []):
            r = m.get("role")
            if r == "system":
                new.append({"role": "system", "content": TEXT_SYS})
            elif r == "user":
                new.append({"role": "user", "content": m.get("content", "")})
            elif r == "assistant":
                tcs = m.get("tool_calls") or []
                if not tcs:
                    if m.get("content"):
                        new.append({"role": "assistant", "content": m["content"]})
                    continue
                cmd = cmd_of(tcs[0])
                reasoning = (m.get("reasoning") or "").strip()
                content = (("THOUGHT: " + reasoning + "\n\n") if reasoning else "") + "```mswea_bash_command\n" + cmd + "\n```"
                new.append({"role": "assistant", "content": content})
                nasst += 1
            elif r == "tool":
                new.append({"role": "user", "content": conv_obs(m.get("content"))})
        d["messages"] = new
        fo.write(json.dumps(d) + "\n")
        ntraj += 1
print("converted %d trajectories, %d assistant text-turns -> %s" % (ntraj, nasst, out))
print("\n=== validate: first trajectory, first 6 msgs ===")
d = json.loads(open(out).readline())
for m in d["messages"][:6]:
    print("---", m["role"], "len", len(m["content"]))
    print(repr(m["content"][:220]))
