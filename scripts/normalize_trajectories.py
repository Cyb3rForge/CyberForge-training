"""Normalize teacher trajectories for retraining so the data matches the SEC-bench
mini-swe-agent EVAL environment (and is shorter). Transforms, all train/eval-aligning:

  1. Combine both HF files (gemma4_agent_trajectories + workflow_gpt_trajectories).
  2. Commands: rewrite `rg ...` -> `grep -rn ...` (ripgrep is absent in the eval container).
  3. Observations (tool messages): cap at 10k chars (head 5k + <elided> + tail 5k) -- the
     same policy mini-swe-agent applies; shrinks 40k-token trajectories and aligns the
     observation distribution with eval (eval must use the same cap; see notes).
  4. Task (user) messages: drop the `rg` mention and the /workspace test-harness +
     /usr/local/bin/secb lines so the PROMPT no longer induces missing tools/paths
     (matches the current eval patch.j2 / poc-san.j2).
  5. Optional --path-map "old=new,..." to rewrite residual /workspace,/testcase command
     paths once the eval-container layout is known (left as a knob; default: just count them).

Writes data/processed/normalized_trajectories.jsonl and prints before/after stats.
The system prompt is intentionally left unchanged -- align the EVAL harness to it instead.
"""
import json, re, argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "AmL-hug/vuljector-secbench-trajectories"
# 2026-06-25: repo now ships a single combined file (the old gemma4_/workflow_ files were
# replaced by all_trajectories.jsonl, 1083 trajectories). Override with --files if needed.
FILES = ["all_trajectories.jsonl"]
OBS_CAP = 6000           # chars; >cap => head 3k + elided marker + tail 3k (tighter: the
HEAD = TAIL = 3000       # model re-reads files/secb each step -> capping curbs context blow-up

def cap_observation(text: str) -> tuple[str, int]:
    if len(text) <= OBS_CAP:
        return text, 0
    elided = len(text) - HEAD - TAIL
    return f"{text[:HEAD]}\n...[{elided} characters elided]...\n{text[-TAIL:]}", elided

RG = re.compile(r'(^|[\s;&|(])rg\b')          # `rg` as a command token
# ripgrep is regex-by-default so `rg "A|B"` matches the alternation; plain grep needs -E.
# The old `rg -> grep -rn` (no -E) taught the model `grep "A|B"` which matches `|` LITERALLY
# -> recon finds nothing -> it loops/never edits. Add -E here and to any other bare grep.
GREP = re.compile(r'(^|[\s;&|(])grep\b(?!\s+-\w*[EP])')   # grep not already using -E/-P
# Canonicalize alternation to EXTENDED regex: under -E, `|` is alternation but `\|` is a
# LITERAL pipe. The teacher wrote `\|` for BASIC grep (correct there); once we force -E that
# `\|` matches literally -> broken alternation. So unescape `\|`->`|` inside grep patterns.
GREP_PAT = re.compile(r'((?:^|[\s;&|(])grep(?:\s+-\S+)*\s+)(["\'])(.*?)\2')
def _unescape_alt(m):
    return f'{m.group(1)}{m.group(2)}{m.group(3).replace(chr(92) + "|", "|")}{m.group(2)}'
def fix_command(c: str, pathmap: dict, keep_rg: bool = False) -> str:
    if not keep_rg:
        # ripgrep is absent in the grep-only eval env: rewrite rg->grep -rnE and canonicalize
        # alternation. With --keep-rg (ripgrep mounted into the eval containers) leave rg as-is.
        c = RG.sub(r'\1grep -rnE', c)
        c = GREP.sub(r'\1grep -E', c)
        c = GREP_PAT.sub(_unescape_alt, c)
    # env-align teacher paths to the secb.eval container: the teacher's test runner is
    # `secb repro` here, and its PoC/exploit files live in /testcase.
    c = re.sub(r'(?:bash\s+|sh\s+|\./)?/workspace/run/unit_tests/test\.sh\b[^\s|&;]*', 'secb repro', c)
    c = c.replace('/workspace/run/attempt/artifacts/exploit_files', '/testcase')
    c = c.replace('/workspace/run/attempt/artifacts', '/testcase')
    for old, new in pathmap.items():
        c = c.replace(old, new)
    return c

def clean_task(t: str, keep_rg: bool = False) -> str:
    if not keep_rg:   # with rg mounted into the eval containers, keep the rg mention
        t = t.replace("`find`, `grep`, and `rg`", "`find` and `grep`")
        t = t.replace("find`, `grep` and `rg`", "find` and `grep`")
    t = re.sub(r'^.*/workspace/run/unit_tests/test\.sh.*$\n?', '', t, flags=re.M)
    t = re.sub(r'^.*Inspect `?/usr/local/bin/secb`?.*$\n?', '', t, flags=re.M)
    return t

def get_cmds(msg):
    """Yield (setter, command) so we can rewrite in place across str/dict arg shapes."""
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        a = fn.get("arguments")
        if isinstance(a, str):
            try: d = json.loads(a)
            except Exception: d = None
            if isinstance(d, dict) and "command" in d:
                def setter(v, _d=d, _fn=fn): _d["command"] = v; _fn["arguments"] = json.dumps(_d)
                yield setter, d["command"]
            else:
                def setter(v, _fn=fn): _fn["arguments"] = v
                yield setter, a
        elif isinstance(a, dict) and "command" in a:
            def setter(v, _a=a): _a["command"] = v
            yield setter, a["command"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/normalized_trajectories.jsonl")
    ap.add_argument("--path-map", default="", help='comma list old=new for /workspace,/testcase rewrites')
    ap.add_argument("--keep-rg", action="store_true", help="keep rg commands as-is (ripgrep is mounted into eval containers); skip the rg->grep rewrite")
    args = ap.parse_args()
    pathmap = dict(kv.split("=", 1) for kv in args.path_map.split(",") if "=" in kv)

    rows = []
    for f in FILES:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        rows += [json.loads(l) for l in open(p) if l.strip()]
    print(f"loaded {len(rows)} trajectories from {len(FILES)} files")

    st = {"rg_before":0,"rg_after":0,"ws_before":0,"ws_after":0,"tc_before":0,"tc_after":0,
          "obs_capped":0,"chars_before":0,"chars_after":0,"task_cleaned":0}
    def count(c, key):
        st[f"rg_{key}"]+=len(RG.findall(c)); st[f"ws_{key}"]+=c.count("/workspace"); st[f"tc_{key}"]+=c.count("/testcase")

    for r in rows:
        for m in r.get("messages", []):
            role = m.get("role")
            if role == "assistant":
                for setter, cmd in get_cmds(m):
                    count(cmd, "before"); st["chars_before"]+=len(cmd)
                    nc = fix_command(cmd, pathmap, args.keep_rg)
                    count(nc, "after"); st["chars_after"]+=len(nc); setter(nc)
            elif role == "tool":
                c = m.get("content")
                if isinstance(c, str):
                    st["chars_before"]+=len(c)
                    capped, elided = cap_observation(c)
                    if elided: st["obs_capped"]+=1
                    st["chars_after"]+=len(capped); m["content"]=capped
            elif role == "user":
                c = m.get("content")
                if isinstance(c, str):
                    nc = clean_task(c, args.keep_rg)
                    if nc != c: st["task_cleaned"]+=1
                    m["content"]=nc

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        for r in rows: fh.write(json.dumps(r)+"\n")
    print(f"wrote {len(rows)} normalized trajectories -> {outp}")
    print(f"  rg in commands:        {st['rg_before']:5d} -> {st['rg_after']}")
    print(f"  /workspace refs:       {st['ws_before']:5d} -> {st['ws_after']}  (residual; need eval path map)")
    print(f"  /testcase refs:        {st['tc_before']:5d} -> {st['tc_after']}")
    print(f"  observations capped:   {st['obs_capped']}")
    print(f"  task messages cleaned: {st['task_cleaned']}")
    print(f"  total text chars:      {st['chars_before']:,} -> {st['chars_after']:,}  ({(1-st['chars_after']/max(1,st['chars_before']))*100:.0f}% smaller)")

if __name__ == "__main__":
    main()
