#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer


def read_token(path: Path) -> str:
    return path.read_text().strip()


def ensure_tokenizer(checkpoint_dir: Path, tokenizer_id: str) -> None:
    required = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    if all((checkpoint_dir / name).exists() for name in required):
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(checkpoint_dir)


def ensure_readme(checkpoint_dir: Path, repo_id: str, base_model: str) -> None:
    readme = checkpoint_dir / "README.md"
    if readme.exists():
        return

    readme.write_text(
        f"""---
base_model: {base_model}
library_name: transformers
tags:
- gemma4
- mini-swe-agent
- agent
- full-finetune
---

# {repo_id.split('/')[-1]}

Full fine-tune of `{base_model}` on Mini-SWE-Agent-style security benchmark trajectories.

## Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, dtype="auto", device_map="auto")
```

## vLLM

```bash
vllm serve {repo_id} --dtype bfloat16
```

The model was trained on assistant action/tool-call turns from agent trajectories.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default="google/gemma-4-31B-it")
    parser.add_argument("--base-model", default="google/gemma-4-31B-it")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)

    ensure_tokenizer(checkpoint_dir, args.tokenizer_id)
    ensure_readme(checkpoint_dir, args.repo_id, args.base_model)

    token = read_token(args.token_file)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.update_repo_settings(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        token=token,
    )

    print(f"Uploading {checkpoint_dir} to root of {args.repo_id}", flush=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint_dir),
        path_in_repo="",
        commit_message="Publish final checkpoint as root model",
        token=token,
    )
    print("root upload complete", flush=True)


if __name__ == "__main__":
    main()
