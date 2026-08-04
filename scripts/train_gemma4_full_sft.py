#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import timedelta

# This node's `hostname` returns 'dmidecode', which does not resolve to an IP; torchrun
# then sets MASTER_ADDR to it and every env:// rendezvous times out. All ranks are on one
# node here, so force loopback before any torch/accelerate/dist code runs.
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Bind each rank to its own GPU BEFORE any distributed/NCCL init. Without this, NCCL
# "guesses device ID based on global rank" and on some multi-GPU node topologies mismaps
# ranks to GPUs, crashing the first FSDP all-gather with SIGSEGV.
if int(os.environ.get("WORLD_SIZE", "1")) > 1 and torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

import yaml
from torch.utils.data import Dataset, Subset
from transformers import TrainerCallback

# agent_data lives at the repo root (one level above scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_data import BASH_TOOLS  # noqa: E402

# Disable cuDNN's fused SDPA backend: it crashed mid-run with
# "mha_graph.execute(...).is_good() == false" on certain sequence shapes. Force PyTorch
# SDPA onto the reliable flash / mem-efficient kernels instead.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Make checkpoint saves resilient to transient network-filesystem glitches: a safetensors
# shard write can return ENODEV mid-save on a shared scratch filesystem. Retry the shard
# write with backoff so a transient blip doesn't kill a multi-hour run. Patch both the
# safetensors symbol and the one transformers bound at import (save_pretrained -> safe_save_file).
import time as _time  # noqa: E402
import safetensors.torch as _sft  # noqa: E402
import transformers.modeling_utils as _mu  # noqa: E402

_orig_safe_save_file = _sft.save_file


def _save_file_retry(*args, **kwargs):
    for _attempt in range(5):
        try:
            return _orig_safe_save_file(*args, **kwargs)
        except Exception as _exc:  # noqa: BLE001 -- transient FS errors (ENODEV etc.)
            if _attempt == 4:
                raise
            print(f"[save-retry] safetensors write failed ({_exc}); "
                  f"retry {_attempt + 1}/4 in {5 * (_attempt + 1)}s", flush=True)
            _time.sleep(5 * (_attempt + 1))


_sft.save_file = _save_file_retry
_mu.safe_save_file = _save_file_retry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    return parser.parse_args()


def _expand_string(value: str, root: dict[str, Any]) -> str:
    pattern = re.compile(r"\$\{([^}]+)\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in root:
            return str(root[key])
        return os.environ.get(key, match.group(0))

    return pattern.sub(repl, value)


def expand_config(value: Any, root: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: expand_config(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config(v, root) for v in value]
    if isinstance(value, str):
        return _expand_string(value, root)
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    for _ in range(3):
        config = expand_config(config, config)
    return config


def log_phase(message: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    print(f"[{now} rank={rank}] {message}", flush=True)


def configure_early_fsdp_env(fsdp_cfg: dict[str, Any]) -> None:
    if not fsdp_cfg:
        return
    os.environ.setdefault("ACCELERATE_USE_FSDP", "true")
    if fsdp_cfg.get("cpu_ram_efficient_loading"):
        os.environ.setdefault("FSDP_CPU_RAM_EFFICIENT_LOADING", "true")
        os.environ.setdefault("FSDP_SYNC_MODULE_STATES", "true")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        # This node's `hostname` returns 'dmidecode' which does not resolve to an IP, so
        # torchrun sets MASTER_ADDR to it and env:// rendezvous times out. Force loopback
        # (single-node DDP/FSDP): all ranks are local, so 127.0.0.1 is always correct.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        log_phase("initializing process group before from_pretrained for FSDP efficient loading")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=1),
        )


def load_processed_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_model_id(model_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id.strip("/"))
    if len(cleaned) > 120:
        digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
        cleaned = f"{cleaned[:100]}_{digest}"
    return cleaned


def to_list(ids: Any) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return ids.detach().cpu().view(-1).tolist()
    try:
        if "input_ids" in ids:
            return to_list(ids["input_ids"])
    except (TypeError, KeyError):
        pass
    if hasattr(ids, "ids"):
        return list(ids.ids)
    if isinstance(ids, (list, tuple)):
        flattened: list[int] = []
        for item in ids:
            if isinstance(item, int):
                flattened.append(item)
            else:
                flattened.extend(to_list(item))
        return flattened
    return list(ids)


def is_terminal_submit_action(message: dict[str, Any]) -> bool:
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        command = args.get("command") if isinstance(args, dict) else None
        if isinstance(command, str) and command.strip() == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            return True
    return False


# --- Assistant-token masking via the chat template -------------------------
# Best-practice SFT masks loss to exactly the assistant-generated tokens using
# the `{% generation %}` / `{% endgeneration %}` chat-template markers and
# `apply_chat_template(return_assistant_tokens_mask=True)` (see HF/TRL
# assistant_only_loss). Gemma 4's shipped template has no such markers, so we
# inject them into a *training-only* copy of the template. The markers are
# placed to capture: the reasoning thought block + tool-call(s) + any assistant
# text content, while EXCLUDING role headers, the (forward-scanned) tool
# responses, and the pending `<|tool_response>` opener the template appends
# after a tool call. The base serving template is left untouched.
_GEN_OPEN_REASONING_TOOLCALL = "    {#- Render reasoning/reasoning_content as thinking channel -#}"
_GEN_CLOSE_TOOLCALL = (
    "                {%- set ns.prev_message_type = 'tool_call' -%}\n"
    "            {%- endif -%}\n"
)
_GEN_CONTENT = "            {{- captured_content -}}\n"


def build_training_chat_template(base_template: str) -> str:
    # Non-Gemma templates (e.g. gpt-oss harmony) can supply a pre-verified training template
    # via SECB_TRAIN_CHAT_TEMPLATE. It must already contain {% generation %} markers whose
    # mask was verified to cover exactly the assistant turn.
    _ext = __import__("os").environ.get("SECB_TRAIN_CHAT_TEMPLATE")
    if _ext:
        _t = open(_ext).read()
        if "generation" not in _t:
            raise ValueError(f"SECB_TRAIN_CHAT_TEMPLATE={_ext} has no {{% generation %}} markers")
        print(f"[template] using external training chat template: {_ext}", flush=True)
        return _t

    """Return a copy of the Gemma 4 chat template with {% generation %} markers
    around the assistant-generated spans, for return_assistant_tokens_mask."""
    if "{% generation %}" in base_template:
        return base_template

    def require(anchor: str, label: str) -> None:
        count = base_template.count(anchor)
        if count != 1:
            raise ValueError(
                f"Cannot add assistant-mask markers: anchor '{label}' found {count} times "
                "(expected 1). The Gemma 4 chat template differs from the verified version."
            )

    require(_GEN_OPEN_REASONING_TOOLCALL, "reasoning/tool_call open")
    require(_GEN_CLOSE_TOOLCALL, "tool_call close")
    require(_GEN_CONTENT, "content")

    # Span 1: reasoning + tool_calls. Only assistant ('model') messages emit
    # content here (reasoning requires tool_calls; tool_calls require the key),
    # so wrapping unconditionally yields an empty span for other roles.
    template = base_template.replace(
        _GEN_OPEN_REASONING_TOOLCALL,
        "    {% generation %}\n" + _GEN_OPEN_REASONING_TOOLCALL,
    )
    template = template.replace(
        _GEN_CLOSE_TOOLCALL,
        _GEN_CLOSE_TOOLCALL + "            {% endgeneration %}\n",
    )
    # Span 2: assistant text content. captured_content is emitted for every
    # role, so guard by role and keep open+close inside the same if-branch
    # (Jinja requires generation blocks balanced within a block scope).
    template = template.replace(
        _GEN_CONTENT,
        "            {%- if role == 'model' -%}{% generation %}{{- captured_content -}}{% endgeneration %}"
        "{%- else -%}{{- captured_content -}}{%- endif -%}\n",
    )
    return template


class MiniSweTurnDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        cache_dir: Path,
        cache_model_id: str,
        max_seq_length: int,
        max_examples: int = 0,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_examples = max_examples
        # Training-only template with {% generation %} markers for exact
        # assistant-token masking (does not affect the saved/serving template).
        base_template = tokenizer.chat_template or tokenizer.get_chat_template()
        self.train_template = build_training_chat_template(base_template)
        # TOOL-MODE data (harmony tool_calls): the eval sends the bash tool definition, so
        # the training prompt must render the same `# Tools` namespace in the developer
        # message. Gate on SECB_TRAIN_TOOLS so the TEXT-mode path (which sends no tools)
        # is unchanged. Value "bash"/"1"/"true" -> pass BASH_TOOLS.
        self.train_tools = (
            BASH_TOOLS
            if os.environ.get("SECB_TRAIN_TOOLS", "").strip().lower() in {"bash", "1", "true", "yes"}
            else None
        )
        if self.train_tools is not None:
            log_phase("SECB_TRAIN_TOOLS set -> rendering BASH_TOOLS in training prompts (tool mode)")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = (
            cache_dir
            / f"mini_swe_turns_genmask_headtail_{safe_model_id(cache_model_id)}_{max_seq_length}_{max_examples}.pt"
        )
        self.examples = self._load_or_build()
        if not self.examples:
            raise ValueError("No supervised assistant-action examples were built")

    def _load_or_build(self) -> list[dict[str, torch.Tensor]]:
        if self.cache_path.exists():
            try:
                log_phase(f"loading tokenized cache {self.cache_path}")
                cached = torch.load(self.cache_path, map_location="cpu")
                examples = [self._coerce_example(ex) for ex in cached]
                if examples:
                    log_phase(f"loaded {len(examples)} cached turn examples")
                    return examples
            except Exception as exc:
                log_phase(f"cache load failed ({exc}); rebuilding")

        examples = self._build_examples()
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            tmp = self.cache_path.with_suffix(".tmp")
            torch.save(examples, tmp)
            tmp.replace(self.cache_path)
            log_phase(f"saved tokenized cache {self.cache_path}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return examples

    def _coerce_example(self, example: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.as_tensor(example["input_ids"], dtype=torch.long),
            "labels": torch.as_tensor(example["labels"], dtype=torch.long),
        }

    def _encode_with_mask(self, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
        """Encode messages and return (input_ids, assistant_mask) where the mask
        is 1 for tokens generated by the assistant (reasoning + tool calls +
        text content) and 0 for everything else, via the {% generation %} markers."""
        enc = self.tokenizer.apply_chat_template(
            messages,
            # TEXT-MODE data: assistant content is a literal ```mswea_bash_command``` block,
            # and the eval harness (LitellmTextbasedModel) sends NO tools -> tools=None.
            # TOOL-MODE data: the eval sends the bash tool definition, so we render it too
            # (self.train_tools = BASH_TOOLS when SECB_TRAIN_TOOLS is set). Matching the
            # `# Tools` preamble to eval avoids a structural train/eval mismatch.
            tools=self.train_tools,
            chat_template=self.train_template,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        return to_list(enc["input_ids"]), [int(m) for m in enc["assistant_masks"]]

    def _build_examples(self) -> list[dict[str, torch.Tensor]]:
        examples: list[dict[str, torch.Tensor]] = []
        for row in self.rows:
            messages = row.get("messages") or []
            # HEAD = <bos> + system + first user message (task / bug description +
            # sanitizer report). On over-length examples we preserve this head and
            # middle-truncate (see below): edit turns are the longest examples (they
            # come late, after recon), so plain left-truncation stripped the task
            # framing from ~78% of them -- training the model to emit a fix with no
            # statement of what to fix and no action rules, off-distribution from the
            # fully-framed context it always sees at eval. Head is ~2k tokens (p90 3.5k).
            head_end = 1
            while head_end < len(messages) and messages[head_end].get("role") != "user":
                head_end += 1
            try:
                head_ids, _ = self._encode_with_mask(messages[: head_end + 1])
                head_len = len(head_ids)
            except Exception:
                head_len = 0
            for idx, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                # Supervise tool-call turns (tool-calling data) OR plain-content turns
                # (TEXT-mode data: assistant content = THOUGHT + ```mswea_bash_command``` block).
                # The {% generation %} content span already masks assistant content, so the
                # target-position logic below works for both shapes.
                if not message.get("tool_calls") and not message.get("content"):
                    continue
                # NOTE: previously the terminal submit action (echo
                # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT) was skipped here. It is now
                # SUPERVISED so the model learns WHEN to stop -- at eval the model never
                # submitted (rambled instead). is_terminal_submit_action is kept for
                # reference but no longer excludes the turn.

                try:
                    _, prefix_mask = self._encode_with_mask(messages[:idx])
                    full_ids, full_mask = self._encode_with_mask(messages[: idx + 1])
                except Exception as exc:
                    log_phase(f"skipping example {row.get('id')} turn {idx}: chat template failed: {exc}")
                    continue

                # The current action's supervised tokens are the assistant tokens
                # in messages[:idx+1] that are not already present in messages[:idx]
                # -- i.e. the last (m_full - m_prefix) masked positions. This is the
                # reasoning thought + tool call (+ text content) for this turn, and
                # never the pending <|tool_response>/tool output (masked by the markers).
                masked_positions = [p for p, flag in enumerate(full_mask) if flag]
                num_target = len(masked_positions) - sum(prefix_mask)
                if num_target <= 0:
                    continue
                target_positions = masked_positions[-num_target:]

                # STOP-TOKEN FIX: the token immediately after the last supervised action token is
                # the turn/handoff stop (<|tool_response> id 50, <turn|> 106, or <eos> 1 -- all in
                # generation_config.eos_token_id). The old cut DROPPED it, so the model never learned
                # to STOP after a tool-call and over-generated at inference (finish_reason=length,
                # dozens of duplicate tool-calls, 0 edits). Supervise it so the model learns to stop.
                sp = target_positions[-1] + 1
                if sp < len(full_ids) and full_ids[sp] in (50, 106, 1):
                    target_positions = target_positions + [sp]

                # Cut context at the end of the action (now INCLUDING the supervised stop token)
                # and supervise the action tokens + that stop.
                input_ids = full_ids[: target_positions[-1] + 1]
                labels = [-100] * len(input_ids)
                for p in target_positions:
                    labels[p] = full_ids[p]

                if len(input_ids) > self.max_seq_length:
                    # MIDDLE-TRUNCATION: keep the HEAD (bos + system + task/bug
                    # description) and the TAIL (recent recon + the supervised action),
                    # eliding the middle. The action lives at the very end so it is always
                    # in the tail; the head is never supervised (its labels are already
                    # -100). This keeps the task framing the model has at eval -- plain
                    # left-truncation stripped it from ~78% of edit turns. Fall back to
                    # left-truncation only if the head alone is implausibly large.
                    if 0 < head_len < self.max_seq_length - 512:
                        tail_budget = self.max_seq_length - head_len
                        input_ids = input_ids[:head_len] + input_ids[-tail_budget:]
                        labels = ([-100] * head_len) + labels[-tail_budget:]
                    else:
                        input_ids = input_ids[-self.max_seq_length :]
                        labels = labels[-self.max_seq_length :]
                        bos = self.tokenizer.bos_token_id
                        if bos is not None and input_ids[0] != bos:
                            input_ids[0] = bos
                            labels[0] = -100

                if all(label == -100 for label in labels):
                    continue

                examples.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )
                if self.max_examples and len(examples) >= self.max_examples:
                    log_phase(f"built {len(examples)} turn examples (max_examples reached)")
                    return examples

        log_phase(f"built {len(examples)} turn examples")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


class DataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(feature["input_ids"].numel() for feature in features)
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            ids = feature["input_ids"]
            labs = feature["labels"]
            pad_len = max_len - ids.numel()
            input_ids.append(torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id))
            labels.append(torch.nn.functional.pad(labs, (0, pad_len), value=-100))
            attention_mask.append(torch.nn.functional.pad(torch.ones_like(ids), (0, pad_len), value=0))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }


class HubCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        repo_id: str,
        token_file: str | None,
        private: bool = True,
        repo_type: str = "model",
        fail_on_error: bool = False,
    ) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.fail_on_error = fail_on_error
        self.api = HfApi()
        self.token = None
        if token_file:
            self.token = Path(token_file).read_text(encoding="utf-8").strip()
        else:
            self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=True,
                token=self.token,
            )
        except Exception as exc:
            log_phase(f"Hugging Face repo setup failed: {exc}")
            if self.fail_on_error:
                raise

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not state.is_world_process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return control
        log_phase(f"uploading {checkpoint_dir.name} to Hugging Face repo {self.repo_id}")
        try:
            self.api.upload_folder(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                folder_path=str(checkpoint_dir),
                path_in_repo=checkpoint_dir.name,
                commit_message=f"Upload {checkpoint_dir.name}",
                token=self.token,
            )
            log_phase(f"uploaded {checkpoint_dir.name} to Hugging Face repo {self.repo_id}")
        except Exception as exc:
            log_phase(f"Hugging Face upload failed for {checkpoint_dir.name}: {exc}")
            if self.fail_on_error:
                raise
        return control


def split_dataset(dataset: Dataset, eval_fraction: float, seed: int) -> tuple[Dataset, Dataset | None]:
    n_total = len(dataset)
    if eval_fraction <= 0 or n_total < 2:
        return dataset, None
    n_eval = max(1, int(round(n_total * eval_fraction)))
    n_eval = min(n_eval, n_total - 1)
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[n_eval:]), Subset(dataset, indices[:n_eval])


def configure_train_scope(model: torch.nn.Module, scope: str) -> None:
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if scope == "all":
            should_train = True
        elif scope == "language_model":
            should_train = (
                name.startswith("language_model.")
                or name.startswith("model.language_model.")
                or name == "lm_head.weight"
                or name.endswith(".lm_head.weight")
            )
        elif scope == "language_model_no_embed":
            # Train the decoder layers + norms but FREEZE the (tied) input/output
            # embedding. Full-FT of embed_tokens/lm_head at lr 1e-5 collapsed the
            # output distribution into garbage; freezing it preserves the vocab mapping.
            is_lm = (
                name.startswith("language_model.")
                or name.startswith("model.language_model.")
            )
            is_embed = "embed_tokens" in name
            is_lmhead = name == "lm_head.weight" or name.endswith(".lm_head.weight")
            should_train = is_lm and not is_embed and not is_lmhead
        else:
            raise ValueError(f"Unknown train_scope: {scope}")
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(f"Train scope {scope}: trainable={trainable:,} frozen={frozen:,}", flush=True)


def verify_save_integrity(save_dir: str, base_dir: str, min_cos: float = 0.5) -> list[int]:
    """Detect decoder layers that a consolidated FSDP save failed to gather (they
    retain fresh-init weights). Compares each layer's input_layernorm.weight cosine
    to the base model; a gathered layer is ~1.0, an un-gathered one is ~0. Low memory
    (one small tensor per layer). Returns the list of corrupted layer indices."""
    from safetensors import safe_open

    def widx(d: str) -> dict:
        return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    try:
        wf, wb = widx(save_dir), widx(base_dir)
    except Exception as exc:
        log_phase(f"save-integrity check skipped (no consolidated index, e.g. LoRA adapter): {exc}")
        return []
    layers = sorted({int(m.group(1)) for k in wf for m in [re.search(r"layers\.(\d+)\.", k)] if m})
    bad: list[int] = []
    for layer in layers:
        key = next((k for k in wf if f"layers.{layer}.input_layernorm.weight" in k), None)
        if key is None or key not in wb:
            continue
        try:
            a = safe_open(os.path.join(save_dir, wf[key]), framework="pt").get_slice(key)[:].float()
            b = safe_open(os.path.join(base_dir, wb[key]), framework="pt").get_slice(key)[:].float()
            cos = float((a * b).sum() / ((a.norm() * b.norm()) + 1e-30))
        except Exception as exc:
            log_phase(f"save-integrity: could not check layer {layer}: {exc}")
            continue
        if cos < min_cos:
            bad.append(layer)
    return bad


# --- Memory-efficient fused-linear cross-entropy (Liger FLCE) ---------------
# At 32k seq / 256k vocab the HF forward materializes a [batch, 32768, 262144]
# logits tensor (~16.8 GB fwd + ~16.8 GB grad in bf16) and OOMs an 80 GB A100.
# Liger's fused-linear-CE fuses the lm_head projection with a chunked softmax
# cross-entropy and never forms that logits tensor. We install it by binding a
# replacement forward onto the (unwrapped) Gemma-4 causal-LM instance so the
# fused loss is computed INSIDE the model's forward. That placement is what
# makes it FSDP-correct: under `full_shard auto_wrap` (wrapping only the decoder
# layers) the tied lm_head/embed weight lives in the FSDP ROOT unit and is only
# all-gathered by the root's pre-forward hook -- i.e. exactly during this call.
# Computing FLCE here (rather than in a Trainer.compute_loss that reaches into
# `model.lm_head.weight` from outside the forward) means we read the full,
# gathered weight and gradients flow back into it correctly. Gated behind the
# `use_liger_flce` config flag; when unset the model is untouched.
def _liger_flce_causal_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    **kwargs,
):
    """Fused-linear-CE forward for Gemma4Unified(ForConditionalGeneration|ForCausalLM).

    When `labels` (or a pre-shifted `shift_labels`) is present the loss is computed
    via Liger's fused linear cross-entropy directly from the final hidden states +
    the lm_head weight -- the full logits tensor is never allocated. Otherwise it
    falls back to the standard materialized-logits path (generation / logit
    consumers), so non-loss callers behave exactly as before."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

    # These are consumed by the loss, not the text backbone.
    num_items_in_batch = kwargs.pop("num_items_in_batch", None)
    shift_labels = kwargs.pop("shift_labels", None)
    # Text-mode SFT never carries vision/audio inputs; drop any that slipped in so
    # the (multimodal) backbone runs its pure-text path.
    for _k in (
        "pixel_values", "pixel_values_videos", "input_features", "input_features_mask",
        "image_position_ids", "video_position_ids", "mm_token_type_ids",
    ):
        kwargs.pop(_k, None)

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    text_cfg = self.config.get_text_config()
    softcap = getattr(text_cfg, "final_logit_softcapping", None)

    loss = None
    logits = None
    if labels is not None or shift_labels is not None:
        # FUSED path: hidden_states are kept full-length; LigerForCausalLMLoss does
        # the causal shift (pad+slice), the -100 mask, the softcap and the fused
        # projection+CE internally. num_items_in_batch reproduces HF's grad-accum
        # sum-normalization exactly.
        loss = LigerForCausalLMLoss(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            hidden_size=text_cfg.hidden_size,
            final_logit_softcapping=softcap,
            num_items_in_batch=num_items_in_batch,
            shift_labels=shift_labels,
        )
    else:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )


def install_liger_flce(model: torch.nn.Module) -> None:
    """Bind the fused-linear-CE forward onto the underlying Gemma-4 causal LM.

    Handles the PEFT/LoRA wrapper (bind to the base causal LM that PeftModel calls)
    and validates the instance exposes the pieces the fused forward needs
    (`.model`, `.lm_head`, `config.get_text_config`)."""
    from types import MethodType

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    missing = [
        attr for attr in ("model", "lm_head", "config")
        if not hasattr(base, attr)
    ]
    if missing or not hasattr(base.config, "get_text_config"):
        raise ValueError(
            "use_liger_flce: could not locate the Gemma-4 causal-LM structure "
            f"(missing {missing or 'config.get_text_config'}) on {type(base).__name__}; "
            "FLCE not installed."
        )
    base.forward = MethodType(_liger_flce_causal_forward, base)
    log_phase(
        f"installed Liger fused-linear cross-entropy on {type(base).__name__} "
        f"(vocab={base.config.get_text_config().vocab_size}, "
        f"softcap={getattr(base.config.get_text_config(), 'final_logit_softcapping', None)})"
    )


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    dataset_cfg = config["dataset"]
    optimizer_cfg = config["optimizer"]
    wandb_cfg = config.get("wandb", {})

    if cli.max_steps is not None:
        config["max_steps"] = cli.max_steps
    if cli.max_train_examples is not None:
        dataset_cfg["max_train_examples"] = cli.max_train_examples
    if cli.output_dir is not None:
        config["output_dir"] = str(cli.output_dir)

    fsdp_cfg = dict(config.get("fsdp", {}))
    fsdp_strategy = fsdp_cfg.pop("strategy", "")
    configure_early_fsdp_env(fsdp_cfg)

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    use_liger_flce = bool(config.get("use_liger_flce", False))

    class LigerFlceTrainer(Trainer):
        """Trainer whose compute_loss routes the batch through the model's
        fused-linear-CE forward (installed by install_liger_flce), so the
        [batch, seq, vocab] logits tensor is never materialized. num_items_in_batch
        is forwarded so gradient-accumulation loss scaling matches stock HF exactly.
        The loss is computed inside model.forward -> FSDP-correct lm_head access."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if num_items_in_batch is not None:
                inputs = {**inputs, "num_items_in_batch": num_items_in_batch}
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

    tokenizer_id = config.get("tokenizer_id", config["model_id"])
    initial_model_id = config.get("initial_model_id", config["model_id"])

    log_phase(f"loading tokenizer from {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_phase("loading processed rows and building turn dataset")
    rows = load_processed_jsonl(Path(dataset_cfg["data_files"]))
    dataset = MiniSweTurnDataset(
        rows=rows,
        tokenizer=tokenizer,
        cache_dir=Path(dataset_cfg["cache_dir"]),
        # Include the data-file stem so the cache key is DATA-AWARE (different data ->
        # different cache). Without this, a text-mode run would silently reload the
        # tool-mode cache built at the same seq_len.
        cache_model_id=config["model_id"] + "_" + Path(dataset_cfg["data_files"]).stem,
        max_seq_length=int(dataset_cfg["max_seq_len"]),
        max_examples=int(dataset_cfg.get("max_train_examples") or 0),
    )
    train_dataset, eval_dataset = split_dataset(
        dataset,
        float(dataset_cfg.get("eval_fraction", 0.0)),
        int(config.get("seed", 42)),
    )

    log_phase(f"loading model with from_pretrained: {initial_model_id}")
    _model_kwargs: dict[str, Any] = dict(
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    # Single-process model-parallel path (device_map="auto") — pipeline-shards the model
    # across the node's GPUs, no torchrun/FSDP/NCCL. Leave unset for the DDP/FSDP
    # (torchrun) path.
    if config.get("device_map"):
        _model_kwargs["device_map"] = config["device_map"]
    model = AutoModelForCausalLM.from_pretrained(initial_model_id, **_model_kwargs)
    log_phase("model loaded; configuring train scope")
    model.config.use_cache = False
    lora_cfg = config.get("lora")
    if lora_cfg:
        # LoRA path: freeze the base, train low-rank adapters. Frozen base structurally
        # prevents catastrophic forgetting, and removes the 31B optimizer states (so a
        # larger window fits). Runs under DDP (no fsdp block in the LoRA config).
        from peft import LoraConfig, get_peft_model
        # target_modules may be a regex STRING (e.g. to hit only language_model.* linears and
        # skip the vision tower's Gemma4ClippableLinear wrappers) or a list of name suffixes.
        tm = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=tm if isinstance(tm, str) else list(tm),
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=bool(lora_cfg.get("use_dora", False)),
            use_rslora=bool(lora_cfg.get("use_rslora", False)),
        )
        model = get_peft_model(model, peft_config)
        # With a frozen base, gradient checkpointing needs grads to flow into the inputs.
        model.enable_input_require_grads()
        if int(os.environ.get("RANK", "0")) == 0:
            model.print_trainable_parameters()
    else:
        configure_train_scope(model, config.get("train_scope", "language_model"))
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if use_liger_flce:
        # Memory-efficient loss: fuse lm_head projection + chunked cross-entropy so the
        # 32k x 256k logits tensor is never allocated (fits 12B full-FT under 80 GB).
        install_liger_flce(model)
    log_phase("model train scope configured")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if wandb_cfg:
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "vuljector-secbench"))
        os.environ.setdefault("WANDB_RUN_GROUP", wandb_cfg.get("group", config["exp_name"]))
        os.environ.setdefault("WANDB_JOB_TYPE", wandb_cfg.get("job_type", "full_finetune_fsdp"))
        os.environ.setdefault("WANDB_WATCH", "false")

    max_steps = int(config.get("max_steps", -1))
    if str(config.get("save_steps", "")).lower() in {"auto_10", "ten_percent"}:
        if max_steps <= 0:
            raise ValueError("save_steps: auto_10 requires max_steps > 0")
        config["save_steps"] = max(1, max_steps // 10)
    if str(config.get("eval_steps", "")).lower() in {"auto_10", "ten_percent"}:
        config["eval_steps"] = config["save_steps"]

    log_phase("creating TrainingArguments")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config["exp_name"],
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        num_train_epochs=float(config.get("epochs", 1)),
        max_steps=max_steps,
        warmup_ratio=float(config.get("lr_scheduler", {}).get("warmup_ratio", 0.0)),
        lr_scheduler_type=config.get("lr_scheduler", {}).get("name", "cosine"),
        optim=optimizer_cfg.get("name", "adamw_torch_fused"),
        bf16=True,
        tf32=True,
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        logging_steps=int(config.get("log_every_n_steps", 1)),
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        save_only_model=bool(config.get("save_only_model", False)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(config.get("eval_steps", config.get("save_steps", 100))) if eval_dataset else None,
        report_to=["wandb"] if wandb_cfg else [],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        fsdp=fsdp_strategy,
        fsdp_config=fsdp_cfg,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=config.get("ddp_find_unused_parameters"),
        ddp_timeout=3600,
    )

    log_phase("creating Trainer")
    callbacks = []
    hub_cfg = config.get("hub", {})
    if hub_cfg.get("upload_checkpoints"):
        callbacks.append(
            HubCheckpointCallback(
                repo_id=hub_cfg["repo_id"],
                token_file=hub_cfg.get("token_file"),
                private=bool(hub_cfg.get("private", True)),
                repo_type=hub_cfg.get("repo_type", "model"),
                fail_on_error=bool(hub_cfg.get("fail_on_error", False)),
            )
        )

    trainer_cls = LigerFlceTrainer if use_liger_flce else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    log_phase("starting trainer.train()")
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint") or None)
    if config.get("save_final_model", True):
        log_phase("trainer.train() completed; saving final model")
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(output_dir / "final_model")
        print(f"Saved final model to {output_dir / 'final_model'}", flush=True)
        # SAVE-INTEGRITY GATE: the FSDP FULL_STATE_DICT consolidation in the final
        # save has been observed to silently drop the LAST decoder layer (leaving
        # fresh-init garbage -> the model emits token-salad). Verify and flag loudly;
        # the in-loop checkpoints are reliable and should be used if this fails.
        if int(os.environ.get("RANK", "0")) == 0:
            base_dir = initial_model_id
            try:
                from huggingface_hub import snapshot_download
                base_dir = snapshot_download(config["model_id"])
            except Exception:
                pass
            bad = verify_save_integrity(str(output_dir / "final_model"), str(base_dir))
            if bad:
                msg = (f"SAVE-INTEGRITY FAILURE: final_model decoder layer(s) {bad} are CORRUPTED "
                       f"(cosine-to-base ~0 -> FSDP consolidation did not gather them). DO NOT use "
                       f"final_model; use the last in-loop checkpoint under {output_dir} instead.")
                log_phase("ERROR: " + msg)
                (output_dir / "final_model" / "SAVE_CORRUPTED.txt").write_text(msg + "\n", encoding="utf-8")
            else:
                log_phase("save-integrity check PASSED: all decoder layers gathered (no collapse).")
    else:
        log_phase("trainer.train() completed; skipping final_model duplicate save")


if __name__ == "__main__":
    main()
