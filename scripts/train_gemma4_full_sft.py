#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import Dataset, Subset
from transformers import TrainerCallback


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
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = (
            cache_dir
            / f"mini_swe_turns_full_{safe_model_id(cache_model_id)}_{max_seq_length}_{max_examples}.pt"
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

    def _encode_messages(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> list[int]:
        return to_list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def _build_examples(self) -> list[dict[str, torch.Tensor]]:
        examples: list[dict[str, torch.Tensor]] = []
        for row in self.rows:
            messages = row.get("messages") or []
            for idx, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                if not message.get("tool_calls"):
                    continue
                if is_terminal_submit_action(message):
                    continue

                prefix_messages = messages[:idx]
                full_messages = messages[: idx + 1]
                try:
                    prefix_ids = self._encode_messages(prefix_messages, add_generation_prompt=True)
                    full_ids = self._encode_messages(full_messages, add_generation_prompt=False)
                except Exception as exc:
                    log_phase(f"skipping example {row.get('id')} turn {idx}: chat template failed: {exc}")
                    continue

                if len(full_ids) <= len(prefix_ids):
                    continue

                label_start = len(prefix_ids)
                if len(full_ids) > self.max_seq_length:
                    overflow = len(full_ids) - self.max_seq_length
                    full_ids = full_ids[overflow:]
                    label_start = max(0, label_start - overflow)

                labels = [-100] * label_start + full_ids[label_start:]
                if all(label == -100 for label in labels):
                    continue

                examples.append(
                    {
                        "input_ids": torch.tensor(full_ids, dtype=torch.long),
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
        else:
            raise ValueError(f"Unknown train_scope: {scope}")
        param.requires_grad_(should_train)
        if should_train:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(f"Train scope {scope}: trainable={trainable:,} frozen={frozen:,}", flush=True)


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
        cache_model_id=config["model_id"],
        max_seq_length=int(dataset_cfg["max_seq_len"]),
        max_examples=int(dataset_cfg.get("max_train_examples") or 0),
    )
    train_dataset, eval_dataset = split_dataset(
        dataset,
        float(dataset_cfg.get("eval_fraction", 0.0)),
        int(config.get("seed", 42)),
    )

    log_phase(f"loading model with from_pretrained: {initial_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        initial_model_id,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    log_phase("model loaded; configuring train scope")
    model.config.use_cache = False
    configure_train_scope(model, config.get("train_scope", "language_model"))
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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

    trainer = Trainer(
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
    else:
        log_phase("trainer.train() completed; skipping final_model duplicate save")


if __name__ == "__main__":
    main()
