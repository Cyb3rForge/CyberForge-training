#!/usr/bin/env python3
"""Efficient single-GPU SFT of google/gemma-4-31B-it via Unsloth (LoRA / QLoRA).

Grounded in: NIST AVIATOR LoRA_FT recipe (Unsloth + QLoRA, r32/a32, paged_adamw_8bit,
lr2e-4, 18k seq, unsloth grad-ckpt, train_on_responses_only) and Unsloth's Gemma-4 guide
(Gemma-4-31B supported; bf16-LoRA ~65-75GB or QLoRA ~22GB on one H200; >3 epochs overfits;
Gemma turn markers <start_of_turn>user/model for completion masking).

Every axis is a CLI flag so a launcher can sweep. Checkpoints saved frequently so the
best (early-stopped) checkpoint can be picked by eval — the session's key finding was that
the epoch-1.3 checkpoint (72.7% patch) beat the epoch-3 final (64%, overtrained).
"""
import unsloth  # noqa: F401  (must import before torch/trl; monkey-patches)
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
import argparse, json, os
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def build_dataset(path, tokenizer, task_filter, success_only, max_examples):
    """v4 rows are {'id','messages':[{role,content}...],'metadata':{task_mode,validation_status,...}}.
    Gemma has no system role -> fold system into the first user turn. Apply the gemma chat
    template to produce a 'text' field; train_on_responses_only later masks non-model tokens."""
    rows = []
    for line in open(path):
        d = json.loads(line)
        meta = d.get("metadata", {})
        if task_filter and task_filter != "both" and meta.get("task_mode") != task_filter:
            continue
        if success_only and meta.get("validation_status") != "success":
            continue
        msgs = d["messages"]
        # fold system -> first user
        if msgs and msgs[0]["role"] == "system":
            sys = msgs[0]["content"]
            rest = msgs[1:]
            for i, m in enumerate(rest):
                if m["role"] == "user":
                    rest[i] = {"role": "user", "content": sys + "\n\n" + m["content"]}
                    break
            msgs = rest
        # normalize assistant/model role name; gemma template expects user/assistant
        conv = [{"role": ("assistant" if m["role"] in ("assistant", "model") else m["role"]),
                 "content": m["content"]} for m in msgs]
        rows.append({"conversations": conv})
    if max_examples and max_examples > 0:
        rows = rows[:max_examples]

    def fmt(batch):
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in batch["conversations"]]
        return {"text": texts}

    ds = Dataset.from_list(rows)
    ds = ds.map(fmt, batched=True, remove_columns=["conversations"])
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_len", type=int, default=16384)
    p.add_argument("--load_in_4bit", type=int, default=0)   # 0=bf16 LoRA, 1=QLoRA
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--optim", default="adamw_8bit")
    p.add_argument("--save_steps", type=int, default=60)     # frequent -> early-stop candidates
    p.add_argument("--save_total_limit", type=int, default=20)
    p.add_argument("--rslora", type=int, default=0)
    p.add_argument("--task", default="both", choices=["both", "patch", "poc"])
    p.add_argument("--success_only", type=int, default=1)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--seed", type=int, default=3407)
    a = p.parse_args()
    print("CONFIG:", json.dumps(vars(a), indent=2), flush=True)

    model, tokenizer = FastModel.from_pretrained(
        model_name=a.model, max_seq_length=a.max_seq_len, dtype=None,
        load_in_4bit=bool(a.load_in_4bit), full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
        use_gradient_checkpointing="unsloth", use_rslora=bool(a.rslora),
        random_state=a.seed,
    )
    ds = build_dataset(a.data, tokenizer, a.task, bool(a.success_only), a.max_examples)
    print(f"DATASET: {len(ds)} examples (task={a.task} success_only={a.success_only})", flush=True)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=a.max_seq_len,
            per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
            warmup_ratio=a.warmup_ratio, num_train_epochs=a.epochs, learning_rate=a.lr,
            optim=a.optim, weight_decay=a.weight_decay, lr_scheduler_type=a.scheduler,
            bf16=True, logging_steps=1, seed=a.seed, output_dir=a.output_dir,
            save_steps=a.save_steps, save_total_limit=a.save_total_limit, report_to="none",
            dataset_num_proc=4,
        ),
    )
    # Gemma completion-only masking (VERIFY markers -> non-masked assistant tokens)
    # Gemma-4 uses <|turn>role\n ... <turn|> markers (NOT the old <start_of_turn>...);
    # verified by decoding the actual chat template. Wrong markers => all labels -100.
    trainer = train_on_responses_only(
        trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n",
    )
    # sanity: how many label tokens are unmasked in example 0
    try:
        ex = trainer.train_dataset[0]
        n_lab = sum(1 for t in ex["labels"] if t != -100)
        print(f"MASK CHECK ex0: {n_lab} supervised tokens of {len(ex['labels'])}", flush=True)
        assert n_lab > 0, "ZERO supervised tokens -> chat markers mismatch; fix train_on_responses_only"
    except Exception as e:
        print("mask check note:", e, flush=True)

    trainer.train()
    model.save_pretrained(os.path.join(a.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(a.output_dir, "final_adapter"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
