# Gemma-4-31B Fine-Tuning on Vuljector Trajectories

Fine-tune `google/gemma-4-31B-it` on successful vulnerability-fixing **agent
trajectories** (the Vuljector teacher corpus) and measure whether it generalizes
to the held-out **SEC-bench** benchmark (patch repair + PoC generation).

Two training paths are supported: **multi-GPU LoRA** (recommended) and
**single-GPU LoRA via Unsloth**. Full fine-tuning is included for reference.

---

## How it works

Each teacher trajectory is a full agent episode (system + bug description, then
`THOUGHT → bash command → observation` turns). Training expands it into **one
example per assistant action**: the context is the whole history up to that
action, and **loss is applied only to the action** (its thought + command +
stop token). Observations, prompts, and headers are context, never targets.

When an example exceeds the context window it is **middle-truncated**: the head
(task/bug description) and the tail (recent recon + the action) are kept, the
middle is elided. This preserves the task framing the model always sees at eval.
1083 trajectories → ~19,376 training examples.

---

## Setup

- **Cluster:** SLURM, partition `8i-147gb` (8× H200 141 GB per node), qos `gpulimit`.
- **Environment:** built automatically by the sbatch scripts with `uv`
  (torch cu128 + `transformers>=5.8.1`, required for Gemma-4). No manual install.
- **Model:** `google/gemma-4-31B-it` is gated — you need a Hugging Face token
  with access, in a local file (see below). Keep token files **outside** the repo.

---

## 1. Data

The training data ships prebuilt on the Hub; the sbatch scripts pull it
automatically. To regenerate it from raw trajectories:

```bash
# raw tarballs -> tool-calling JSONL
python scripts/prepare_agent_data.py --raw-dir "$RAW_DIR" \
  --output data/processed/gemma4_agent_trajectories.jsonl \
  --model-id google/gemma-4-31B-it

# align to the SEC-bench eval env (rg->grep, cap observations, strip missing paths)
python scripts/normalize_trajectories.py

# tool-calling -> text-mode protocol (THOUGHT + ```mswea_bash_command``` block)
python scripts/convert_to_text.py
```

The final file used by every config is
`data/processed/normalized_trajectories_v4_text.jsonl`.

---

## 2. Train

### Multi-GPU LoRA (recommended)

```bash
HF_KEY_FILE=/path/to/hf_token.txt \
CONFIG=configs/train/lora_text_16k.yml \
  sbatch --gres=gpu:8 sbatch/lora_gemma4_31b.sbatch
```

Fewer GPUs work too — pass `--gres=gpu:N`; the trainer uses whatever it is given.
Add `WANDB_KEY_FILE=/path/to/wandb.txt` to enable W&B logging (off by default).

### Single-GPU LoRA (Unsloth)

```bash
HF_KEY_FILE=/path/to/hf_token.txt \
  sbatch sbatch/unsloth_gemma4.sbatch
```

Lower memory (bf16 LoRA ~70 GB / QLoRA ~22 GB on one H200) but slower on the
long 16k trajectories.

### Full fine-tuning (reference)

```bash
HF_KEY_FILE=/path/to/hf_token.txt \
CONFIG=configs/train/full_ft_gemma4_31b_mini_swe.yml \
  sbatch --gres=gpu:8 sbatch/full_ft_gemma4_31b_mini_swe.sbatch
```

> Note: eval an **intermediate** checkpoint, not `final_model` — the FSDP
> full-state-dict save can corrupt the final gather.

---

## 3. Configurations

| Config | Path | Window | Notes |
|--------|------|--------|-------|
| **`lora_text_16k`** | LoRA r32/α64 | 16k | **Best recipe** (see below) |
| `lora_text_8k` | LoRA r32/α64 | 8k | Faster; truncates long edit turns |
| `lora_text_24k` | LoRA r32/α64 | 24k | More context, slower |
| `lora_champ16k` | LoRA r32/α64 | 16k | 16k recipe tuned for few-GPU resume |
| `full_ft_gemma4_31b_mini_swe` | full BF16 | — | Full fine-tuning reference |

All LoRA configs share: r=32, α=64, dropout 0.05, lr `1e-4` cosine (warmup 0.03),
`adamw_torch_fused`, 3 epochs, adapters on the language-model attention + MLP
projections only.

### Successful recipe

The best model came from **`lora_text_16k`** (r32/α64, lr `1e-4` cosine, 16k
window, effective batch 16). The key detail is **checkpoint selection**: the
**epoch ≈ 1.3 checkpoint** is the sweet spot — the epoch-3 final **overtrains**
and regresses. Save often (`save_steps`) and pick an early checkpoint by eval.

### Reference scores (SEC-bench, full 300 instances)

| Model | Patch | PoC |
|-------|-------|-----|
| Gemma-4-31B base | ~63% | ~24% |
| Teacher (gpt-5.4-mini) — ceiling | 70.7% | 23.3% |

> The eval serve drifts several points over ~a day — always confirm the base
> reproduces ~63% on the **current** serve before trusting any FT comparison,
> and compare base vs. FT on the **same serve, same instances**.

---

## 4. Evaluate

Serve the base model with vLLM runtime-LoRA enabled, load a trained adapter, and
run SEC-bench against it:

```bash
# on the serve host
vllm serve google/gemma-4-31B-it --enable-lora --max-lora-rank 64 \
  --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' --dtype bfloat16

# load an adapter at runtime
curl -X POST localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name":"ft","lora_path":"/abs/path/to/checkpoint-XXXX"}'
```

Then run the SEC-bench harness (patch = populate testcase ON, PoC = OFF) and
score patch (`report_medium.jsonl`) and PoC (`report_sanitizer.jsonl`).

---

## 5. Publish a checkpoint

```bash
HF_KEY_FILE=/path/to/hf_token.txt \
CHECKPOINT_DIR=/path/to/checkpoint-XXXX \
REPO_ID=your-org/your-model \
  sbatch sbatch/publish_hf_model.sbatch
```

Uploads the adapter + tokenizer to a (private) Hub repo.

---

## Layout

```
agent_data.py                     # trajectory parsing library
scripts/prepare_agent_data.py     # raw tarballs   -> tool-calling JSONL
scripts/normalize_trajectories.py # align to eval env (rg->grep, cap obs)
scripts/convert_to_text.py        # tool-calling   -> text-mode JSONL
scripts/train_gemma4_full_sft.py  # LoRA / full trainer (DDP / FSDP)
scripts/train_gemma4_unsloth.py   # single-GPU LoRA / QLoRA trainer
scripts/publish_hf_model.py       # push adapter to the Hub
configs/train/*.yml               # training configs (see table)
sbatch/*.sbatch                   # SLURM launchers
model_metadata/                   # Gemma-4 config + chat template snapshots
```
