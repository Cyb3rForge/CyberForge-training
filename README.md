# CyberForge — Teacher Trajectory Cleansing and SFT

Code to (a) cleanse raw teacher agent trajectories into eval-aligned supervised
fine-tuning (SFT) data, and (b) run SFT of Gemma-4 (E4B / 12B / 31B) on those trajectories for
the CyberForge project.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Gemma-4-31B is gated; export a Hugging Face token with access (`HF_TOKEN`)
before training.

## 1. Generate teacher trajectories

Cleanse raw teacher trajectories into eval-aligned SFT data. `build_data.py`
linearises native tool-calling episodes to one bash command per assistant turn,
rewrites teacher-env artifacts to the eval container (e.g. `rg`→`grep`, strips
paths absent at eval), caps long observations, and drops failed
command-not-found turns.

```bash
python data_prep/build_data.py --input <raw_trajectories.jsonl> --output <sft_data.jsonl> \
    --secb-root <path/to/SEC-bench>
```

The system/user prompts are rendered from the SEC-bench harness templates, so a
SEC-bench checkout is required; point `--secb-root` (or `SECB_ROOT`) at it.

Produces a JSONL file of chat-format trajectories (system / user / assistant
turns) in the layout the trainer consumes, at the path given by `--output`.

## 2. Run SFT

```bash
python scripts/train_gemma4_full_sft.py --config configs/train/gemma4_31b_lora.yml
```

Trains a LoRA adapter on top of the frozen base model. Point the config's
`dataset.data_files` at the JSONL from step 1. Adapters and checkpoints are
written to the config's `output_dir` (override with `--output-dir`).

## Fine-tuning on the released trajectories

Cleansed teacher trajectories are provided under `../../data/teacher_trajectories/`:

- `gpt-5.4-mini_teacher_trajectories.jsonl`
- `gemma-4-31b_teacher_trajectories.jsonl`

To SFT directly on them, set `dataset.data_files` in a LoRA config
(`configs/train/gemma4_31b_lora.yml`, `gemma4_12b_lora.yml`, or
`gemma4_e4b_lora.yml`) to one of these
files and run:

```bash
python scripts/train_gemma4_full_sft.py --config configs/train/gemma4_31b_lora.yml
```

## Layout

```
data_prep/build_data.py           # cleanse raw trajectories -> eval-aligned SFT JSONL
scripts/train_gemma4_full_sft.py  # LoRA SFT trainer
agent_data.py                     # trajectory parsing helpers
configs/train/*.yml               # LoRA configs (12B / 31B)
```
