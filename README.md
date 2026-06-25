# Gemma 4 Agent Trajectory Fine-Tuning

This workspace full fine-tunes `google/gemma-4-31B-it` on the successful
`MananSuri27/vuljector-secbench-trajectories` agent trajectories.

The raw dataset snapshot is in:

```text
$RAW_DIR
```

## Format

Each dataset row points to a per-run tarball containing `trajectory.json`,
task prompts, task metadata, and validation outputs. The training path uses
only `trajectory.json.messages`, drops the terminal `exit` pseudo-role, keeps
`system`, `user`, `assistant`, and `tool` messages, and parses assistant
`bash` tool-call arguments into Gemma 4's native tool-call format.

The training setup follows SWE-smith's recipe shape: resolved trajectories are
converted to chat JSONL, inputs are masked out (`train_on_input: false`), and
full BF16 training runs with sharded distributed training, activation
checkpointing, chunk-friendly target masking, and W&B logging. Unlike
SWE-smith's XML SWE-agent data, this workspace keeps Mini-SWE-Agent alignment
by preserving Gemma 4 native `bash` tool calls.

Gemma 4's chat template supports native tool use:

```text
<|turn>system ... <|tool>declaration:bash{...}<tool|><turn|>
<|turn>user ... <turn|>
<|turn>model
<|tool_call>call:bash{command:<|"|>...<|"|>}<tool_call|>
<|tool_response>response:bash{value:<|"|>...<|"|>}<tool_response|>
```

The trainer creates one example per assistant action. The context is the
complete Mini-SWE history up through that action, and loss is applied only to
the current assistant action. User prompts, system text, and terminal/tool
observations are context, not targets.

Loss masking uses the chat template's `{% generation %}` markers with
`apply_chat_template(return_assistant_tokens_mask=True)` (the HF/TRL
`assistant_only_loss` best practice), via a training-only copy of the Gemma 4
template built at runtime by `build_training_chat_template`. The supervised span
is exactly the action's reasoning thought block + tool call(s) + any assistant
text, and never the role headers, tool observations, or the pending
`<|tool_response>` marker the template appends after a tool call. The serving
template is unchanged.

## Prep

```bash
cd gemma-agent-finetune
uv run --with huggingface_hub --with transformers --with sentencepiece --with 'jinja2>=3.1.0' \
  python scripts/prepare_agent_data.py \
  --raw-dir "$RAW_DIR" \
  --output data/processed/gemma4_agent_trajectories.jsonl \
  --model-id google/gemma-4-31B-it
```

## Run Full Fine-Tuning On Slurm

```bash
cd gemma-agent-finetune
RAW_DIR=/path/to/vuljector-secbench-trajectories \
WANDB_KEY_FILE=/path/to/wandb_key.txt \
HF_KEY_FILE=/path/to/hf_key.txt \
sbatch sbatch/full_ft_gemma4_31b_mini_swe.sbatch
```

The job reads API tokens from local files supplied through environment
variables. Keep those files outside the repository.

Useful smoke-test overrides:

```bash
CONFIG=configs/train/full_ft_gemma4_31b_mini_swe.yml \
  sbatch sbatch/full_ft_gemma4_31b_mini_swe.sbatch
```
