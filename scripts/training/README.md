# CPT training

`train_cpt_qwen35.py` trains a QLoRA or LoRA adapter for `Qwen3.5-9B-Base` on the six prepared CPT corpora. The script does not alter the base model and writes model artifacts only to the external model root.

Set `ILR_MODEL_ROOT` if `~` resolves differently on the training node:

```bash
export ILR_MODEL_ROOT=/home/mh1f25/scratch/llms_model/ilr_llms
```

Install the optional training stack:

```bash
uv pip install --python .venv/bin/python -e '.[training]'
```

Run a no-training check:

```bash
.venv/bin/python scripts/training/train_cpt_qwen35.py --dry-run
```

Start QLoRA CPT:

```bash
.venv/bin/python scripts/training/train_cpt_qwen35.py \
  --method qlora --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16
```

The default run uses all six corpora for one epoch. The run configuration is written beside the adapter. Intermediate checkpoints are written to `checkpoints/qwen3.5-9b-cpt-all-v1/`, and the final adapter/tokenizer are written to `adapters/qwen3.5-9b-cpt-all-v1/`. Resume an interrupted run with `--resume-from-checkpoint <checkpoint-directory>`.

For a first GPU smoke test, select one repository and cap the steps:

```bash
.venv/bin/python scripts/training/train_cpt_qwen35.py \
  --repositories redhat_v1 --max-steps 20 --save-steps 10 \
  --eval-steps 10 --method qlora
```

The script uses 2,048-token packed causal-LM blocks by default. Tokenization and packing can consume substantial local cache space; ensure the project cache and external checkpoint filesystem have enough capacity before starting the full run.
