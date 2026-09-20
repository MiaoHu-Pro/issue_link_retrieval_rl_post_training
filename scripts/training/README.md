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

## Slurm submission

Submit the A100 job from the project root:

```bash
sbatch scripts/training/submit_train_cpt_qwen35.sh
```

The submission script activates `ilr_rl_post_training_env`, requests one A100 GPU, 80 GB host memory, eight CPUs, and 60 hours, then runs the QLoRA defaults above. It requires the offline model at `~/scratch/llms_model/ilr_llms/base/Qwen3.5-9B-Base` and checks CUDA, BF16 support, PyTorch, Transformers, Datasets, and PEFT before training. Logs are written to `logs/qwen35-cpt-<job-id>.out` when submitted from the project root.

Forward optional trainer arguments after the script:

```bash
sbatch scripts/training/submit_train_cpt_qwen35.sh \
  --repositories redhat_v1 \
  --max-steps 20 \
  --save-steps 10 \
  --eval-steps 10
```

The shell script sets `HF_DATASETS_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; all model files must therefore be downloaded before submission. The QLoRA adapter and checkpoints remain under `~/scratch/llms_model/ilr_llms/`.
