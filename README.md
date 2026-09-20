# Issue Link Prediction through LLM Post-Training

This directory is the project root for relation-conditioned issue retrieval using a fixed candidate retriever and an LLM selection policy.

The research design, dataset requirements, model choices, and evaluation protocol are described in the [work plan](docs/issue_link_prediction_post_training_model_work_plan.md).

The copied datasets have been inspected. See the [CPT dataset preparation plan](docs/cpt_dataset_preparation_plan.md) for measured source statistics, filtering rules, splits, and planned outputs. The audit can be rerun with `python scripts/data/audit_cpt_sources.py` from this directory.

## Directory structure

```text
ilr_rl_post_training/
├── configs/                 # Versioned experiment configuration
│   ├── data/
│   ├── retrieval/
│   ├── src/ilr_post_training/   # Implementation modules
│   ├── data/
│   ├── retrieval/
│   ├── model/               # Prompt rendering, decoding, validation
│   ├── training/
│   ├── rewards/
│   └── evaluation/
├── scripts/
│   ├── data/                # Dataset preparation entry points
│   ├── training/            # Training launchers
│   └── evaluation/          # Evaluation launchers
├── experiment_results/
│   ├── runs/                # Per-run configuration, metrics, predictions
│   ├── tables/              # Aggregated comparison tables
│   ├── figures/             # Publication/analysis figures
│   └── reports/             # Experiment summaries and error analysis
├── docs/                    # Research and implementation documentation
├── tests/                   # Dataset, reward, metric, integration checks
├── notebooks/               # Exploratory analysis
├── logs/                    # Runtime logs
└── cache/                   # Rebuildable indexes and tokenization caches
```

## Data and model conventions

Use dataset-version subdirectories, for example `data/processed/redhat_v1/` and `data/training/sft/redhat_v1/`. Preserve source files and record transformations in `data/manifests/`. The existing processed source data is in `../retrieving_relation_tail_for_new_issue_by_plm/data/`; the legacy implementation is in `../ILR/`. The scaffold does not copy those datasets or download models.

Store all model weights outside this project in `~/scratch/llms_model/ilr_llms`. The model root is defined in `configs/paths.yaml`; loaders must expand `~` with `Path.expanduser()` before resolving paths. Suggested external subdirectories are `base/Qwen3-8B/`, `retrievers/`, `checkpoints/<run_id>/`, `adapters/<run_id>/`, and `exported/<run_id>/`. Store every checkpoint and adapter externally and record its resolved path in the run configuration. `configs/models/` contains configuration only, never weights. PPO and GRPO share the `data/training/rl/` input format; distinguish their optimizer settings in training configuration.

## Experiment conventions

Use `experiment_results/runs/<run_id>/` for each run. A suggested run ID is `20260919_redhat_qwen3_8b_sft_seed42`. Each run should preserve its resolved configuration, source/model revisions, metrics, predictions, and checkpoint locations. Store reusable configurations under `configs/`; keep generated run configurations with their corresponding results.

The concrete scaffold uses `data/` for the work plan's proposed dataset artifacts, `~/scratch/llms_model/ilr_llms/` for weights, and `experiment_results/runs/` for run records. These are the implementation locations for the conceptual `artifacts/` and `runs/` paths in the plan.

Large datasets, model weights, raw run outputs, caches, and logs are ignored by Git. Source code, configuration, documentation, tests, and curated tables/figures/reports remain trackable. Empty directories contain `.gitkeep` placeholders.

## Current status

The model storage path is configured in `configs/paths.yaml`. The RedHat CPT preparation pipeline is implemented; model training and tokenizer-specific packing are separate later stages.

## Build and validate the RedHat CPT dataset

Use the existing Python 3.12 virtual environment from this project directory:

```bash
.venv/bin/python scripts/data/prepare_cpt_dataset.py
.venv/bin/python scripts/data/validate_cpt_dataset.py --version redhat_v1
.venv/bin/python -m unittest discover -s tests -v
```

The builder writes versioned JSONL to `data/training/cpt/redhat_v1/` and refuses to overwrite an existing version. Canonical records and exclusion ledgers are in `data/processed/cpt/redhat_v1/`; split identity maps are in `data/splits/cpt/redhat_v1/`. This preparation stage uses only the Python standard library and does not download models. See [the CPT implementation note](docs/cpt_dataset_implemented.md) for counts, verification, and how to load the corpus.

## Qwen3.5 CPT training

The training entry point is [train_cpt_qwen35.py](scripts/training/train_cpt_qwen35.py). It reads all six prepared JSONL corpora, packs them into causal-LM blocks, and saves only the LoRA/QLoRA adapter outside the project.

Install the training dependencies in the Python 3.12 environment:

```bash
uv pip install --python .venv/bin/python -e '.[training]'
```

Run a data/configuration smoke test first:

```bash
.venv/bin/python scripts/training/train_cpt_qwen35.py --dry-run
```

Start the one-epoch QLoRA run:

```bash
.venv/bin/python scripts/training/train_cpt_qwen35.py \
  --method qlora \
  --max-seq-length 2048 \
  --gradient-accumulation-steps 16
```

The base model is expected at `~/scratch/llms_model/ilr_llms/base/Qwen3.5-9B-Base`. Checkpoints go to `~/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-all-v1/`; the final adapter goes to `~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-cpt-all-v1/`. Use `--resume-from-checkpoint` after an interruption. The script refuses to train if the model directory is missing.
