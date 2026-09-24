# Qwen3.5 SFT implementation and run guide

## Scope

The project has two independent supervised fine-tuning tasks:

| Task | Input | Completion | Purpose |
|---|---|---|---|
| Set retrieval | Query issue, supplied relation, candidate pool | JSON tail-label list | Primary multi-tail method |
| Pointwise | Query issue, supplied relation, one candidate | JSON validity boolean | Baseline corresponding to the paper's triple scorer |

The training entry points are:

    scripts/training/train_sft_set_retrieval.py
    scripts/training/train_sft_pointwise.py

Both use `scripts/training/sft_common.py`. One parameterized Slurm launcher submits either task:

    scripts/training/submit_train_sft_qwen35.sh

Set retrieval and pointwise must be separate Slurm jobs. Their context lengths, sampling, runtime, checkpoints, and failure states differ.

## Relationship to the published ILR process

The exact implemented v1 retrieval algorithm, proposed v2 staged design, equations, ablations, and paper-ready methodology text are documented in `docs/retrieval_strategy.md`.

The old ILR model creates triples `(s, r, o)`, scores each triple independently, and ranks all eligible earlier tail issues. Its issue-type filter is a valid baseline component when the permitted types are learned from training data only.

The paper's flexible time window is not used by the new retriever. Its boundary depends on the master issue of the relevant bucket. That bucket is unknown for a genuinely new issue, so this information is unavailable at deployment time. It may be retained only as a clearly identified reproduction or oracle condition.

The deployable pipeline is:

    all issues created before query s
      -> indexed relation-conditioned retrieval
      -> high-recall top M candidates
      -> optional lightweight reranker
      -> top K candidate records
      -> set-retrieval LLM

Candidate retrieval and LLM SFT are separate components. The SFT scripts consume fixed candidate pools and do not search the full issue collection.

## Publication-critical dataset policy

The current `v1_full` natural lexical pools are valid for software smoke tests and pilot experiments. They should not be the only pools used for the final paper. Their recorded candidate recall ranges from about 0.26 to 0.49 across repositories, which places a low upper bound on end-to-end recall.

Use this policy for the final v2 dataset:

| Partition | Candidate construction |
|---|---|
| Train | Include all recorded eligible gold tails, add naturally retrieved hard negatives, randomize order |
| Validation | Natural retrieval only; never insert gold |
| Test | Natural retrieval only; never insert gold |

Gold inclusion in the training pool supplies positive supervision. It must not be described as retriever performance. Validation and test candidate recall measure the real first-stage ceiling. Select retrieval method and M using validation only.

Benchmark at least BM25, dense retrieval, and a hybrid method. For every query, eligibility requires `candidate.created < query.created`. Build the index chronologically or enforce this condition as a metadata filter. Do not expose future issues, future comments, or target-link fields.

Evaluate candidate Recall@M for several values such as 32, 64, 128, 500, and 1000, together with latency and index size. Choose M on validation rather than fixing it from test results. A later LLM stage can operate on K smaller than M after lightweight reranking.

## Data discovery and streaming

`--dataset` accepts `Apache`, `Jira`, `RedHat`, `MongoDB`, `Qt`, `Mojang`, or `all`, case-insensitively. `--data-version v1_full` resolves paths such as:

    data/training/sft/set_retrieval/apache_v1_full/train.jsonl
    data/training/sft/pointwise/apache_v1_full/train.jsonl

The JSONL files are streamed. They are not loaded or tokenized into a second full Arrow dataset, because individual training files are multiple gigabytes.

An unsuffixed `--dataset all` run with no explicit `--max-steps` uses exhaustive mode. It concatenates the six shuffled repository streams without replacement and calculates one-pass optimizer steps from the manifests. The recommended production commands use batch size 2 and gradient accumulation 8:

| Task | Training records | Effective batch | One-pass optimizer steps |
|---|---:|---:|---:|
| Set retrieval | 468,487 | 16 | 29,281 |
| Pointwise, all negatives | 14,991,370 | 16 | 936,961 |

The recommended pointwise all-repository run keeps every one of the 161,214 positives and an exact deterministic 2% of the 14,830,156 negatives. This retains 457,817 records and resolves to 28,614 steps at effective batch size 16. The selector uses an exact per-repository quota, so all six repositories contribute data without relying on an approximate random count.

Supplying `--max-steps` selects fixed-step mode, including smoke tests. In fixed-step all-repository ablations, temperature sampling uses:

To avoid thousands of validation passes, exhaustive defaults scale with the resolved run length. With the recommended configurations, set retrieval logs every 30 steps, saves every 1,000 steps, and evaluates every 1,465 steps. The 2%-negative pointwise run logs every 29 steps, saves every 1,000 steps, and evaluates every 1,431 steps. Explicit `--logging-steps`, `--save-steps`, and `--eval-steps` still override these values.

    p_d = n_d^alpha / sum_j(n_j^alpha), alpha = 0.5

This limits domination by Apache and Mojang. Use `--sampling proportional` as an ablation. Exhaustive mode does not use sampling probabilities and does not repeat smaller repositories. Run-level paths, counts, candidate recall, training mode, records per pass, resolved steps, and hyperparameters are written to `run_config.json`.

## Prompt construction and loss

The tokenizer chat template renders the system, user, and assistant messages. Thinking mode is disabled when the installed tokenizer supports that option. The labels for every system and user token are `-100`; loss is computed only on assistant completion tokens:

    L_SFT = -(1 / |y|) sum_t log p_theta(y_t | x, y_<t)

Set-retrieval records can contain 32 long issue descriptions. The trainer reconstructs a compact prompt from sidecar fields and assigns a token budget to the query and every candidate. This preserves every candidate label instead of allowing ordinary right truncation to remove later candidates. Pointwise prompts preserve the query and beginning of the single candidate; completion space is always reserved.

The set completion is:

    {"tails":["C001","C004"]}

The pointwise completion is:

    {"valid":true}

Do not train explanations or chain-of-thought text. The output validator used during final evaluation must reject malformed JSON, duplicate labels, and labels outside the supplied candidate pool.

## Base and CPT initialization

Base runs create a fresh LoRA adapter over:

    ${ILR_MODEL_ROOT}/base/Qwen3.5-9B-Base

CPT runs load this adapter as trainable initialization:

    ${ILR_MODEL_ROOT}/adapters/qwen3.5-9b-cpt-all-v1

The original CPT adapter remains unchanged. The trained CPT+SFT adapter is saved to a new directory. This means the resulting adapter contains the cumulative CPT and SFT update.

The default adapter configuration matches CPT: rank 32, alpha 64, dropout 0.05, all linear modules, and causal-language-model task type. LoRA is the default for one A100 80 GB. Pass `--method qlora` if sequence length or memory use requires 4-bit base weights.

## Pointwise class balance

Each query contributes up to 32 pointwise records, while positives are sparse. Fixed-step training keeps all positive rows and deterministically retains 10% of negative rows by default:

    --negative-keep-probability 0.10

The decision is deterministic, so repeated runs use the same subset. Exhaustive pointwise training may set a smaller negative retention probability while keeping all positives and completing one exact pass over the retained rows. The recommended `0.02` setting retains 457,817 rows. A `1.0` setting visits all 14,991,370 rows but takes about 60 days per model at the observed one-H200 speed. Test evaluation must use the complete natural candidate pool. Negative downsampling changes the training prior, so classification thresholds must be selected on the complete validation set.

## Dry runs

Dry runs verify data paths and manifests without loading model weights:

    python scripts/training/train_sft_set_retrieval.py \
      --initialization base --dataset Apache --dry-run

    python scripts/training/train_sft_pointwise.py \
      --initialization cpt --dataset RedHat --dry-run

For the all-repository mixture:

    python scripts/training/train_sft_set_retrieval.py \
      --initialization cpt --dataset all \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 8 \
      --dry-run

The dry-run output must say `"training_mode": "exhaustive"`, with 468,487 records and 29,281 steps for set retrieval.

## Slurm submission

Submit each task independently:

    sbatch scripts/training/submit_train_sft_qwen35.sh \
      --task set_retrieval \
      --initialization base \
      --dataset Apache

    sbatch scripts/training/submit_train_sft_qwen35.sh \
      --task pointwise \
      --initialization cpt \
      --dataset Apache

Arguments not consumed by the shell launcher are forwarded to Python. A short smoke run is:

    sbatch scripts/training/submit_train_sft_qwen35.sh \
      --task set_retrieval \
      --initialization cpt \
      --dataset RedHat \
      --max-steps 10 \
      --eval-steps 5 \
      --save-steps 5

The launcher defaults are starting points, not final paper hyperparameters:

| Task | Maximum length | Steps | Gradient accumulation |
|---|---:|---:|---:|
| Set retrieval | 4096 | 1000 | 8 |
| Pointwise | 2048 | 2000 | 16 |

Tune learning rate, steps, and context length on validation data. Record GPU hours and perform at least three seeds for the final configurations used in significance testing.

## Outputs and resumption

Default output names follow:

    adapters/qwen3.5-9b-base-sft-set-apache-v1-full
    adapters/qwen3.5-9b-cpt-sft-set-apache-v1-full
    adapters/qwen3.5-9b-base-sft-pointwise-apache-v1-full
    adapters/qwen3.5-9b-cpt-sft-pointwise-apache-v1-full

Intermediate Trainer checkpoints use the parallel `checkpoints/` hierarchy. The trainer refuses to start over an existing non-empty final adapter directory unless `--resume-from-checkpoint` is supplied. Preserve `run_config.json`, `validation_metrics.json`, Slurm logs, package versions, seed, and data manifest with every reported run.

## Troubleshooting

### `BatchEncoding` cannot be concatenated with a list

Transformers 5.x may return a `BatchEncoding`, tensor, or one-item batched list from `apply_chat_template`, rather than a flat Python list. The shared trainer normalizes all of these forms before joining prompt and completion IDs. If a server checkout reports:

```text
TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'
```

update `scripts/training/sft_common.py` to the version containing `token_id_list`. A failed run may already have created `run_config.json` in its output directory. Rerun a smoke test with a fresh suffix such as `--run-suffix smoke-fix1`, or resume only when a valid `checkpoint-N` directory actually exists.

The warning that Linux kernel 4.18 is older than the recommended kernel is emitted by the training stack and is not the cause of this type error. Monitor the corrected run for a genuine hang, but do not attribute an immediate Python traceback to the kernel warning.

## Evaluation boundary

The training scripts report validation cross-entropy. They do not claim retrieval effectiveness. A separate deterministic evaluator must generate constrained JSON and calculate candidate Recall@M, end-to-end Recall@k, Hits/Rr@k, MAP, nDCG, set precision/recall/F1/F2, exact-set match, invalid-output rate, and latency.

Report the paper's Rr@k and MAP definitions for direct baseline comparison, plus complete-set metrics because Rr@k only requires one relevant issue to be found. Report results per repository, macro averages, and micro averages. Evaluate Base+SFT and CPT+SFT on identical frozen candidate pools.
