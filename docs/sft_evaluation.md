# SFT evaluation guide

The evaluation suite measures the two SFT tasks against the same natural candidate pools and recorded issue links. Run commands from the project root on the server:

```bash
cd ~/scratch/its_project/issue_link_retrieval_rl_post_training
```

## Set-retrieval evaluation

This evaluator greedily generates the ordered `tails` JSON list. It validates labels against the supplied candidate pool and reports candidate recall, end-to-end and pool-conditioned precision/recall/F1/F2, exact-set accuracy, MAP, MRR, Recall@k, Hits@k, nDCG@k, empty-answer accuracy, parse errors, invalid labels, duplicates, latency, and generated-token counts.

Use a small smoke run before a full test:

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=eval-set-cpt-apache-smoke \
  scripts/evaluation/submit_evaluate_sft.sh \
  --task set_retrieval \
  --initialization cpt \
  --model-dataset apache \
  --eval-dataset apache \
  --split test \
  --data-version v1_full \
  --max-samples 10
```

Remove `--max-samples 10` and use a new job name for the full evaluation. If the trained adapter has a suffix, such as `smoke-fix1`, add `--adapter-suffix smoke-fix1`. The suffix must match the SFT adapter directory name.

## Pointwise evaluation

For each candidate, this evaluator computes the conditional log likelihood of the fixed completions `{"valid":true}` and `{"valid":false}`. Their two-way softmax is the validity probability. It reports binary accuracy, precision, recall, F1, AUROC, average precision, and the same query-level retrieval metrics after ranking candidates by that probability.

Select a threshold on validation data only:

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=eval-pw-cpt-apache-val \
  scripts/evaluation/submit_evaluate_sft.sh \
  --task pointwise \
  --initialization cpt \
  --model-dataset apache \
  --eval-dataset apache \
  --split validation \
  --select-threshold
```

Read `selected_threshold.json` in the resulting output directory, then freeze that value for test:

```bash
sbatch --partition=a100 --gres=gpu:1 \
  --job-name=eval-pw-cpt-apache-test \
  scripts/evaluation/submit_evaluate_sft.sh \
  --task pointwise \
  --initialization cpt \
  --model-dataset apache \
  --eval-dataset apache \
  --split test \
  --threshold 0.63
```

Replace `0.63` with the actual validation-selected value. Pointwise evaluation performs two teacher-forced completions per candidate and can take substantially longer than set generation. Its default `--batch-size 2` evaluates four sequences per forward pass. Increase this gradually on an A100 if memory permits; reduce it to `1` after an out-of-memory error.

## The 28-run matrix

Preview all commands without submitting them:

```bash
scripts/evaluation/submit_evaluation_matrix.sh --max-samples 10 --dry-run
```

Submit the 28 core jobs:

```bash
scripts/evaluation/submit_evaluation_matrix.sh --partition a100 --gres gpu:1
```

The matrix covers seven training datasets (`apache`, `redhat`, `jira`, `mongodb`, `qt`, `mojang`, and `all`), base and CPT initialization, and both SFT tasks. Repository-specific adapters are evaluated on their repository; the `all` adapter is evaluated across all six repositories. Cluster QOS limits can leave excess jobs pending, which is expected.

For final pointwise paper results, first run a validation matrix with `--split validation`, collect the selected threshold for each adapter, and submit test jobs individually with their frozen thresholds. The generic test matrix uses `0.5` and must not be presented as tuned-threshold results unless `0.5` was fixed in advance.

## Outputs

Outputs are stored under:

```text
experiment_results/evaluation/<adapter>/<evaluation-dataset>-<version>/<split>/<full-or-shard>/
```

Each run writes:

- `run_config.json`: resolved model, adapter, data, and runtime settings;
- `metrics.json`: overall and per-repository results;
- `predictions.jsonl.gz`: compressed raw generations or candidate scores;
- `selected_threshold.json`: validation-selected pointwise threshold, when requested.

Use `--no-save-predictions` only when disk space is more important than per-example error analysis. Use `--overwrite` to replace an existing completed evaluation. For adapters evaluated in shards, every query is assigned deterministically by query ID, and all candidates for a pointwise query remain in one shard. Shard metrics are partial; aggregate their prediction files before reporting a paper result.

Candidate recall is a first-stage retrieval property. End-to-end recall counts every recorded gold tail, including those absent from the candidate pool. Pool-conditioned metrics isolate the SFT model's selection or reranking behavior. Report both rather than presenting pool-conditioned recall as full retrieval recall.
