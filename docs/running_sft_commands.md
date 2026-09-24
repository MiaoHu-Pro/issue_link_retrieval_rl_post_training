# Running the production SFT experiments

This guide contains copy-and-paste commands for the 28 production SFT jobs:

```text
7 dataset configurations × 2 initializations × 2 tasks = 28 jobs
```

The datasets are Apache, Jira, RedHat, MongoDB, Qt, Mojang, and the combined corpus. The initializations are Qwen3.5-9B-Base and the completed CPT adapter. The tasks are set retrieval and pointwise classification.

## 1. Common protocol and setup

Run every command from the project root:

```bash
cd ~/scratch/its_project/issue_link_retrieval_rl_post_training
mkdir -p logs
```

The launcher expects:

```text
Base model: ~/scratch/llms_model/ilr_llms/base/Qwen3.5-9B-Base
CPT adapter: ~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-cpt-all-v1
Conda environment: ilr_rl_post_training_env
```

All commands use batch size 2 and gradient accumulation 8, giving effective batch size 16. Set retrieval consumes every training record once. Pointwise retains every positive and an exact deterministic 5% of negatives with:

```bash
--negative-keep-probability 0.05
```

You may change `0.05`, but use a matching checkpoint label such as `neg02`, `neg03`, or `neg05`. Never resume a checkpoint created with a different retention probability, batch size, or accumulation setting.

The selected `0.05` setting retains 741,507 negatives for 161,214 positives in the combined corpus, or about 4.60 negatives per positive. The ratio varies by repository because each repository has a different candidate-pool size: it ranges from 2.90:1 for RedHat to 9.04:1 for Jira. A fixed 5% probability therefore approximates a 5:1 ratio only over the combined corpus; it does not impose a fixed per-repository class ratio. Prefer retrieval-derived hard negatives if later experiments need more informative negatives.

| Dataset | Set records | Set steps | Point positives | Retained point records | Point steps |
|---|---:|---:|---:|---:|---:|
| Apache | 110,122 | 6,883 | 41,060 | 215,202 | 13,451 |
| Jira | 101,100 | 6,319 | 17,792 | 178,656 | 11,166 |
| RedHat | 57,935 | 3,621 | 31,447 | 122,571 | 7,661 |
| MongoDB | 32,012 | 2,001 | 14,978 | 65,448 | 4,091 |
| Qt | 14,707 | 920 | 5,341 | 28,605 | 1,788 |
| Mojang | 152,611 | 9,539 | 50,596 | 292,239 | 18,265 |
| All six | 468,487 | 29,281 | 161,214 | 902,721 | 56,421 |

`--overwrite-output` replaces only the adapter named by a command, after training succeeds. Existing weights remain usable while training runs. Slurm arguments such as `--partition`, `--gres`, and `--job-name` must precede the script path.

## 2. Apache: four production jobs

### Apache Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-apache-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset apache \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-apache-v1-full-exhaustive-b2-ga8"
```

### Apache CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-apache-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset apache \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-apache-v1-full-exhaustive-b2-ga8"
```

### Apache Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-apache-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset apache \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-apache-v1-full-exhaustive-neg05-b2-ga8"
```

### Apache CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-apache-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset apache \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-apache-v1-full-exhaustive-neg05-b2-ga8"
```

## 3. Jira: four production jobs

### Jira Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-jira-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset jira \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-jira-v1-full-exhaustive-b2-ga8"
```

### Jira CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-jira-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset jira \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-jira-v1-full-exhaustive-b2-ga8"
```

### Jira Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-jira-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset jira \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-jira-v1-full-exhaustive-neg05-b2-ga8"
```

### Jira CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-jira-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset jira \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-jira-v1-full-exhaustive-neg05-b2-ga8"
```

## 4. RedHat: four production jobs

### RedHat Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-redhat-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset redhat \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-redhat-v1-full-exhaustive-b2-ga8"
```

### RedHat CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-redhat-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset redhat \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-redhat-v1-full-exhaustive-b2-ga8"
```

### RedHat Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-redhat-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset redhat \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-redhat-v1-full-exhaustive-neg05-b2-ga8"
```

### RedHat CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-redhat-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset redhat \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-redhat-v1-full-exhaustive-neg05-b2-ga8"
```

## 5. MongoDB: four production jobs

### MongoDB Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-mongodb-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset mongodb \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-mongodb-v1-full-exhaustive-b2-ga8"
```

### MongoDB CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-mongodb-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset mongodb \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-mongodb-v1-full-exhaustive-b2-ga8"
```

### MongoDB Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-mongodb-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset mongodb \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-mongodb-v1-full-exhaustive-neg05-b2-ga8"
```

### MongoDB CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-mongodb-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset mongodb \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-mongodb-v1-full-exhaustive-neg05-b2-ga8"
```

## 6. Qt: four production jobs

### Qt Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-qt-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset qt \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-qt-v1-full-exhaustive-b2-ga8"
```

### Qt CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-qt-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset qt \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-qt-v1-full-exhaustive-b2-ga8"
```

### Qt Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-qt-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset qt \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-qt-v1-full-exhaustive-neg05-b2-ga8"
```

### Qt CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-qt-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset qt \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-qt-v1-full-exhaustive-neg05-b2-ga8"
```

## 7. Mojang: four production jobs

### Mojang Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-mojang-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset mojang \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-mojang-v1-full-exhaustive-b2-ga8"
```

### Mojang CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-mojang-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset mojang \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-mojang-v1-full-exhaustive-b2-ga8"
```

### Mojang Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-mojang-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset mojang \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-mojang-v1-full-exhaustive-neg05-b2-ga8"
```

### Mojang CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-mojang-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset mojang \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-mojang-v1-full-exhaustive-neg05-b2-ga8"
```

## 8. All six repositories: four production jobs

### All repositories Base, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-base-all-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization base \
  --dataset all \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-set-all-v1-full-exhaustive-b2-ga8"
```

### All repositories CPT, set retrieval

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-set-cpt-all-full-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task set_retrieval \
  --initialization cpt \
  --dataset all \
  --data-version v1_full \
  --training-mode exhaustive \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-set-all-v1-full-exhaustive-b2-ga8"
```

### All repositories Base, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-base-all-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization base \
  --dataset all \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-base-sft-pointwise-all-v1-full-exhaustive-neg05-b2-ga8"
```

### All repositories CPT, pointwise

```bash
sbatch --partition=i7_h200 --gres=gpu:1 \
  --job-name=sft-pw-cpt-all-neg05-b2-ga8 \
  scripts/training/submit_train_sft_qwen35.sh \
  --task pointwise \
  --initialization cpt \
  --dataset all \
  --data-version v1_full \
  --training-mode exhaustive \
  --negative-keep-probability 0.05 \
  --overwrite-output \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --checkpoint-dir "${HOME}/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-sft-pointwise-all-v1-full-exhaustive-neg05-b2-ga8"
```

## 9. Resuming and monitoring

Check jobs:

```bash
squeue -u "${USER}"
```

Inspect checkpoints:

```bash
find ~/scratch/llms_model/ilr_llms/checkpoints \
  -maxdepth 2 -type d -name 'checkpoint-*' -print | sort -V
```

If a job reaches the wall-time, submit the identical command again. The new log must contain:

```text
Auto-resuming from .../checkpoint-N
```

Do not change retention probability, batch size, accumulation, seed, or checkpoint path when resuming. Check completion with:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode
```

## 10. Test evaluation

Submit the 28 default-threshold test evaluations with:

```bash
scripts/evaluation/submit_evaluation_matrix.sh \
  --partition i7_h200 \
  --gres gpu:1 \
  --split test \
  --data-version v1_full
```

Preview them with:

```bash
scripts/evaluation/submit_evaluation_matrix.sh \
  --partition i7_h200 \
  --gres gpu:1 \
  --split test \
  --data-version v1_full \
  --dry-run
```

For pointwise classification, select the threshold on validation and freeze it for test. See `docs/sft_evaluation.md`. Results are written under:

```text
experiment_results/evaluation/<adapter>/<evaluation-dataset>-v1_full/test/full/
```
