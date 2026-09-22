# Continued Pretraining (CPT) for Issue-Link Retrieval

This note describes how to continue pretraining Qwen3.5-9B on the six prepared issue corpora before supervised fine-tuning (SFT). It covers the objective, loss calculation, LoRA/QLoRA implementation, data flow, and A100 execution.

## 1. What CPT does

Continued pretraining, also called domain-adaptive pretraining, starts from an existing language model and trains it further on text from the target domain. Here the domain is software issue tracking: issue titles, descriptions, error messages, stack traces, versions, and development terminology.

CPT teaches the model the language and patterns of issue reports. It does **not** directly teach the relation task:

```text
Does issue S have relation r to issue O?
```

That supervision is introduced later by SFT, DPO, GRPO, or PPO. CPT should therefore use issue text only and must not train on link labels, candidate IDs, relation targets, or answer-revealing metadata.

The project currently uses six separate corpora:

```text
data/training/cpt/redhat_v1/
data/training/cpt/apache_v1/
data/training/cpt/jira_v1/
data/training/cpt/mongodb_v1/
data/training/cpt/qt_v1/
data/training/cpt/mojang_v1/
```

Together they contain 1,882,509 training documents and 18,861 validation documents. They are separate repository corpora with different historical cutoffs; they are not a cross-repository deduplicated corpus.

## 2. CPT training objective

Let a cleaned issue document be tokenized as:

\[
x=(x_1,x_2,\ldots,x_T),
\]

where each \(x_t\) is a token. A causal language model predicts the next token from the preceding tokens:

\[
p_\theta(x_t\mid x_{<t}).
\]

The training target is the same document shifted by one position:

```text
Input:  Title: dependency failure <eos> Description: packaging fails ...
Target:               dependency failure <eos> Description: packaging fails ... <eos>
```

For one sequence, the autoregressive cross-entropy loss is:

\[
\mathcal{L}_{\mathrm{CPT}}(x)
=-\frac{1}{\sum_t m_t}
\sum_{t=1}^{T}m_t\log p_\theta(x_t\mid x_{<t}),
\]

where \(m_t\in\{0,1\}\) is a loss mask. Real document tokens and document-end tokens have \(m_t=1\); padding tokens have \(m_t=0\).

For a batch of packed sequences, the loss is the mean over all non-padding target tokens:

\[
\mathcal{L}_{\mathrm{batch}}
=-\frac{1}{\sum_{b,t}m_{b,t}}
\sum_{b,t}m_{b,t}\log p_\theta(x_{b,t}\mid x_{b,<t}).
\]

The model minimizes this loss with gradient descent. Perplexity is derived from the average loss:

\[
\mathrm{PPL}=e^{\mathcal{L}}.
\]

Loss and perplexity are useful for monitoring domain adaptation, but downstream retrieval metrics decide whether CPT helped. A lower language-model loss does not guarantee better issue-link retrieval.

## 3. Document preparation

The CPT builder has already produced JSONL records with exactly one model-input field:

```json
{"text":"Title: dependency failure\nDescription: packaging fails when the required dependency is unavailable."}
```

The builder:

- Uses only the historical training membership for each repository.
- Excludes held-out issue identities.
- Uses title and description, not project, status, issue IDs, or relation labels.
- Removes empty and placeholder records.
- Masks recognizable issue references where possible.
- Removes exact sanitized text overlap with held-out issues.
- Keeps exact duplicate identities in provenance while exposing one representative document.
- Keeps substantive identical descriptions in one CPT split.
- Creates deterministic train/validation groups.

The model must not receive these sidecar fields as input:

```text
issue_uid, relation, tail_uid, candidate labels, gold labels,
project metadata, split labels, quality flags, or link timestamps
```

They remain available for provenance and later retrieval experiments.

## 4. Tokenization and packing

The training script appends the tokenizer's EOS token to each document, tokenizes the text, concatenates tokenized documents, and divides the stream into fixed-length blocks. The default block length is 2,048 tokens.

Conceptually:

```text
document A + <eos> + document B + <eos> + document C + <eos>
```

then:

```text
block 1: tokens 0 ... 2047
block 2: tokens 2048 ... 4095
...
```

Each block is trained with causal next-token prediction. Padding is not required when all blocks have the same length. Any incomplete final remainder is currently discarded by the packing function; the run should record this token count if token-level accounting is added later.

Packing improves GPU utilization compared with padding each short issue independently. It also means a later document can be visible to the model after its EOS boundary through ordinary causal attention. This is acceptable for the initial domain-adaptation run because documents are independent language examples, but it should be documented as ordinary packed causal attention.

The tokenizer must come from the same Qwen3.5 checkpoint used for training:

```text
~/scratch/llms_model/ilr_llms/base/Qwen3.5-9B-Base/
```

Do not use a Qwen3 tokenizer with Qwen3.5 weights. Record the tokenizer revision, vocabulary size, special-token IDs, raw-token count, packed-token count, and discarded remainder.

## 5. LoRA and QLoRA CPT

### LoRA

LoRA freezes the base model and inserts trainable low-rank updates into selected linear layers. For a frozen weight matrix \(W_0\), the effective weight is:

\[
W=W_0+\Delta W,
\qquad
\Delta W=\frac{\alpha}{r}BA,
\]

where \(A\) and \(B\) are trainable low-rank matrices, \(r\) is the rank, and \(\alpha\) scales the update. The language-model loss is unchanged; gradients update only the LoRA parameters.

The current script uses:

```text
rank r       = 32
LoRA alpha   = 64
dropout      = 0.05
target       = all-linear
```

### QLoRA

QLoRA stores the frozen base weights in 4-bit quantized form and computes adapter updates in higher precision, normally BF16. It substantially reduces memory but adds quantization/dequantization overhead.

For an A100 80GB, LoRA is the preferred first choice because it usually provides better throughput and avoids 4-bit quantization overhead. QLoRA remains useful when memory is tight, when the sequence length increases, or when multiple policy/reference models must share one GPU.

The current recommendation is:

```text
CPT: LoRA on A100 80GB
SFT: LoRA on A100 80GB
GRPO/PPO: QLoRA if memory becomes limiting
```

The base model is never overwritten. CPT adapter output is stored externally:

```text
~/scratch/llms_model/ilr_llms/adapters/qwen3.5-9b-cpt-all-v1/
```

Intermediate checkpoints are stored at:

```text
~/scratch/llms_model/ilr_llms/checkpoints/qwen3.5-9b-cpt-all-v1/
```

The next stage loads the base model plus the CPT adapter and continues adapter training for SFT. The SFT adapter receives a separate directory name.

## 6. Current implementation

Training entry point:

```text
scripts/training/train_cpt_qwen35.py
```

The script performs these operations:

1. Locates the six prepared JSONL corpora.
2. Loads the training and validation documents.
3. Loads Qwen3.5-9B-Base locally.
4. Creates a LoRA or QLoRA model.
5. Tokenizes the documents with the Qwen3.5 tokenizer.
6. Packs tokens into fixed-length causal-LM blocks.
7. Uses `DataCollatorForLanguageModeling` with `mlm=False`.
8. Trains with the Hugging Face `Trainer`.
9. Saves the adapter, tokenizer, configuration, and checkpoints externally.

The default one-epoch configuration is:

```text
method                         = qlora
max sequence length           = 2048
per-device batch size         = 1
gradient accumulation         = 16
learning rate                 = 2e-4
LoRA rank                     = 32
LoRA alpha                    = 64
gradient checkpointing        = enabled
```

For the A100 80GB, the more practical first run is:

```bash
sbatch scripts/training/submit_train_cpt_qwen35.sh \
  --method lora \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --dataloader-num-workers 8
```

The effective batch size remains approximately comparable:

\[
1\times16=2\times8.
\]

Run a small smoke test before the full corpus:

```bash
sbatch scripts/training/submit_train_cpt_qwen35.sh \
  --method lora \
  --repositories redhat_v1 \
  --max-steps 20 \
  --save-steps 10 \
  --eval-steps 10
```

The server environment should contain:

```text
Python       3.12.14
PyTorch      2.14.0+cu126
Transformers 5.16.1
Datasets     5.0.1
PEFT         0.21.0
Accelerate   compatible version
BitsAndBytes compatible version for QLoRA
```

The Slurm script checks CUDA visibility, GPU name, BF16 support, and the training packages before launching the Python process. It uses offline Hugging Face mode because the model is downloaded locally.

## 7. Monitoring and interpreting loss

The training log reports values such as:

```text
{'loss': '3.472', 'grad_norm': '1.204', 'learning_rate': '...', 'epoch': '...'}
```

Interpret them as follows:

- `loss`: mean next-token cross-entropy over the logged interval.
- `grad_norm`: size of the gradient before or after clipping, depending on trainer version.
- `learning_rate`: current adapter learning rate. It may be near zero during warmup.
- `epoch`: fraction of the configured corpus pass completed.
- `eval_loss`: validation language-model loss, if evaluation has run.

The initial learning rate can be zero or very small during warmup. It should increase during the warmup portion and then decay according to the trainer scheduler. A stable run should show finite loss, finite gradient norms, and no repeated NaNs or divergence.

Monitor both GPU utilization and progress:

```bash
srun --jobid=<job-id> --overlap nvidia-smi
tail -f logs/qwen35-cpt-<job-id>.out
```

Memory use does not need to approach all 80GB. GPU utilization and tokens per second are the important throughput measures. The first observed QLoRA run used about 25.6GB and 38% utilization at roughly 56 seconds per update, which projected to more than 60 hours. LoRA with per-device batch size 2 is expected to use more memory and may improve throughput.

The kernel warning about Linux 4.18 is a cluster-level concern. It cannot be fixed in the training script. If the step counter stops progressing, report the job and kernel version to the cluster administrator.

## 8. What CPT does not answer

CPT loss cannot establish that the model learned issue-link relations. After CPT, evaluate:

1. Language-model validation loss and perplexity.
2. SFT loss on relation-conditioned candidate-selection prompts.
3. Candidate recall and final multi-tail retrieval metrics.
4. Comparison against the no-CPT SFT control.

The critical downstream comparison is:

```text
Qwen3.5-9B-Base → SFT
Qwen3.5-9B-Base → CPT → SFT
```

Use identical SFT examples, candidate pools, split boundaries, seeds, and evaluation procedures. Any retrieval improvement is attributed to CPT only if the CPT and no-CPT branches are otherwise matched.

## 9. Reproducibility checklist

Record the following in the run output:

- Base model path and exact revision.
- Tokenizer path and exact revision.
- Corpus versions and manifest hashes.
- Repository order and sampling weights.
- Raw and packed token counts.
- Maximum sequence length and packing policy.
- LoRA/QLoRA method and adapter configuration.
- Effective batch size, learning rate, warmup, scheduler, and epochs.
- Python, PyTorch, Transformers, Datasets, PEFT, Accelerate, and BitsAndBytes versions.
- GPU model, driver, CUDA version, and Linux kernel version.
- Checkpoint and adapter paths.
- Validation loss, perplexity, tokens per second, wall time, and peak memory.

The CPT adapter is an initialization for SFT, not the final issue-link retrieval model. The final model must be evaluated on relation-conditioned candidate pools and compared with the original pointwise triple-scoring baselines.




--

# CPT training

`../scripts/training/train_cpt_qwen35.py` trains a QLoRA or LoRA adapter for `Qwen3.5-9B-Base` on the six prepared CPT corpora. The script does not alter the base model and writes model artifacts only to the external model root.

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

