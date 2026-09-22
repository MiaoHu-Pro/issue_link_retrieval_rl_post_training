# SFT Task Workflow for Issue-Link Retrieval

> **Publication protocol update (2026-09-22):** the `v1_full` exports use natural lexical pools in every split and are retained for pilots and ablations. The final v2 SFT training split should explicitly include eligible recorded positives plus naturally retrieved hard negatives. Validation and test must use natural retrieval without gold insertion. See `docs/sft_training.md` for the implemented trainer and authoritative execution protocol.

This project uses two SFT tasks. Set retrieval is the primary proposed method; pointwise classification is the comparable baseline for the original ILR scorer.

## 1. Task definitions

### Pointwise baseline

Input is (query issue s, relation r, candidate issue o). Output is JSON such as {"valid":true} or {"valid":false}. Each candidate is scored independently and ranked by positive-class probability or the log probability of the valid response. This reproduces the old triple-validity formulation and does not directly optimize complete multi-tail retrieval.

### Set retrieval (primary)

Input is the query issue s, relation r, relation definition, and candidate pool C. Output is JSON such as {"tails":["C001","C007"]}. The application maps temporary labels back to issue IDs and emits (s,r,candidate) triples. Candidate labels are temporary and should be randomized during augmentation.

### Relation-and-tail retrieval (later extension)

When relation selection is required, use:

    Input:  query issue s + candidate pool C
    Output: [{"relation":"blocks","tails":["C001"]}, {"relation":"relates to","tails":["C004"]}]

Do not mix this into the first SFT experiment. Measure relation identification and tail selection separately first.

## 2. Dataset construction

The builder is in src/ilr_post_training/data/sft.py with entry point scripts/data/prepare_sft_datasets.py.

It reads, per repository:

    data/raw/<Repository>/ID_Name_Project_Type_Status_sMention_Time.txt
    data/raw/<Repository>/relation2id.txt
    data/raw/<Repository>/train.txt
    data/raw/<Repository>/test.txt

For every (head, relation) group it resolves the query and recorded tails, applies creation-time eligibility, retrieves a natural lexical candidate pool, assigns temporary labels, keeps only naturally retrieved positives in the target, and writes the same pool to set and pointwise exports.

The lexical retriever indexes title and description tokens, ignores very common terms, scores candidates with IDF-weighted overlap, and fills a shortfall with the newest eligible issues. It does not use link labels to retrieve candidates.

Candidate recall is:

    Recall_candidate = |C(s,r) intersect O*(s,r)| / |O*(s,r)|

If a positive is absent from C, the SFT policy cannot learn to select it in that example. The manifest records missing gold tails and candidate recall by split.

## 3. Splits and labels

The builder treats train.txt as the historical training link pool and test.txt as the held-out link pool. Training query groups are deterministically hashed into 90% train and 10% validation. All groups from test.txt are assigned to test. No query group is split across partitions.

The initial labels use a closed-world recorded-link regime:

    Recorded relation link       positive
    Candidate absent from graph  operational negative
    Other recorded relation      not automatically a negative
    Unreviewed candidate         unknown in the real world

Records mark this regime as closed_world_recorded_links. A later adjudicated regime should replace operational negatives with reviewed positive, negative, or uncertain judgments.

## 4. Output schemas

    data/training/sft/set_retrieval/<repository>_v1/
      train.jsonl
      validation.jsonl
      test.jsonl
      manifest.json

    data/training/sft/pointwise/<repository>_v1/
      train.jsonl
      validation.jsonl
      test.jsonl
      manifest.json

Set records contain prompt, completion, candidate records, gold candidate labels, source split, and label metadata. The SFT trainer should consume only prompt and completion; sidecar fields support evaluation and auditing.

Pointwise records contain one candidate, the same query/relation context, and a JSON boolean completion. The label field is an evaluation sidecar.

Prompts include query text, relation name, relation definition, and candidate labels/text. They exclude gold labels, link counts, target IDs, split names, judgment status, and future metadata. Retriever rank and score are stored as sidecar fields and should be removed from the rendered prompt unless a separate retriever-score ablation is intended.

## 5. Commands

Run a small RedHat pilot first:

    cd ilr_rl_post_training
    .venv/bin/python scripts/data/prepare_sft_datasets.py \
      --repositories RedHat --candidate-count 32 --max-groups 1000

Build all six repositories after reviewing the pilot:

    .venv/bin/python scripts/data/prepare_sft_datasets.py \
      --repositories Apache Jira RedHat MongoDB Qt Mojang --candidate-count 32

The builder refuses to overwrite an existing version. Use a new version when changing the retriever, candidate count, or label policy.

## 6. SFT model training

Start from the CPT adapter lineage:

    Qwen3.5-9B-Base
      + qwen3.5-9b-cpt-all-v1 adapter
      -> SFT on set-retrieval records
      -> qwen3.5-9b-cpt-sft-all-v1 adapter

The set-retrieval SFT loss is completion cross-entropy. Prompt tokens are masked:

    L_SFT = -(1 / |y|) sum_t log p_theta(y_t | x, y_<t)

The order of equally relevant tails is not semantically meaningful. Create target permutations or canonicalize target order by candidate presentation order. Do not interpret arbitrary recorded-link order as a preference ranking.

Use the pointwise dataset with a separate adapter/run. It is evaluated as a scorer, not as the primary multi-tail policy. Select its threshold on validation data and freeze it for test evaluation.

## 7. Validation and evaluation

Before training, audit that every positive is either in the pool or counted as a candidate miss, labels are unique, temporal eligibility holds, query groups do not overlap, gold fields are absent from prompts, relation definitions are stable, and candidate permutations preserve target sets.

Report candidate recall separately from policy metrics. For set retrieval, report Recall@k, Hits@k, MAP@k, MRR, nDCG@k, set precision/recall/F1/F2, exact-set match, empty-answer accuracy, invalid-output rate, and latency. Report pool-conditional metrics and end-to-end metrics that count positives absent from the pool.

The primary comparison is:

    Base -> SFT
    CPT  -> SFT

Use identical natural candidate pools, prompt templates, splits, seeds, and evaluation code.

## 8. Known limitations

The current builder uses transformed legacy link files. Their relation direction and five-category unification inherit the limitations documented in the CPT and main work plans. A verified directional benchmark requires original inward/outward link provenance.

Closed-world negative labels are metadata negatives, not human-confirmed invalid relations. They can contain false negatives. A judged candidate pool is needed before claiming semantic precision.

The natural lexical pool is only a baseline candidate retriever. Its recall bounds final retrieval recall. Dense retrieval, hybrid retrieval, and hard-negative mining should be added as later candidate-pool versions.

## 8.1 Relation-unsupplied extension

The implemented builder supplies the relation in every prompt. This isolates tail retrieval and is the primary SFT experiment. A separate relation-identification task should omit the relation line and use the same query and candidate pool. Its completion should be JSON such as:

    {"relations":[{"relation":"blocks","tails":["C001"]}]}

Create this corpus only after the relation-conditioned task has a stable candidate retriever. Group recorded links by relation, retain relations whose eligible tails occur in the natural pool, and use an empty `relations` array when no judged relation is retrieved. An unrecorded relation remains an operational negative, not a confirmed false relation. Canonicalize labels while preserving inverse relations unless the benchmark explicitly declares them equivalent.

## 9. Current implementation status

The builder and schemas have been implemented and syntax-checked. A 100-query RedHat pilot is saved as:

    data/training/sft/set_retrieval/redhat_v1/
    data/training/sft/pointwise/redhat_v1/

The pilot contains 89 train set examples, 11 validation set examples, 2,848 train pointwise examples, and 352 validation pointwise examples. It has no test examples because the development limit stops inside the historical training groups. Its candidate recall was approximately 0.40 on train and 0.31 on validation, demonstrating why candidate recall must be reported and why gold tails must not be silently injected.

The pilot is a schema and leakage-check artifact. Complete `v1_full` exports have subsequently been materialized for all six repositories. They remain lexical-pool pilot and ablation data because their candidate recall is too low for the final main protocol.

    .venv/bin/python scripts/data/prepare_sft_datasets.py \
      --repositories Apache Jira RedHat MongoDB Qt Mojang \
      --candidate-count 32 --version-suffix full

The full build can take substantially longer than the pilot because it performs lexical retrieval for every query-relation group and writes both set and pointwise records. The builder publishes each repository only after staging; it refuses to overwrite existing versions.

The full v1 manifests report candidate recall of roughly 0.26 to 0.49 depending on repository and split. Do not treat empty targets caused by first-stage misses as evidence that a known-positive query truly has no related issue. Build v2 training pools with explicit recorded positives and hard negatives, and build v2 validation/test pools by natural retrieval only. If lexical retrieval is insufficient, replace it with a persisted sparse, dense, or hybrid index while keeping the record schemas unchanged.

## 10. Next SFT execution plan

1. Run one repository with `--max-groups 1000` and inspect prompts, completions, candidate recall, temporal eligibility, and file sizes.
2. Run the complete repository with a version suffix such as `redhat_full_v2`; retain its manifest as the data card.
3. Repeat for the other repositories, then merge only through a new manifest-backed version. Do not mix candidate counts or retriever versions in one training run.
4. Train set-retrieval SFT from `Qwen3.5-9B-Base` or the completed CPT adapter. Mask prompt tokens and compute cross-entropy only over JSON completion tokens. Validate JSON parsing and candidate membership.
5. Train the pointwise adapter separately for comparison with the original scorer. Select its threshold on validation data and freeze it for test evaluation.
6. Compare Base+SFT and CPT+SFT with identical candidate pools. Report candidate recall separately from end-to-end retrieval metrics, then add DPO or GRPO after SFT produces valid candidate-only outputs.
