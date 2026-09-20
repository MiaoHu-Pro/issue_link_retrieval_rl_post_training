# Issue Link Prediction through LLM Post-Training: Work Plan

Date: 19 September 2026  
Status: research and implementation plan, grounded in the local paper, code, and processed datasets. No training runs or performance claims are implied.

## 1. Recommendation and research hypothesis

Build a **fixed candidate retriever followed by a relation-conditioned LLM that selects an ordered set of issue identifiers**. Start with **Qwen/Qwen3-8B → SFT → GRPO**. Preserve **CPT → SFT → DPO → GRPO** as an ablation branch, with PPO as an alternative to GRPO. Do not assume that more training stages produce a better model.

The defensible hypothesis is that optimizing a complete response can improve multi-tail coverage and precision compared with independent issue-triple classification. The contribution should be the task formulation, relation semantics, set-level optimization, and treatment of incomplete annotations. The training-stage sequence itself is not novel: RL document reranking already exists, for example [Rank-R1](https://arxiv.org/abs/2503.06034).

Three questions must be answered separately:

1. Can the candidate retriever expose the relevant issues?
2. Can the model distinguish the requested relation from textual similarity and other relations?
3. Does outcome optimization improve selected-set quality over a sufficiently trained SFT model at a defensible computational cost?

**Immediate feasibility:** the available processed files support a five-category, multi-tail experiment. They do **not** by themselves establish the original direction of a link. A directional benchmark requires raw link provenance. This is the main prerequisite for the strongest version of the proposed contribution.

## 2. What the local evidence actually establishes

### 2.1 Paper versus the proposed description

Source: [local paper](../../paper/Issue%20Links%20Retrieval%20for%20New%20Issues%20in%20Issue%20Tracking%20Systems.pdf), especially Sections 3 and 4.

| Topic | Finding and consequence |
|---|---|
| Published models | Section 4.3–4.4 reports **BERT and GPT-2**, not a RoBERTa-only study. Include these paper baselines; the repository's RoBERTa implementation is an additional baseline. |
| Relation vocabulary | Section 4.1 merges raw link types into **general relation, duplication, temporal causal, composition, workflow**. These are broader than `blocks` and `is blocked by`; do not present them as equivalent tasks. |
| Metrics | Paper Eq. 10 is an at-least-one-success measure, comparable to Hits@k. Eq. 11 averages reciprocal rank of the highest-ranked relevant item, which is **MRR**, despite being called MAP. Implement genuine multi-positive MAP separately. |
| Temporal protocol | The paper describes a time-based partition and new-issue retrieval. Do not characterize these results as incremental post-training results without evidence of model updates. Specify model updates and candidate-index updates separately. |
| Apache scale | Table 1 reports 878,495 training issues and 141,850 training links. Candidate access is essential; parameter memorization is not a retrieval interface. |
| Legacy filtering | Section 3 uses temporal and issue-type filters. Reproduce them as a legacy configuration and measure the positives they remove. Do not assume they are harmless. |

### 2.2 Code findings that affect the new design

- [ILR README](../../ILR/README.md) documents external data download and a BERT launch example. The checked-out `ILR/` has code and experimental results but **no `data/` directory**.
- [Raw export script](../../ILR/data_processing/0.export_datasets.py) expects MongoDB database `JiraRepos`. Its active output omits `issuelinks`; the link export is commented out. An issue-text export alone cannot recover link supervision.
- [Issue preprocessing](../../ILR/data_processing/1.read_json_for_issue_obj.py) cleans text and truncates the cleaned description sequence to 300 elements. The processed text is already lossy; rebuilding from raw text is preferable for CPT.
- [Link preprocessing](../../ILR/data_processing/2.get_issue_links.py) uses `type.name`, chooses an inward/outward neighbor, then puts the newer issue first **without inverting the relation**. Its fourth output column is the newer endpoint's creation time, not a verified link-creation timestamp. This destroys information required for a direction-sensitive benchmark.
- [RoBERTa implementation](../../ILR/models/RoBERTa.py) provides a classification head and cross-entropy scoring; this supports an additional legacy scorer comparison.
- [Data utilities](../../ILR/data_process_utilities.py) contain entity-corruption negative sampling and an undirected relation-neighbor traversal. Neither absence of a recorded edge nor graph reachability establishes semantic negative or positive labels for arbitrary relation types. Do not inherit those assumptions automatically.
- [ERDse](../../ILR/ERDse.py) reads relation names and descriptions. Retain explicit relation definitions in new prompts.

### 2.3 Processed datasets actually found

Usable local source root: [`retrieving_relation_tail_for_new_issue_by_plm/data/`](../../retrieving_relation_tail_for_new_issue_by_plm/data/). It contains Apache, Jira, RedHat, MongoDB, Qt, and Mojang. Use the first four for the paper-aligned study; Qt/Mojang are optional extensions.

The following counts were obtained by reading local `train.txt` and `test.txt`, grouping by `(head key, unified relation)`, and deduplicating tails within each group. They are **before** the proposed temporal, integrity, and text-availability checks.

| Dataset | Train rows | Train query–relation groups | Train groups with >1 tail | Maximum train tails | Existing test-file rows | Maximum tails in test file |
|---|---:|---:|---:|---:|---:|---:|
| Apache | 141,850 | 122,365 | 12,578 | 32 | 23,822 | 185 |
| Jira | 126,811 | 112,325 | 10,830 | 50 | 9,639 | 16 |
| RedHat | 75,445 | 64,334 | 6,596 | 82 | 21,333 | 33 |
| MongoDB | 44,289 | 35,421 | 5,015 | 45 | 9,877 | 27 |

The existing `test.txt` row totals match the paper's validation-plus-test link totals. Treat them as a held-out pool until the actual validation selection is reconstructed; do not call the entire file the published test set.

Observed `entity2id.txt` header counts: Apache 1,014,926; Jira 274,545; RedHat 353,000; MongoDB 137,172. These are file-declared counts, not an assertion that all records have passed integrity checks.

| Available file | Observed format | New use |
|---|---|---|
| `ID_Name_Project_Type_Status_sMention_Time.txt` | No count header; 8 tab-separated fields: numeric source ID, issue key, cleaned title, project, type, status, cleaned description, created time | Issue table; mark text as processed snapshot text |
| `entity2id.txt` | Count header, then `issue_key<TAB>source_numeric_id` | ID resolution; numbers are not necessarily contiguous model indices |
| `relation2id.txt` | Count header, then `name<TAB>description<TAB>relation_id`; five categories in inspected datasets | Legacy relation registry |
| `train.txt`, `test.txt` | Count header, then `head_key<TAB>relation_name<TAB>tail_key<TAB>time` | Legacy supervision; reconstruct splits after auditing |
| `train2id.txt`, `test2id.txt` | Count header, then **head ID, tail ID, relation ID** | Join checks; beware that relation is last, unlike `train.txt` |
| `<Dataset>_issue_links.txt`, `train_original_links.txt` | Original type names with endpoints and timestamp | Type-mapping audit; type names alone do not restore inward/outward direction |
| `total_triples_with_created_time.txt` | Unified triples with endpoint-derived time | Cross-check coverage and deduplication |

## 3. Define two benchmark tracks

### Track A: available-data experiment

Use the five legacy categories and processed text. Predict the legacy newer-head-to-older-tail associations. Call this **relation-category-conditioned issue retrieval**. Do not interpret `temporal causal` as a validated directed `blocks` label, or infer that every `duplication` item is literally a duplicate: that category also includes cloning/replacement types.

This track can test SFT, DPO, and RL immediately after dataset conversion. It measures agreement with recorded, transformed links under a stated closed-world assumption.

### Track B: directional issue retrieval

Obtain original JIRA records containing both `type.inward`/`type.outward` and `inwardIssue`/`outwardIssue`, ideally with changelogs or historical snapshots. Preserve the relation relative to the current issue:

```python
# Proposed conversion logic; validate against hand-inspected raw examples.
if "outwardIssue" in link:
    emit(current_issue, link["type"]["outward"], link["outwardIssue"]["key"])
if "inwardIssue" in link:
    emit(current_issue, link["type"]["inward"], link["inwardIssue"]["key"])
```

Maintain a reviewed registry with `canonical_name`, `definition`, `inverse`, `symmetric`, `raw_aliases`, and repository scope. Examples: `blocks ↔ is_blocked_by`; `duplicates ↔ is_duplicated_by`; `relates_to ↔ relates_to` only where verified symmetric. Do not collapse causes, dependency, and workflow links simply because their language is similar.

For newer-query retrieval, if an edge must be reversed to put the newer endpoint first, **also invert its relation**. Deduplicate opposite encodings of the same physical edge. Never guess missing direction from issue ages, text, or generic type names. Unrecoverable records remain Track A data or are excluded from Track B.

### Shared target and inference contract

For query creation/prediction time `t_s`, define eligible issues:

\[
\mathcal E(t_s)=\{o:o\ne s,\;t_o<t_s,\;o\text{ is accessible in the chosen repository scope}\}.
\]

Use strictly earlier timestamps initially; report excluded ties. Do not automatically restrict to the same project: inspected data contains cross-project links.

\[
O^*(s,r)=\{o\in\mathcal E(t_s):(s,r,o)\text{ has the designated positive judgment}\}.
\]

The model receives issue text, relation definition, and candidate records and emits `{"tails":["C003","C017"]}` or `{"tails":[]}`. Closing the JSON and EOS implements STOP. There is no rationale requirement. The application maps temporary labels back to repository-qualified IDs and constructs triples with the supplied query and relation.

This is a response-level contextual bandit, with autoregressive token actions inside the response. No search agent or learned reward model is required for the first version.

## 4. Base-model selection and checkpoint lineage

The following are concrete choices, not claims that they are the newest or best models for this dataset. Model cards were checked on the document date.

| Role | Exact model ID | Decision |
|---|---|---|
| Main practical policy | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Default for SFT → GRPO. This is already post-trained, despite lacking an `Instruct` suffix. Disable thinking with the supported chat-template setting and train short structured responses. |
| Controlled CPT backbone | [`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base) | Use the same raw checkpoint for both Base → SFT and Base → CPT → SFT. This is the clean CPT comparison. |
| Resource pilot | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | Validate dataset plumbing, output handling, and RL stability before expensive 8B runs. Do not interpret its result as an 8B ablation. |
| Alternative family/version control | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), with [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) for a matching CPT branch | Optional replication to test whether gains depend on Qwen3. |
| Dense candidate encoder | [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | Frozen English dense retrieval baseline, combined with lexical retrieval. Follow the encoder's query/document formatting. |
| Legacy scorers | BERT and GPT-2 from the paper; RoBERTa from local code | Reproduce exact checkpoint IDs from recoverable experiment configuration; otherwise label new runs explicitly, e.g. `bert-base-cased`, `gpt2`, `roberta-base`. |

Qwen3-8B's documented native context is 32,768 tokens. Start with an **8,192-token total budget**, expanding only after profiling. The proposed prompt budgets and optimization settings below are our experimental choices, not model-card recommendations. [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B).

Checkpoint branches:

```text
Qwen3-8B (post-trained) ─ SFT ┬─ final SFT baseline
                             ├─ DPO
                             ├─ GRPO                 ← first main experiment
                             ├─ PPO
                             └─ DPO ─ GRPO

Qwen3-8B-Base ────────── SFT ─ same selected branches
              └─ CPT ─ SFT ┬─ DPO
                           ├─ GRPO
                           ├─ PPO
                           └─ DPO ─ GRPO              ← proposed full pipeline
```

**Do not estimate CPT's effect by comparing Base+CPT+SFT with post-trained Qwen3-8B+SFT.** That changes both domain adaptation and upstream instruction training. Compare within the same initialization. A CPT branch on the post-trained checkpoint is optional, but must have its own same-initialization control.

Freeze model revisions, tokenizer, chat template, precision, adapter configuration, and software versions. A base checkpoint may need an explicitly supplied, versioned prompt/completion template. Do not assume it inherits a suitable conversational template.

## 5. Raw-to-canonical dataset workflow

### Step 1 — Inventory and provenance

Create a source manifest with paths, SHA-256, size, repository, schema, count-header presence, observed row counts, timestamp range, and missing-field counts. Keep original files immutable. Use the actual sibling dataset root as a configurable input; do not silently assume `ILR/data` exists.

Support two adapters:

- **Processed adapter:** stream the TSV files identified above. Resolve string and numeric IDs; validate headers; quarantine malformed rows with reasons. Preserve empty fields and Unicode. Never use whitespace splitting for textual fields.
- **Raw adapter:** accept JSONL and nested JIRA `fields` objects or the flattened exporter shape. Extract text and link provenance separately. Export both text and links if re-running the MongoDB export. Detect a missing link source explicitly.

Do not execute the old export scripts unchanged: several contain hard-coded paths, lossy transformations, and special-case timestamp repairs.

### Step 2 — Canonical tables

Use Parquet for large tables and JSONL for training exports. Suggested schemas:

```text
issues:
  issue_uid = repository + ':' + issue_key
  repository, source_numeric_id, issue_key, project, issue_type
  created_at_utc, title, description
  text_observed_at, historical_text_available, source_file, source_row
  raw_text_hash, normalized_text_hash, processing_version

links:
  head_uid, relation_id, tail_uid, physical_edge_id
  raw_type_name, raw_inward, raw_outward, direction_preserved
  label_status = positive | negative | unknown | disputed
  link_created_at, label_observed_at, source_file, source_row

relations:
  repository_scope, relation_id, name, definition, inverse_id
  symmetric, legacy_category, mapping_version

queries:
  query_id, query_uid, relation_id, prediction_time, split
  eligible_gold_uids, label_cutoff, protocol_id
```

Use `null` for unavailable historical times. A newer-endpoint timestamp must never populate `link_created_at`. Identifiers from different repositories can collide; qualify all joins by repository.

### Step 3 — Text preparation

Preserve original title, error identifiers, version strings, code, and useful stack traces in raw-derived records. Normalize markup and control characters conservatively. Deduplicate repeated boilerplate. Keep absent descriptions explicitly missing; do not pretend a copied title is an independently informative description.

Exclude `issuelinks`, target-link tables, and explicit answer-revealing references from model input. Define a deterministic issue-key/URL masking policy for the strict text-only benchmark and apply it across stages. Inspect residual leakage manually; key masking alone does not remove statements such as “duplicate of the previous report.” Comments and status/resolution fields are excluded unless a valid prediction-time snapshot exists.

For processed snapshots, report that edits to descriptions cannot be undone. Sanitized current text is **not** a reconstruction of original issue text.

### Step 4 — Splitting and temporal validity

Use a frozen-model evaluation, not incremental parameter updates. Maintain two explicitly different evidence levels:

| Protocol | What is possible | Permitted claim |
|---|---|---|
| Snapshot chronological proxy | Available processed issue creation times plus final text/link snapshot | Generalization to later-created queries under a retrospective snapshot benchmark |
| Historical deployment simulation | Prediction-time issue text and label observation/changelog history | Retrieval using information genuinely available at each prediction time |

For the proxy, preserve the paper's broad training cutoff where possible, then divide the later query pool chronologically into validation and test, e.g. earliest 20% of distinct later query issues for validation, remaining 80% for test. This creates a **new reproducible static protocol**, not an exact replication of the paper's unspecified validation selection. Derive boundaries from timestamps and store them, rather than splitting individual triples randomly.

Assign all relations and all tails for a query issue to one partition. Ensure inverse encodings of a held-out physical link do not enter supervised training. Remove held-out query text from CPT and training prompts in the strict inductive variant. Earlier test issues can become eligible candidates for later test queries, with a frozen policy and a time-aware index; log this index update policy.

For historical simulation, training labels must be observed by the training cutoff. Choose a fixed outcome horizon (proposed pilot: 90 days) for which future links count as gold, and exclude queries without a complete observation horizon. Vary 30/180 days as sensitivity checks. If link times are missing, do not claim this experiment has been implemented.

Distinguish holdout-label access from evaluation: the evaluator can use later recorded labels, but candidate retrieval and training cannot. Do not use held-out labels to mine training negatives or tune filters. Foundation-model pretraining contamination cannot be ruled out simply by excluding text from local CPT; disclose that separate limitation.

### Step 5 — Aggregate multiple positives

Build `gold[(query_uid, relation_id)]` as a set after canonicalization and eligibility filtering. Deduplicate physical edges and duplicate rows. Do not close graphs transitively by default: generic relation links are not necessarily transitive.

The learning unit is one **query–relation–candidate pool**, not one independent positive triple. Preserve multiple positives in the same prompt wherever the context allows them. Track the positive-cardinality histogram after every filtering step.

### Step 6 — Judgment quality and empty answers

Maintain two training/evaluation regimes:

1. **Closed-world metadata regime:** recorded edges are positive; missing edges are operational negatives. Mark all corresponding metrics and rewards as metadata agreement. A zero-recorded-positive response is not a verified semantic empty answer.
2. **Adjudicated regime:** reviewers judge each `(query, relation, candidate)` as positive, negative, or uncertain. Begin with a proposed 300 training, 100 validation, and 200 test query–relation pools of 32 candidates, stratified by relation, positive count, and retrieval difficulty. This is 19,200 candidate judgments before double review, so budget reviewer time explicitly. Enlarge only after estimating throughput.

Use two reviewers on a subset and disputed cases; record disagreement and resolution. Reviewers should see relation definitions and permitted issue text, not model identity or reward. Freeze adjudicated test pools before comparing policies. Include test candidates from a predeclared union of retrieval methods so judging does not favor one model.

A complete candidate-pool judgment establishes an empty answer **within that pool**, not the absence of relevant issues in the entire repository. Report those scopes separately. A link of another type is not automatically a negative. Unknown/disputed judgments must not silently become strong negatives.

For the first fully supervised reward implementation, require all candidate labels to be resolved or use the explicit closed-world regime. Removing unjudged items creates a different candidate distribution and must be labeled a judged-subset experiment. A future positive-unlabeled reward is a separate method, not a claim supported by merely ignoring unknown predictions.

## 6. Candidate retrieval and context construction

Use BM25 over title/description as the minimum baseline. Add the frozen BGE encoder and combine ranked lists with reciprocal-rank fusion. Proposed initial fusion constant: 60, validation-selected. Retrieve using both query text alone and query text plus the relation definition; compare candidate recall before selecting one. Relation text can dilute lexical retrieval, so do not assume conditioning helps every retriever.

Retrieve 200 lexical and 200 dense candidates, merge, apply time eligibility and self-exclusion, then retain top `M`. Filtering must occur before final top-M selection, or overfetch until enough eligible items are found. Do not exclude all other projects or use graph-derived gold information to retrieve candidates.

Pilot `M=32`; evaluate `M ∈ {16,32,64,128}`. Materialize immutable candidate pools with retriever/model revisions, index snapshot, rank, and text-budget configuration. All compared policies and scorers use the same pools.

A suggested 8,192-token budget is approximately 800 tokens for query/instructions, 32 × 180 tokens for candidate records, and 1,024 reserved for completion, leaving template overhead. Count with the actual tokenizer. Store selected text spans and truncation flags; do not let a library silently truncate the prompt, remove gold candidates, or cut the supervised completion.

Candidate labels are temporary strings such as `C001`. Randomize both candidate presentation order and label assignment during training; remap targets and metadata together. Use a fixed seeded permutation at evaluation, plus a permutation-robustness diagnostic. Keep retriever scores outside the main prompt initially.

For `M=128`, either use a profiled larger context or a separately evaluated chunk/select/merge architecture. Local chunk ranks are not automatically globally comparable. Any final merge stage must be trained and evaluated with the same intermediate selection process, and its pruning losses count toward recall.

\[
\mathrm{Recall}_{final}(s,r)\leq\frac{|C_M(s,r)\cap O^*(s,r)|}{|O^*(s,r)|}.
\]

Measure the upper bound **after** any context-budget pruning. The observed Apache group with 185 tails means an M=32 system cannot retrieve every tail even with perfect decisions. Report high-cardinality results separately and an output-length/candidate-cap ceiling. Do not silently discard these queries or advertise full retrieval.

Use naturally retrieved pools for validation, test, and the main RL branch. A gold-injected training curriculum is allowed only as a separately marked SFT ablation; it must not replace natural pools in the principal comparison.

## 7. A single intermediate record produces all training datasets

Synthetic illustration only; the following is not a sampled local issue:

```json
{
  "query_id": "toy_train_001",
  "split": "train",
  "query_uid": "toy:S",
  "query_text": "Release packaging fails until the dependency fix is applied.",
  "relation_id": "is_blocked_by",
  "relation_definition": "The query cannot proceed until the candidate issue is resolved.",
  "candidate_records": [
    {"label": "C001", "issue_uid": "toy:A", "text": "Fix required dependency resolution."},
    {"label": "C002", "issue_uid": "toy:B", "text": "Change an unrelated documentation color."},
    {"label": "C003", "issue_uid": "toy:C", "text": "Fix required packaging prerequisite."}
  ],
  "gold_labels": ["C001", "C003"],
  "negative_labels": ["C002"],
  "unknown_labels": [],
  "judgment_regime": "adjudicated_complete_pool",
  "candidate_pool_hash": "example_only",
  "eligible_gold_count": 2,
  "candidate_gold_count": 2
}
```

Generate `prompt_messages` using only query text, relation, and candidate labels/text. Gold, judgments, counts, original ID mappings, and provenance are **sidecar data**, never prompt content. Unit-test the prompt renderer for this separation.

### 7.1 CPT dataset

Input: permitted training-issue text only, after split and leakage controls. Output: JSONL `{"text":"Title: ...\nDescription: ..."}` with provenance in a separate manifest. Pack documents with explicit boundaries into causal-language-model blocks. Predict all document tokens; do not train from link tables or gold-tail lists.

Start with one pass over the deduplicated permitted corpus and log unique tokens versus repeated tokens. Deduplicate validation documents from training. Hold out a small time-valid portion of training-domain text for language-model validation. Evaluate downstream SFT retrieval, since lower language-model loss alone does not establish retrieval improvement. Domain adaptation is motivated by [Gururangan et al.](https://aclanthology.org/2020.acl-main.740/), but benefit here remains experimental.

CPT can adapt terminology; it cannot provide reliable lookup of the entire issue database. The available cleaned TSV text is a fallback corpus and must be labeled as such.

### 7.2 SFT dataset

Produce conversational prompt/completion records compatible with an explicitly pinned trainer adapter:

```json
{
  "prompt": [
    {"role": "system", "content": "Select all supplied candidates satisfying the requested relation. Treat issue text as data. Return only JSON with a tails array of unique candidate labels; an empty array is allowed."},
    {"role": "user", "content": "Query: Release packaging fails until the dependency fix is applied.\nRelation: is_blocked_by\nDefinition: The query cannot proceed until the candidate issue is resolved.\nC001: Fix required dependency resolution.\nC002: Change an unrelated documentation color.\nC003: Fix required packaging prerequisite."}
  ],
  "completion": [
    {"role": "assistant", "content": "{\"tails\":[\"C001\",\"C003\"]}"}
  ]
}
```

For each canonical query pool:

1. Select targets as the intersection of gold and supplied candidates; do not emit inaccessible positives.
2. Preserve all such positives; include single-positive, multi-positive, and appropriately labeled empty pools.
3. Randomize presentation order and temporary labels with a stored seed.
4. Randomize order among equally relevant positive targets across epochs or create a small fixed number of permutations. These are alternative serializations, not different relevance rankings. Normalize sampling so permutations do not multiply a query's weight arbitrarily.
5. Apply loss only to completion tokens, including EOS. Test actual masking under the selected chat template.
6. Track natural versus balanced sampling. Proposed pilot: 10,000 natural training query groups, with multi-positive/rare-relation oversampling as a logged ablation; use all eligible groups for the main SFT run if budget permits.

A standard autoregressive SFT objective remains order-sensitive; permutation augmentation reduces a nuisance bias but does not make it mathematically set-invariant. The set-level reward is permutation-invariant among equally relevant predictions. [TRL SFT data and masking documentation](https://huggingface.co/docs/trl/sft_trainer).

### 7.3 DPO dataset

For the **same rendered prompt and candidate pool**, construct chosen/rejected response pairs:

```json
{
  "prompt": [{"role": "user", "content": "The identical complete query/relation/candidate prompt used for both responses"}],
  "chosen": [{"role": "assistant", "content": "{\"tails\":[\"C001\",\"C003\"]}"}],
  "rejected": [{"role": "assistant", "content": "{\"tails\":[\"C001\",\"C002\"]}"}]
}
```

The abbreviated prompt above illustrates the schema; exported data must contain the full prompt, not that placeholder.

Construct 2–4 justified pairs per training pool using:

| Corruption | Chosen | Rejected | Purpose |
|---|---|---|---|
| Omit positive | `[C001,C003]` | `[C001]` | Coverage |
| Add verified/operational negative | `[C001,C003]` | `[C001,C003,C002]` | Precision |
| Replace positive | `[C001,C003]` | `[C001,C002]` | Relation discrimination |
| Move negative earlier, same selected set | `[C001,C003,C002]` | `[C002,C001,C003]` | Ranking quality |
| Judged empty pool | `[]` | `[C002]` | Abstention, only when C002 is a justified negative in that pool |

Also sample 4 responses from the SFT policy on training prompts, score with the same designated reward, and retain unequal-quality pairs. Deduplicate; cap pairs per query; log corruption versus model-generated sources. Suggested minimum reward margin: 0.05, chosen on validation and treated as a proposal. Never prefer one permutation of only equally valid tails over another.

Freeze the SFT checkpoint as reference. Save `query_id`, candidate hash, reward values, judgment regime, and pair-generation seed in a sidecar. Train against actual completion likelihoods including EOS. DPO uses paired responses rather than individual positive/negative triples and requires no online sampling during optimization itself. [DPO paper](https://arxiv.org/abs/2305.18290), [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer).

### 7.4 GRPO dataset

Export prompt-only training records with reward metadata retained in non-rendered columns:

```json
{
  "query_id": "toy_train_001",
  "prompt": [{"role": "user", "content": "The full query/relation/candidate prompt"}],
  "valid_labels": ["C001", "C002", "C003"],
  "gold_labels": ["C001", "C003"],
  "judgment_regime": "adjudicated_complete_pool",
  "candidate_pool_hash": "example_only"
}
```

There is **no required chosen/rejected pair or fixed target completion**. The current policy samples `G=4` outputs per prompt initially; compare `G=8` if reward diversity is poor. All outputs in a group must share the identical rendered prompt and label mapping. Compute verifiable response rewards from sidecar judgments.

Start from the SFT checkpoint; optionally compare DPO initialization. Freeze a reference at the start of each RL branch. Normalize rewards within each prompt group, monitor zero-variance groups, and perform the chosen clipped update. [GRPO/DeepSeekMath](https://arxiv.org/abs/2402.03300).

Explicitly pin trainer settings: current TRL exposes multiple loss and reward-normalization variants, and its documented defaults need not reproduce the original GRPO objective. Record the actual `loss_type`, clipping, reward scaling, and nonzero KL coefficient when claiming reference regularization. [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer).

### 7.5 PPO dataset

Use the **same prompt-plus-sidecar dataset and reward** as GRPO. PPO does not require converting triples into DPO preference pairs. During rollout, retain tokens, old-policy log probabilities, masks, values, terminal rewards, and termination state.

Initialize actor from the same SFT checkpoint used for the matched GRPO run. Use a frozen reference and a trainable value head/critic. Compute returns and generalized advantages from terminal retrieval reward, with explicitly configured KL treatment. Update actor and critic on fresh rollouts using a clipped PPO objective; discard stale rollouts outside the configured PPO update cycle.

A learned reward model is unnecessary, but a **value model is still required for this PPO design**. Integrate the callable retrieval reward into the selected implementation. The versioned [TRL PPO documentation](https://huggingface.co/docs/trl/v0.24.0/ppo_trainer) illustrates a model-based reward interface; do not assume a GRPO callback drops into it unchanged. Budget a small reward-adapter/integration task and test reward injection before full training.

For a fair optimizer comparison, match initialization, prompts, reward, candidate pools, and precision; report both equal rollout-token budget and actual GPU-hours. Matching optimizer steps alone is insufficient.

## 8. Reward definition and failure handling

For a completely judged pool, let `G_C` be its gold labels and `P` the selected unique valid labels. Define:

\[
TP=|P\cap G_C|,\quad FP=|P\setminus G_C|,\quad FN=|G_C\setminus P|,
\]

\[
F_2=\frac{5TP}{5TP+4FN+FP}.
\]

For nonempty `G_C`, use binary-gain ranking quality:

\[
DCG@k=\sum_{i=1}^{\min(k,|y|)}\frac{\mathbf1[y_i\in G_C]}{\log_2(i+1)},\quad
IDCG@k=\sum_{i=1}^{\min(k,|G_C|)}\frac{1}{\log_2(i+1)},
\]

\[
R=\alpha F_2+(1-\alpha)DCG@k/IDCG@k.
\]

Proposed starting settings: `alpha=0.75`, `k=10`; tune alpha in `{0.5,0.75,1.0}` on validation. These are experimental choices. IDCG uses all gold in the candidate pool, not only returned positives. The set term covers positives beyond rank k.

Define edge cases before training:

| Case | Reward/handling |
|---|---|
| Completely judged empty gold and empty response | Reward 1 |
| Completely judged empty gold and any nonempty response | Reward 0 |
| Nonempty gold and empty response | Reward 0 |
| Malformed JSON, out-of-pool label, repeated label, unfinished/truncated response | Reward -1 in the initial strict implementation; record separate invalidity categories |
| Valid nonempty response on resolved pool | Formula above |
| Unknown/disputed candidate judgments | Exclude from the complete-judgment RL branch or use the explicitly labeled closed-world regime; no silent conversion |

Do not repair outputs before computing training reward. At inference, a validator may reject or deduplicate for the application, but log raw validity and evaluate the exact declared policy. Avoid hidden retries or report their additional latency and effect.

Audit reward gaming with `select none`, `select all`, `top-N retriever`, one-correct-only, malformed, duplicate, and shuffled outputs. F2 can favor broad selection when precision is uncertain. Tune against independent triple precision/recall and report false additions; a higher training reward alone is not a success criterion. Do not reward explanations or output length.

For GRPO, record both reward and advantage statistics. A group whose responses all achieve the same reward has no group-relative reward signal. Do not preferentially resample only high-reward prompts; difficulty sampling changes the data distribution and must be controlled.

Structured decoding is desirable, but implement it carefully: a JSON grammar alone does not ensure candidate membership or uniqueness. Prefixes such as `C001` can span several tokenizer tokens. Use a grammar/trie and selected-label state if enforcing all constraints. Sampling, old-policy probabilities, and update probabilities must describe the **same constrained policy**. If the trainer cannot support this consistently, use unconstrained JSON generation plus the strict validator/penalty for all main comparisons, and add constrained decoding only as a separate supported configuration.

## 9. Training settings, compute, and experiment controls

These are initial search ranges, not verified optimal hyperparameters or hardware guarantees.

| Stage | Proposed starting settings |
|---|---|
| CPT | One deduplicated corpus pass; 2k–4k token blocks; LR around 1e-5 for full updates or separately tuned adapters; validate forgetting and downstream retrieval |
| SFT | LoRA rank 32, alpha 64, dropout 0.05; LR 1e-4; 1–3 epochs; effective batch 32 prompts; completion-only loss |
| DPO | Same adapter capacity; LR 5e-6; beta 0.1; 1 epoch initially |
| GRPO | LR 1e-6; G=4; clip 0.2; explicit KL beta 0.02; sample temperature 0.8; validate group statistics |
| PPO | Actor LR 1e-6; critic LR pilot 5e-6; clip 0.2; explicit KL coefficient 0.02; gamma 1.0 and GAE lambda 0.95 as starting values; 1–2 update epochs per rollout batch |

Use gradient checkpointing and BF16 when supported. QLoRA is a pilot option, but quantization, rollout engine, and adapter compatibility must be smoke-tested. Do not change precision between matched methods without reporting it. Preserve complete adapter lineage: disabling an adapter is not necessarily a correct frozen SFT reference after several adaptation stages.

Provisional hardware planning: profile 4B SFT on available hardware; reserve more memory for 8B long-context RL, especially PPO's critic and multiple rollout/reference states. An 8B BF16 weight tensor alone is roughly 16 GB; this excludes gradients, optimizer, activations, reference, critic, and KV cache. Do not promise that a single 24 GB GPU can run the proposed RL configuration. Select actual batch size and context only after measuring peak memory on 50–100 representative prompts.

Measure input tokens, generated tokens, tokens/sec, wall time, GPU-hours, peak memory, and failed/truncated episodes. For GRPO, generated-token count scales approximately with `prompts × G × mean completion length`; prompt prefill and multiple training passes also matter. There is no reliable dollar-cost estimate without hardware/rental details.

Keep optimizer-specific environments versioned if necessary. Do not install the legacy requirements wholesale into the new training environment. Pin mutually compatible PyTorch, Transformers, TRL, PEFT, Datasets, and optional rollout-engine versions only after a successful smoke test. The old and modern trainer APIs are not interchangeable.

## 10. Evaluation and ablations

### Mandatory comparisons

| ID | Configuration | What it establishes |
|---|---|---|
| R0 | BM25; dense-only; hybrid retrieval | Candidate access and simple rankings |
| R1 | Paper BERT and GPT-2 scorers | Historical model-family baseline under the new protocol |
| R2 | Repository RoBERTa scorer | Additional strong encoder scorer |
| L0 | Qwen3-8B zero/few-shot, fixed demonstrations | Value of task-specific training |
| L1 | Qwen3-8B + SFT | Main supervised baseline |
| L2 | L1 + continued SFT, matched extra budget | Whether gains are just additional optimization |
| L3 | L1 + DPO | Preference optimization |
| L4 | L1 + GRPO | Primary outcome-optimization test |
| L5 | L1 + PPO | Alternative optimizer |
| L6 | L1 + DPO + GRPO | DPO initialization for RL |
| C0 | Qwen3-8B-Base + SFT | Raw-backbone control |
| C1 | Same Base + CPT + SFT | Isolated CPT effect |
| C2–C5 | C1 + DPO; +GRPO; +DPO+GRPO; +PPO | Full proposed family, lower priority after the main result |

For a clean architecture-versus-objective analysis, add an LLM pointwise scorer trained on the same candidate labels. An encoder-versus-generative comparison changes more than just the optimization loss.

Compare scorers on identical pools. For selected-triple metrics, choose scorer thresholds on validation and freeze them; reporting only their top-k ranking would leave them without a comparable abstention/selection rule. Zero-shot/few-shot demonstrations must come only from training data.

### Metrics with explicit denominators

- **Candidate recall@M:** intersection of candidate pool with all eligible recorded positives, divided by all eligible positives; report macro and micro versions and pool-completeness rate.
- **Final Recall@k:** `|returned first k ∩ gold| / |gold|`; no padding with extra retriever candidates when the model returns fewer than k.
- **Hits@k:** indicator that any positive is in the returned first k. Retain for paper comparison but name it accurately.
- **AP@k:** `sum_i Precision@i × relevant(i) / min(k, |gold|)` as the declared truncated convention. Report k in the name. Full AP uses `|gold|` as denominator, with unreturned positives contributing zero; average per query to obtain MAP.
- **MRR:** reciprocal rank of the first returned relevant item, or zero if none; corresponds to the paper's Eq. 11 interpretation.
- **nDCG@k:** binary gain; use full eligible gold for end-to-end IDCG, and pool gold only for explicitly conditional analysis.
- **Selected-triple precision, recall, F1, F2:** report micro totals and macro query scores. All-relevant-set denominators include candidate misses.
- **Exact-set match:** selected set equals gold. Also report complete-recall rate and number of unsupported additions.
- **Empty-pool accuracy and false-positive rate:** separate judged-empty from no-recorded-link pools. Exclude empty gold from recall/AP/nDCG averages and report their population separately; specify macro precision conventions.
- **Efficiency:** p50/p95 retrieval, prefill, generation, and validation latency; total latency; input/output tokens; training cost; invalidity and truncation rates.

Report by repository, relation, single/multiple/high-cardinality gold, title-only versus richer text, rare relation, and within/across-project links. Include natural-prevalence evaluation even if training is balanced. A benchmark containing only known-positive queries cannot establish deployment precision or empty-answer performance; build a separately labeled sample across all query–relation combinations.

Use at least three seeds for principal comparisons when resources permit, and paired bootstrap confidence intervals clustered by query issue, since its relation prompts are correlated. Define a primary metric before test access: proposed **macro end-to-end F2**, with triple precision and high-cardinality recall as mandatory companion metrics. Report tradeoffs rather than selecting whichever metric improves.

Additional ablations: set reward alone versus mixed reward; natural versus injected SFT pools; M/context size; no relation versus name versus definition; legacy categories versus verified directional labels; closed-world versus adjudicated training; candidate permutation; continued SFT matched budget. Do not change all these factors simultaneously.

## 11. Proposed implementation layout and execution order

Model storage configuration: **all model weights, training checkpoints, adapters, retriever weights, and exported models belong under `~/scratch/llms_model/ilr_llms/`, outside the project**. Use `configs/paths.yaml` and expand the home-directory marker before resolving paths. Run directories hold configuration, metrics, predictions, and references to external checkpoints; they must not hold weight files.

The following files/scripts are **to be implemented**, not existing executable commands:

```text
ilr_rl_post_training/
  docs/issue_link_prediction_post_training_model_work_plan.md
  configs/{data,retrieval,sft,cpt,dpo,grpo,ppo,evaluation}.yaml
  src/data/{inventory,read_processed,read_raw,canonicalize,split,build_pools}.py
  src/data/{export_cpt,export_sft,export_dpo,export_rl}.py
  src/retrieval/{bm25,dense,hybrid}.py
  src/model/{prompting,output_validation,decoding}.py
  src/training/{cpt,sft,dpo,grpo,ppo}.py
  src/rewards/retrieval_reward.py
  src/evaluation/{metrics,legacy_baselines,report}.py
  tests/
  artifacts/<dataset_version>/
    manifest.json
    canonical/{issues,links,relations,queries}.parquet
    splits/{train,validation,test}_query_ids.json
    candidates/{train,validation,test}.jsonl
    training/{cpt,sft,dpo,rl}.jsonl
    judgments/{train,validation,test}.jsonl
  runs/<run_id>/{config,metrics,predictions,provenance}/
```

Keep large artifacts out of source control. Configuration must separate raw source root, output root, track, snapshot/historical protocol, label regime, split cutoffs, and model revision.

| Milestone | Deliverable | Acceptance criterion |
|---|---|---|
| 1. Data audit | Manifest, schema adapters, relation/provenance report | All kept rows resolve IDs; dropped rows counted; Track B feasibility explicitly determined |
| 2. Benchmark | Canonical tables, temporal query splits, leakage report | No prohibited query/physical-edge overlap; eligible-tail and timestamp assertions pass |
| 3. Candidate access | Frozen BM25/hybrid pools and coverage curves | Report recall and full-pool coverage for every M and relation; choose M on validation |
| 4. SFT pilot | 4B pipeline smoke test, then 8B baseline | Valid complete outputs; meaningful fit on a tiny audited subset; held-out metrics computed end to end |
| 5. Primary RL study | Matched SFT, continued SFT, GRPO runs | Stable reward/probability handling; no test tuning; uncertainty and compute reported |
| 6. Preferences/PPO | DPO pairs and matched optimizer branches | Pairs share prompts; PPO reward adapter verified; rollout budgets recorded |
| 7. CPT study | Same-initialization CPT controls | Domain loss plus downstream benefit/forgetting assessed; no upstream-init confound |
| 8. Final study | Four-repository results, adjudication analysis, artifacts | Claims match track and temporal evidence; all baselines use declared comparable conditions |

A sensible first repository is **RedHat**: moderate scale and multi-tail supervision. MongoDB is smaller but the paper reports substantial missing-description limitations. Use Apache for the later scale/cardinality stress test. These are workload choices, not predictions of model quality.

Proposed initial candidate gate: aim for validation macro candidate recall ≥0.90 at an affordable M, while reporting per-relation failures. This is an engineering target, not a measured result or a reason to delete hard queries. If recall is substantially lower, improve candidate access before interpreting LLM recall failures.

## 12. Required verification before spending training budget

Test dataset semantics and training integration, not just file existence:

1. Synthetic inward/outward examples preserve relation direction; reversing endpoints inverts labels; symmetric inverses deduplicate correctly.
2. Source formats parse correctly, including count headers, empty descriptions, tabs, noncontiguous numeric IDs, and `head/tail/relation` ordering.
3. All query variants remain in one split; inverse physical links cannot cross prohibited boundaries; candidate timestamps precede prediction time.
4. Rendering excludes gold, judgment status, counts, and answer-bearing link metadata.
5. Permuting labels preserves target sets and reward; DPO pairs share identical prompts and have a justified reward ordering.
6. Hand-computed metric examples verify multiple positives, missing candidates, empty gold, short outputs, duplicates, and malformed JSON.
7. SFT masks prompt loss, includes completion/EOS, and does not silently truncate targets.
8. GRPO groups share a prompt, reward metadata is aligned after batching, and a mixed-quality group produces nonzero advantages.
9. PPO terminal rewards reach the correct response; critic gradients and actor/reference separation are verified.
10. If decoding is constrained, rollout and training probability calculations use identical constraints; otherwise use the declared unconstrained baseline.
11. Reference logits remain unchanged after policy updates, including when adapters are used.
12. A final inference run requires only query, relation, accessible candidate records, and model artifacts—not labels.

## 13. Success criteria and limitations

A convincing result is a repeatable improvement over SFT and strong legacy scorers on end-to-end multi-tail metrics, with reported precision, candidate coverage, uncertainty, and cost. If GRPO only improves its own metadata reward, or gains disappear with judged negatives, the study should report that limitation rather than claim semantic relation understanding.

The practical deliverable is a retriever, a selection policy, relation definitions, versioned dataset builders, and an evaluator. CPT, DPO, PPO, and GRPO consume different exports of the same canonical task records; they are not separate unrelated datasets.

The current files support the initial experiment, but three things remain unverified: availability of original directional JSON links, availability of historical text/link timestamps, and actual training hardware capacity. None prevents writing or beginning Track A conversion. The first two determine whether later results can support directional and prediction-time deployment claims.

**Recommended execution priority:** audit and canonicalize the available RedHat files → freeze temporal query splits and natural candidate pools → SFT with Qwen3-8B → compare continued SFT and GRPO → add DPO/PPO → run the controlled Base+CPT branches → scale to the other repositories and verified directional data.
