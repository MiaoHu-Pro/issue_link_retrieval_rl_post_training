# CPT Dataset Preparation Plan

Status: the original preparation design is retained below. The RedHat document-export stage has now been implemented; see [cpt_dataset_implemented.md](cpt_dataset_implemented.md) for actual artifacts, counts, deviations, and remaining tokenizer work.

Project paths are relative to `ilr_rl_post_training/`. Input is `data/raw/`; the final document datasets will be saved in `data/training/cpt/`. Model weights and checkpoints stay outside the project under `~/scratch/llms_model/ilr_llms/`, as configured in `configs/paths.yaml`.

## 1. Decision

Create **repository-specific, versioned CPT corpora first**, starting with RedHat, then Apache, Jira, and MongoDB. Audit Qt and Mojang now but keep them optional extensions. Include eligible unlinked issues as well as linked issues: CPT learns domain language and does not require a positive link.

Use the provided `train_entity2id.txt` membership and verify its creation-time cutoff against the issue table. Exclude all `test_entity2id.txt` issues from CPT, not only heads listed in `test.txt`. Reserve a deterministic 1% of cleaned training-domain document groups for CPT validation; this is separate from downstream retrieval validation/test.

The available files are usable for a **processed-snapshot CPT experiment**. They are not original JIRA text. The legacy preprocessing removes URLs, punctuation, and numbers, lowercases text, and truncates cleaned descriptions. CPT benefit must therefore be measured against no-CPT controls; do not assume these lossy descriptions are an ideal domain adaptation corpus.

## 2. Inputs and their roles

Six dataset directories were found: `Apache`, `Jira`, `RedHat`, `MongoDB`, `Qt`, and `Mojang`, totaling approximately 1.7 GiB including supporting files. Root-level cutoff documents are present. No model files were found in the inspected external `base/` directory, so tokenizer-based token counts are not available from this inspection.

| File inside each repository | Role | Include its contents in CPT text? |
|---|---|---|
| `ID_Name_Project_Type_Status_sMention_Time.txt` | Main issue table: 8 tab-separated columns, no header | Title and description only |
| `train_entity2id.txt` | Training membership: count header, then key, numeric ID, created time | No |
| `test_entity2id.txt` | Held-out membership, same shape | No |
| `train.txt`, `test.txt` | Downstream query/link partition checks | No |
| `relation2id.txt`, `*_issue_links.txt`, triple and frequency files | Supervision/analysis for later retrieval stages | No |
| `cut_off_time.txt`, `jira_max_min_time.txt` at the raw root | Documented cutoff cross-checks | No |

The issue-table columns are:

```text
source_numeric_id, issue_key, title, project, issue_type, status, description, created_at
```

Parse using explicit tab delimiters and preserve empty fields. Never split these rows on arbitrary whitespace. Repository-qualified identity is `repository:issue_key`; numeric IDs also require repository scope.

The legacy split script uses `created_at <= cutoff date 23:59:59`. These dates divide elapsed timeline, not exactly 90% of documents. The timestamps have no retained timezone offset; preserve them as source-local/unspecified rather than claiming verified UTC.

| Repository | Inclusive training cutoff |
|---|---|
| Apache | 2019-12-19 |
| Jira | 2018-01-26 |
| RedHat | 2019-12-06 |
| MongoDB | 2020-10-08 |
| Qt | 2020-02-23 |
| Mojang | 2021-02-06 |

Use the current `train_entity2id.txt` / `test_entity2id.txt` files, not the alternative `*_80_pre.txt` / `*_20_pre.txt` files. Existing `test.txt` is a held-out link pool; do not mix it into CPT while deciding its later retrieval validation/test subdivision.

## 3. Measured source audit

The accompanying [machine-readable audit](../data/manifests/cpt_source_audit.json) reports a full scan of all six issue tables, training/test membership lists, and held-out link heads. It includes hashes for each issue table and membership list. It is a source audit, not a final cleaned-corpus manifest.

The scan covered **2,349,041 issue rows**, including **1,953,198 training rows** before new filtering. All issue rows had eight fields. Across the six repositories, no train/test identity overlap, missing membership-linked issue text, ID/date mismatch, or cutoff-membership violation was observed. There are repeated identities and repeated text; source row totals are therefore not final unique-document totals.

| Repository | Issue rows | Train rows | Held-out rows | Extra repeated train texts | Train rows matching held-out text |
|---|---:|---:|---:|---:|---:|
| Apache | 1,014,926 | 878,495 | 136,431 | 14,588 | 4,761 |
| Jira | 274,545 | 220,108 | 54,437 | 8,674 | 511 |
| RedHat | 353,000 | 270,333 | 82,667 | 15,476 | 2,134 |
| MongoDB | 137,172 | 113,072 | 24,100 | 5,664 | 348 |
| Qt | 148,579 | 122,119 | 26,460 | 1,170 | 833 |
| Mojang | 420,819 | 349,071 | 71,748 | 4,141 | 403 |

| Repository | Empty train descriptions | Description equals title | Extra duplicate issue-key rows (all issues) | Train word count | Median / p95 words per train issue |
|---|---:|---:|---:|---:|---|
| Apache | 3,313 | 77,204 | 17 | 52,831,258 | 33 / 274 |
| Jira | 1,169 | 10,443 | 10 | 12,425,751 | 39 / 179 |
| RedHat | 2,323 | 36,604 | 3 | 15,748,795 | 28 / 303 |
| MongoDB | 859 | 14,562 | 22 | 5,899,588 | 29 / 209 |
| Qt | 182 | 4,427 | 1 | 7,437,368 | 38 / 228 |
| Mojang | 962 | 12,631 | 5 | 12,300,708 | 23 / 96 |

Word counts above count the source title and description, including repeated title fallbacks. They are **not model-token estimates**. The final cleaned/tokenized counts will be lower or otherwise different. Duplicate issue identities have not yet been classified as identical versus conflicting records.


Audit definitions and limitations:

- Counts refer to source rows unless explicitly labeled unique. Exact-text hashes normalize whitespace and letter case over `title + newline + description`, before any proposed new sanitation.
- Repeated title/description pairs can include boilerplate or very short records; they are not necessarily duplicate issue identities.
- Train/test text overlap is measured within each repository. Cross-repository overlap, near duplicates, and description-only duplicates remain preparation tasks.
- Timestamp format was checked during this scan, together with split cutoff consistency; strict calendar parsing and timezone provenance remain builder requirements.
- A description equal to its title is consistent with the legacy missing-description fallback. Do not describe it as independent description information or assume every such instance originally lacked a description.
- A zero count for intact issue-key or URL patterns does not prove absence of answer leakage: old cleaning can turn `ABC-123` into a bare project token and remove the evidence of the reference.

## 4. Preparation pipeline

### Step A — Freeze inputs and validate identities

Write a versioned source manifest listing source-relative paths, SHA-256, file sizes, declared counts, parsed counts, and the preparation configuration hash. Expand the existing audit coverage to every file used by the builder, including cutoffs and downstream query files.

Check exactly eight fields in issue rows and three fields after membership headers. Validate complete dates with a strict parser; quarantine impossible dates and inconsistent IDs. The old timestamp-cleaning code contains repairs, so do not add new guessed repairs silently.

For duplicate `(repository, issue_key)` rows:

1. Collapse fully identical rows, preserving all source-row provenance.
2. If rows disagree in text, numeric ID, or date, quarantine the identity for review. Do not choose an arbitrary “latest” row when update times are absent.
3. Reconcile duplicate membership rows and require consistent membership/date metadata.

Assertions: no identity in both membership lists; every retained training identity resolves to exactly one canonical issue; no held-out query identity appears in eligible CPT inputs. Record every exclusion reason.

### Step B — Freeze the permitted CPT population

Keep issues in the current training membership whose creation date is at or before the recorded cutoff. Exclude all held-out membership IDs. Do not train on linked tails simply because they appear in a training triple if they fail the independent issue eligibility rule.

The full later pool can remain reserved while downstream validation/test query IDs are finalized; excluding that entire pool is already sufficient to prevent its direct issue-text use in CPT. Future changes to the downstream split require rerunning the overlap checks and creating a new dataset version.

Using final-snapshot text from an early-created issue does not establish that the text was available at issue creation. Without historical snapshots, explicitly label the experiment retrospective. Likewise, the foundation model may have seen public issues before local adaptation; this pipeline controls local CPT exposure only.

### Step C — Sanitize text conservatively

Use **title and description only** in the default corpus. Keep project/type/date/status/IDs as non-training metadata. In particular, snapshot status and resolution can expose future outcomes.

Normalize Unicode and whitespace and strip control characters. Treat explicitly empty or configured placeholder fields as absent. Do not repeat the old removal of punctuation, numbers, or stopwords. Do not synthesize missing prose, restore removed version strings, or use an LLM to rewrite issue text.

If normalized description equals normalized title, emit the title once and omit the description section. Keep informative title-only issues. Remove documents that contain no real text after cleaning, rather than filling them with “no description.” Flag short documents for inspection; do not reject every short title by a large arbitrary length threshold.

Detect surviving issue identifiers/issue URLs with repository-aware rules. Mask recognized issue IDs and remove explicit link-reference spans where they expose target links. Do not broadly erase every number or the words “block” and “duplicate”: these can carry domain meaning. Log rule-level removal counts and manually inspect deterministic samples, including title-only, very short, heavily edited, and each repository's examples.

The legacy text has already lost syntax and identifiers, so deterministic screening cannot guarantee removal of every answer-revealing statement. State that limitation in the corpus card. No link tables, positive/negative labels, relation targets, candidate IDs, comments without historical provenance, or generated rationales enter CPT text.

### Step D — Deduplicate and screen held-out overlap

Compute canonical document hashes after sanitation. Collapse exact duplicates within permitted training data, choosing the representative by stable `(created_at, issue_uid, source_row)` order. Keep an alias/provenance map so deduplication never changes downstream issue identities.

For a strict inductive corpus, remove training documents whose sanitized full-text hash matches held-out issue text; keep held-out examples untouched. Also audit substantive description-only duplicates, especially copied reports with different titles. Do not blacklist a generic one-line title shared by unrelated issues without a documented rule.

This is held-out **text-only contamination screening**, not use of test labels to optimize CPT. Record it as a benchmark curation step. If a protocol prohibits any test-text inspection, export a separately named train-only-deduplicated variant and disclose the overlap instead; do not silently mix these variants.

For longer documents, evaluate token-shingle/MinHash near-duplicate detection (proposed starting point: at least 50 whitespace words, 5-word shingles, Jaccard >=0.9 verified after candidate matching). Review samples before enabling automatic exclusion. Very short reports should use exact matching first. These thresholds are proposed, not validated by the current audit.

Build duplicate groups before allocating CPT validation. If near-duplicate grouping is enabled, all members of a group stay in one split. Audit duplicate chains so unrelated reports do not collapse into a large component accidentally.

### Step E — Separate CPT training and CPT validation

Use deterministic group hashing with a fixed recorded seed: for example, a SHA-256-derived bucket modulo 10,000; buckets below 100 go to validation. The resulting validation fraction is approximately 1%, not an exact row count. Hash group identity rather than source-file row number, and report repository/project composition.

This split is made **inside the permitted old-issue population**, after exclusions and deduplication. It measures domain language-model loss and selects CPT checkpoints. It does not replace retrieval validation, and its examples must not appear in CPT gradient updates. Reusing some of those issues later in SFT is allowed, but do not then call CPT-validation loss an independent post-SFT score.

A temporal CPT-validation subset is an optional second protocol. Keep the default fixed so early experiments are comparable. Do not create a new CPT “test” split out of downstream held-out issues for model tuning.

### Step F — Serialize documents and provenance

Save model-independent JSONL first. Each training record has exactly one training field:

```json
{"text":"Title: Dependency resolution fails during packaging\nDescription: Packaging fails when the required dependency is unavailable."}
```

This is a synthetic schema illustration, not a source example. Omit `Description:` when its content is absent or duplicates the title. JSON escaping handles newlines inside the text field.

Save provenance in a separate row-aligned sidecar:

```json
{"row_index":0,"issue_uid":"RedHat:EXAMPLE-1","source_row":123,"created_at":"2018-01-01 12:00:00","text_hash":"example_hash","split":"train","quality_flags":[],"dedup_aliases":[]}
```

The model must consume only `text`. IDs, date, split flags, and quality labels are never concatenated into the language-model sequence. Use stable ordering before sharding; shard at a declared target, for example 50,000 documents, preserving the row mapping and counting incomplete final shards.

### Step G — Tokenize only after selecting a tokenizer revision

The controlled CPT branch in the main work plan uses `Qwen/Qwen3-8B-Base`. Read its tokenizer from the external model root once available, and record exact model/tokenizer revisions and special-token behavior. Do not claim word counts are tokenizer counts or download weights just to produce this plan.

Use plain causal-language-model documents, not chat turns or SFT assistant templates. Add an explicit document-end token according to the selected tokenizer contract, without double-appending EOS. Split/tokenize each corpus partition independently.

Suggested first block length: 2,048 tokens; compare 4,096 only after profiling. Avoid truncating long documents silently: split into nonoverlapping contiguous chunks and track boundary ownership and token coverage. If packing multiple documents together, insert document boundaries and document the attention policy; EOS alone does not block attention across documents. Block-diagonal attention is preferable when supported and tested. Otherwise record ordinary causal cross-document attention as the implementation choice.

All real text tokens contribute to CPT loss. Padding labels are masked, not document content. Count raw tokens, EOS tokens, padding, discarded residuals (ideally zero), and packed tokens separately. Tokenized caches should be versioned under `cache/cpt/<corpus_version>/<tokenizer_revision>/`; the portable JSONL remains under `data/training/cpt/`.

## 5. Output layout

The following are planned outputs, not files already generated:

```text
data/training/cpt/
  redhat_v1/
    train-00000.jsonl
    validation-00000.jsonl
    provenance/
      train-00000.jsonl
      validation-00000.jsonl
    manifest.json
    quality_report.json
    dataset_card.md
  apache_v1/
  jira_v1/
  mongodb_v1/

data/processed/cpt/<repository_version>/
  canonical_issues.parquet
  exclusions.jsonl
  duplicate_groups.jsonl

data/splits/cpt/<repository_version>/
  train_issue_ids.jsonl
  validation_issue_ids.jsonl
```

Use one manifest per corpus with stage counts: source rows → unique identities → eligible identities → usable text → held-out-overlap exclusions → deduplicated documents → CPT train/validation. Counts must reconcile. Keep files in `data/raw/` unchanged.

Do not pool the six repositories by default. A pooled corpus is a separate experiment: declare included repositories, mixing weights, global deduplication, and time scope. Different repository cutoffs mean the union is not a model trained before one common date. A strict common-time experiment across the four main repositories would need a cutoff no later than the earliest selected repository cutoff (2018-01-26 here), followed by recalculated membership. Repository-specific runs avoid that ambiguity.

## 6. Implementation sequence and acceptance checks

| Order | Proposed task | Acceptance criterion |
|---|---|---|
| 1 | Source audit and configuration | Full source counts/hashes; cutoff and schema confirmed; unresolved issues enumerated |
| 2 | Canonical parser and membership filtering | Exact ID joins, strict dates, no training/held-out identity overlap, no silent malformed-row loss |
| 3 | Sanitation and duplicate grouping | Sampled text reviewed; all transformations and exclusions counted; held-out text excluded under the chosen protocol |
| 4 | Deterministic split and JSONL export | Valid JSON; only text reaches the model; duplicate groups cannot cross CPT splits; provenance resolves every document |
| 5 | RedHat tokenizer/packing pilot | Exact token counts and boundary/loss-mask checks; no cross-split packing or unexplained token loss |
| 6 | Expand to other repositories | Same configuration semantics; per-repository corpus cards; no implicit pooling |

Proposed builder entry point: `scripts/data/prepare_cpt_dataset.py`, backed by reusable modules in `src/ilr_post_training/data/`. It is now implemented; see the linked implementation note for the finalized behavior. Its configuration should include source/output roots, selected repository, cutoff, membership files, sanitation version, deduplication policy, validation seed/fraction, sharding size, and overwrite policy. Write to a temporary build directory and publish the versioned result only after validation; do not overwrite previous corpus versions silently.

Before full export, verify a small fixture covering empty fields, title-only documents, duplicate/conflicting IDs, exact-text copies, post-cutoff records, malformed dates, sanitation collisions, and JSON newline escaping. Verify deterministic reruns have identical ordered documents and hashes. After tokenization, verify train/validation disjointness and loss masking using actual token IDs.

Implementation update: the RedHat document corpus and preparation code are complete. See [the implementation note](cpt_dataset_implemented.md). Tokenized arrays and model checkpoints have not been created.
