# CPT Dataset Implementation — RedHat

## Completed scope

Implemented and executed the RedHat corpus preparation stage using the existing **Python 3.12.14** virtual environment. The corpus contains plain title/description text for continued pretraining. No raw source files were modified, no model weights were downloaded, and no training run was started.

The final corpus is [data/training/cpt/redhat_v1](../data/training/cpt/redhat_v1/). This is the first repository-specific implementation of the [preparation plan](cpt_dataset_preparation_plan.md). Other repositories have not been exported by this run.

## Exact results

| Stage / output | Count |
|---|---:|
| Parsed source issue rows | 353,000 |
| Unique source identities | 352,997 |
| Source training rows | 270,333 |
| Unique training identities | 270,332 |
| Extra rows with repeated identities (all source issues) | 3 |
| Conflicting duplicate rows | 0 |
| Excluded training identities | 2,303 |
| Extra eligible identities collapsed by exact-text deduplication | 13,832 |
| Final unique documents | 254,197 |
| Document groups used for splitting | 252,972 |
| CPT training documents | 251,668 |
| CPT validation documents | 2,529 |
| Training whitespace words, including section labels | 15,008,226 |
| Validation whitespace words, including section labels | 152,446 |

These are **document/word counts, not tokenizer token counts**. Validation is approximately 1% of hash groups, so its document fraction need not equal exactly 1%.

Training identities reconcile as:

```text
270332 = 2303 excluded + 13832 aliases + 254197 documents
254197 = 251668 training + 2529 validation
```

Exclusion counts below are unique identities across the entire input, including the intentionally excluded held-out population:

| Reason | Count |
|---|---:|
| `empty_after_cleaning` | 24 |
| `heldout_exact_text_overlap` | 2,209 |
| `heldout_identity` | 82,665 |
| `heldout_substantive_description_overlap` | 70 |

## Saved implementation

| File | Purpose |
|---|---|
| [configs/data/cpt_redhat.toml](../configs/data/cpt_redhat.toml) | Version, source path, cutoff, hash-split seed, shard size, and cleaning settings |
| [src/ilr_post_training/data/cpt.py](../src/ilr_post_training/data/cpt.py) | Parsing, normalization, identity checks, overlap screening, grouping, export, and validation |
| [scripts/data/prepare_cpt_dataset.py](../scripts/data/prepare_cpt_dataset.py) | Project-relative builder entry point |
| [scripts/data/validate_cpt_dataset.py](../scripts/data/validate_cpt_dataset.py) | Recheck source/output hashes, membership, and all emitted rows |
| [tests/test_cpt_data.py](../tests/test_cpt_data.py) | Semantic and integration tests, including deterministic repeated builds |

This stage uses the Python standard library only. The project Python requirement and uv lock metadata were aligned to Python >=3.12. The existing environment was used directly.

## Saved datasets and provenance

```text
data/training/cpt/redhat_v1/
  train-*.jsonl
  validation-*.jsonl
  provenance/<matching-shard-name>.jsonl
  manifest.json
  quality_report.json
  dataset_card.md

data/processed/cpt/redhat_v1/
  canonical_issues.sqlite
  exclusions.jsonl
  quarantine_rows.jsonl
  duplicate_groups.jsonl

data/splits/cpt/redhat_v1/
  train_issue_ids.jsonl
  validation_issue_ids.jsonl
```

Actual model-input shards:

| File | Bytes |
|---|---:|
| `train-00000.jsonl` | 23,476,307 |
| `train-00001.jsonl` | 26,260,449 |
| `train-00002.jsonl` | 26,754,164 |
| `train-00003.jsonl` | 26,128,838 |
| `train-00004.jsonl` | 21,527,689 |
| `train-00005.jsonl` | 671,655 |
| `validation-00000.jsonl` | 1,258,951 |

Only the top-level `train-*.jsonl` files are training inputs. Each line contains exactly `{"text": "..."}`. Validation shards have the same schema. A document uses `Title:` and/or `Description:` sections; absent or duplicate descriptions are omitted. Neither labels nor issue IDs are deliberately concatenated into training text.

Provenance has shard-local row indices, source file/line numbers, canonical issue IDs, creation timestamps, text/group hashes, quality flags, and deduplication alias IDs. Split ledgers include aliases, so their row counts can exceed model-document counts. The SQLite table retains full identity provenance for aliases and excluded records. It includes held-out canonical records for contamination checks and **must not be used wholesale as model input**.

The manifest records exact configuration, builder/config hashes, Python version, source hashes, artifact hashes, counts, and validation results. Its artifact list does not hash the manifest itself. `quality_report.json` includes per-project document counts and deterministic examples by quality flag.

## Implemented rules

- Use `train_entity2id.txt` and `test_entity2id.txt`, not alternate 80/20 files. Assert membership consistency and that held-out query heads remain held out.
- Accept training creation times at or before **2019-12-06 23:59:59**. Preserve the source's unspecified timezone; do not relabel it UTC.
- Preserve only title and description in model input. Historical status, relation labels, IDs, project/type metadata, and link tables stay outside the input.
- Normalize Unicode/whitespace/control characters and remove configured placeholders. Preserve numbers/punctuation that remain in the source. Keep useful title-only/description-only records.
- Mask recognizable issue references for known repository key prefixes, and remove explicit relation-to-reference spans. Do not remove ordinary words such as “blocks” when they describe software behavior.
- Collapse identical identity rows and quarantine conflicting identities. Fail on inconsistent membership files, unresolved membership IDs, or cutoff violations rather than silently widening eligibility.
- Remove training documents matching held-out sanitized full text. Also remove matches to held-out substantive descriptions of at least 50 whitespace words.
- Deduplicate exact sanitized full text using case-folded hashes, choosing the earliest deterministic representative. Keep identity aliases in provenance.
- Group otherwise distinct documents sharing an identical substantive description into the same split. Hash the stable connected-group representative with the configured seed; no group crosses train/validation.
- Export stable 50,000-document shards. Validate staging outputs, rehash original sources, and publish the corpus last. Existing version paths are never overwritten. If publication is interrupted between directory moves, inspect the partial version and choose a new version for recovery.

## Validation performed

Five automated tests passed. They cover missing fields, title fallback removal, issue-reference handling, preservation of useful punctuation/numbers, strict date parsing, conflicting membership rejection, held-out full-text and long-description exclusion, conflicting issue records, duplicate aliases, shared-description split grouping, corruption detection, and byte-identical artifact manifests across two fixture builds.

The real RedHat build independently reread every JSONL/provenance pair and verified hashes, original membership, canonical eligibility, cutoff, split-group consistency, unique text, alias coverage, and count reconciliation. A separate validation command also checked every manifest-listed artifact and original source checksum. Deterministic quality examples were inspected after generation; this is a spot check, not a human relevance judgment or a complete semantic leakage audit.

Final verification reports zero emitted full-text overlap with held-out issues and zero emitted substantive exact-description overlap. There are no tokenization/packing tests because this run does not tokenize data.

## Commands

Run from `ilr_rl_post_training/`:

```bash
# Validate the already generated corpus:
.venv/bin/python scripts/data/validate_cpt_dataset.py --version redhat_v1

# Run the regression tests:
.venv/bin/python -m unittest discover -s tests -v

# Build only when this version does not already exist:
.venv/bin/python scripts/data/prepare_cpt_dataset.py
```

The final command now intentionally refuses to overwrite `redhat_v1`. To reproduce a new build, copy the configuration, change its version, and pass `--config path/to/new_config.toml`. Source paths inside the TOML resolve relative to the project root.

## Other repositories completed

The same implementation has now been run and independently validated for all copied repositories:

| Repository | Version | Training documents | Validation documents |
|---|---|---:|---:|
| RedHat | `redhat_v1` | 251,668 | 2,529 |
| Apache | `apache_v1` | 854,420 | 8,651 |
| Jira | `jira_v1` | 209,074 | 2,074 |
| MongoDB | `mongodb_v1` | 106,270 | 1,056 |
| Qt | `qt_v1` | 119,636 | 1,164 |
| Mojang | `mojang_v1` | 341,441 | 3,387 |

The six separate corpora contain 1,882,509 training documents and 18,861 validation documents in total. These are not a pooled corpus: each repository uses its own source-derived cutoff and its own deduplication namespace. Apache is the largest corpus and was the last build to finish. All six passed the standalone validator with zero emitted exact-text or substantive-description overlap against their respective held-out populations. See the [all-corpora summary](cpt_datasets_summary.md) and [machine-readable summary](../data/manifests/cpt_corpora_summary.json).

To rebuild or validate the full set, use:

```bash
.venv/bin/python scripts/data/prepare_all_cpt_datasets.py --skip-existing
```

The `--skip-existing` option skips rebuilding completed version directories but still validates each one. The builder refuses to overwrite a completed version.

Example dependency-free loading:

```python
import json
from pathlib import Path

corpus = Path("data/training/cpt/redhat_v1")
for shard in sorted(corpus.glob("train-*.jsonl")):
    with shard.open(encoding="utf-8") as handle:
        for line in handle:
            training_text = json.loads(line)["text"]
            # Pass training_text to the chosen causal-LM tokenizer.
```

## Deviations and limitations

**Canonical storage:** SQLite replaces the planned Parquet file so preparation can run without adding packages. Portable training JSONL and the planned directory separation are preserved.

**Near duplicates:** exact full text and substantive exact descriptions are handled. Similarity-based MinHash/Jaccard removal remains disabled pending threshold review. Paraphrased copies and partially shared boilerplate may remain. Cross-repository screening is not claimed for this RedHat-only corpus.

**Historical validity:** inputs were already cleaned and truncated before they were copied into `raw/`; this code cannot restore missing descriptions, numbers, punctuation, or original creation-time text. Excluding post-cutoff identities does not undo later edits to older issues. Pattern-based sanitation cannot guarantee removal of every answer-revealing statement.

**Tokenization:** the external base-model directory was empty at implementation time. These are complete model-independent CPT document datasets. Tokenizer-specific token counting, EOS handling, block packing, and causal-LM training are separate steps once the selected tokenizer is available. No fake token counts or placeholder tokenized data have been produced. Store weights/checkpoints only under `~/scratch/llms_model/ilr_llms/` and tokenized caches under the configured project cache location.
