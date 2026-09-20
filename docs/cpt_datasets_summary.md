# CPT Corpora — All Repositories

All six repository-specific document corpora have been generated. The batch preparation command independently validated the five additional corpora after their builds; RedHat had already passed the same validator.

| Repository | Corpus | Train documents | Validation documents | Excluded train identities | Collapsed exact-text aliases |
|---|---|---:|---:|---:|---:|
| RedHat | [`redhat_v1`](../data/training/cpt/redhat_v1/) | 251,668 | 2,529 | 2,303 | 13,832 |
| Apache | [`apache_v1`](../data/training/cpt/apache_v1/) | 854,420 | 8,651 | 4,875 | 10,535 |
| Jira | [`jira_v1`](../data/training/cpt/jira_v1/) | 209,074 | 2,074 | 539 | 8,411 |
| MongoDB | [`mongodb_v1`](../data/training/cpt/mongodb_v1/) | 106,270 | 1,056 | 354 | 5,371 |
| Qt | [`qt_v1`](../data/training/cpt/qt_v1/) | 119,636 | 1,164 | 837 | 481 |
| Mojang | [`mojang_v1`](../data/training/cpt/mojang_v1/) | 341,441 | 3,387 | 437 | 3,802 |

Totals across separate corpora: **1,882,509 training documents** and **18,861 validation documents**. These totals do not imply a pooled training experiment.

Each directory contains model-input JSONL shards, row-aligned provenance, `manifest.json`, `quality_report.json`, and `dataset_card.md`. Canonical SQLite files, exclusion ledgers, and duplicate groups are under `data/processed/cpt/<version>/`; issue-ID split maps are under `data/splits/cpt/<version>/`.

The same sanitation, exact-text deduplication, substantive-description grouping, and approximate 1% hash-group validation split were used for each repository. Each uses its own documented historical cutoff. No cross-repository duplicate screening or common-time pooled corpus is claimed.

For the detailed implementation and known limitations, see [cpt_dataset_implemented.md](cpt_dataset_implemented.md). The [machine-readable summary](../data/manifests/cpt_corpora_summary.json) preserves per-repository counts and exclusions.

## Reproduce or validate

From the project root, validate all completed versions without overwriting:

```bash
.venv/bin/python scripts/data/prepare_all_cpt_datasets.py --skip-existing
```

On a fresh checkout with the copied raw inputs and no versioned outputs, omit `--skip-existing` to build and validate sequentially. Select repositories with `--repositories Apache Jira MongoDB Qt Mojang`. Existing version directories are never overwritten.

To validate a single corpus:

```bash
.venv/bin/python scripts/data/validate_cpt_dataset.py --version apache_v1
```

To regenerate this summary from completed manifests:

```bash
.venv/bin/python scripts/data/summarize_cpt_datasets.py
```

Tokenization and block packing remain separate: no selected tokenizer is available locally yet. Model weights and checkpoints remain outside the project at `~/scratch/llms_model/ilr_llms/`.
