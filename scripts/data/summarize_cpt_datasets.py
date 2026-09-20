#!/usr/bin/env python3
"""Summarize completed, separately versioned CPT corpora from their manifests."""
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
REPOSITORIES = ("RedHat", "Apache", "Jira", "MongoDB", "Qt", "Mojang")


def main():
    records = []
    for repository in REPOSITORIES:
        config = tomllib.loads((ROOT / "configs/data" / f"cpt_{repository.lower()}.toml").read_text())
        version = config["version"]
        corpus = ROOT / "data/training/cpt" / version
        manifest = json.loads((corpus / "manifest.json").read_text())
        quality = json.loads((corpus / "quality_report.json").read_text())
        if manifest["build_status"] != "validated" or not manifest["validation"]["passed"]:
            raise ValueError(f"{repository} is not validated")
        if manifest["config"] != config:
            raise ValueError(f"{repository} config differs from its built manifest")
        records.append({"repository": repository, "version": version,
                        "corpus_path": str(corpus.relative_to(ROOT)),
                        "counts": manifest["counts"], "output_counts": manifest["output_counts"],
                        "exclusions": quality["exclusions"], "cutoff": config["cutoff"],
                        "validation": manifest["validation"]})
    summary = {"scope": "Six separate repository corpora, not a pooled or cross-repository-deduplicated dataset.",
               "tokenization": "Not performed; counts are documents and whitespace words, not model tokens.",
               "corpora": records,
               "total_train_documents": sum(r["output_counts"]["train"] for r in records),
               "total_validation_documents": sum(r["output_counts"]["validation"] for r in records)}
    output = ROOT / "data/manifests/cpt_corpora_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = ["| Repository | Corpus | Train documents | Validation documents | Excluded train identities | Collapsed exact-text aliases |",
            "|---|---|---:|---:|---:|---:|"]
    for record in records:
        o, c = record["output_counts"], record["counts"]
        rows.append(f"| {record['repository']} | [`{record['version']}`](../{record['corpus_path']}/) | {o['train']:,} | {o['validation']:,} | {c['excluded_train_identities']:,} | {c['exact_text_duplicate_extra_identities']:,} |")
    report = "\n".join(rows)
    doc = f'''# CPT Corpora — All Repositories

All six repository-specific document corpora have been generated. The batch preparation command independently validated the five additional corpora after their builds; RedHat had already passed the same validator.

{report}

Totals across separate corpora: **{summary['total_train_documents']:,} training documents** and **{summary['total_validation_documents']:,} validation documents**. These totals do not imply a pooled training experiment.

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
'''
    (ROOT / "docs/cpt_datasets_summary.md").write_text(doc)
    print(report)
    print(f"\nTotal train={summary['total_train_documents']:,}; validation={summary['total_validation_documents']:,}")


if __name__ == "__main__":
    main()
