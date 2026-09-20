# CPT data configuration

`cpt_redhat.toml` builds the first RedHat processed-snapshot CPT corpus.

Run from the project root with the existing Python 3.12 environment:

```bash
.venv/bin/python scripts/data/prepare_cpt_dataset.py
.venv/bin/python scripts/data/validate_cpt_dataset.py --version redhat_v1
.venv/bin/python -m unittest discover -s tests -v
```

The builder refuses to overwrite an existing version. To change cleaning, splitting,
or source data, copy the configuration to a new file, assign a new `version`, and use:

```bash
.venv/bin/python scripts/data/prepare_cpt_dataset.py --config configs/data/cpt_redhat_v2.toml
```

The v2 command is an example; that configuration has not been created. Configuration
paths are relative to the shell when explicitly supplied; paths inside the TOML are
relative to the project root. The default config is resolved from the script location.

The output contains only title/description text, not relation labels or chat turns.
The 1% validation setting applies to hashed text groups, not an exact document count.
Substantive identical descriptions (50 or more whitespace words) share a split; those
matching a held-out description are excluded. Similarity-based near-duplicate removal
is not enabled. See the implementation note for actual counts and limitations.

## Other repositories

The same preparation protocol is configured in `cpt_apache.toml`, `cpt_jira.toml`,
`cpt_mongodb.toml`, `cpt_qt.toml`, and `cpt_mojang.toml`. Each has its own source
folder, version name, and source-derived temporal cutoff. They produce separate
corpora; they do not pool data across repositories.

Build missing versions and validate any versions already completed:

```bash
.venv/bin/python scripts/data/prepare_all_cpt_datasets.py --skip-existing
```

`--skip-existing` skips only the build of a completed corpus, not its validation.
A partial output directory without a completed manifest is not treated as valid.
Do not run concurrent builds of the same version.

Build selected repositories on a fresh output directory:

```bash
.venv/bin/python scripts/data/prepare_all_cpt_datasets.py --repositories Apache Jira MongoDB Qt Mojang
```

Aggregate the completed corpus manifests:

```bash
.venv/bin/python scripts/data/summarize_cpt_datasets.py
```
