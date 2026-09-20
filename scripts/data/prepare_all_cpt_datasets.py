#!/usr/bin/env python3
"""Build and independently validate repository-specific CPT corpora sequentially."""
import argparse
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
REPOSITORIES = ("RedHat", "Apache", "Jira", "MongoDB", "Qt", "Mojang")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repositories", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    parser.add_argument("--skip-existing", action="store_true", help="Validate completed existing versions without rebuilding them")
    args = parser.parse_args()
    for repository in dict.fromkeys(args.repositories):
        config_path = ROOT / "configs/data" / f"cpt_{repository.lower()}.toml"
        config = tomllib.loads(config_path.read_text())
        version = config["version"]
        manifest = ROOT / "data/training/cpt" / version / "manifest.json"
        print(f"\n=== {repository}: {version} ===", flush=True)
        if args.skip_existing and manifest.exists():
            print("Existing corpus: validating without overwriting.", flush=True)
        else:
            subprocess.run([sys.executable, str(ROOT / "scripts/data/prepare_cpt_dataset.py"), "--config", str(config_path)], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/data/validate_cpt_dataset.py"), "--version", version], check=True)
    print("\nAll selected repository corpora passed independent validation.", flush=True)


if __name__ == "__main__":
    main()
