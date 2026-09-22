#!/usr/bin/env python3
"""Build pointwise and set-retrieval SFT datasets from the six issue repositories."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ilr_post_training.data.sft import main

if __name__ == "__main__":
    main()
