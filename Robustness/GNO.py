# Baseline wrapper for the Robust case.

import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))
os.chdir(CASE_DIR)

from aligned_baseline_common import run_static_cli


if __name__ == "__main__":
    run_static_cli(
        method="GNO",
        out_dir="GNO_{hidden}hidden_results",
        hidden=8,
        epochs=10000,
        lr=1e-2,
        patience=1000,
        in_channels=1,
        out_channels=1,
        ndim=2,
        batch_size=1,
        scheduler_patience=500,
        print_every=200,
        modes=12,
    )

