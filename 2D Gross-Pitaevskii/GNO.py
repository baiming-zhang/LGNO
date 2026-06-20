# Baseline wrapper for the 2D Shordinger (Gross–Pitaevskii) case.

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
        hidden=12,
        epochs=2000000,
        lr=1e-3,
        patience=10000,
        in_channels=3,
        out_channels=2,
        ndim=2,
        batch_size=1,
        scheduler_patience=2000,
        weight_decay=1e-6,
        grad_clip=1.0,
        print_every=200,
        modes=12,
    )

