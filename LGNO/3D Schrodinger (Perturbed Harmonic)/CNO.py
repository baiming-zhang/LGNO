# Baseline wrapper for the 3D Shordinger (Perturbed Harmonic) case.

import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))
os.chdir(CASE_DIR)

from aligned_baseline_common import run_static_cli


if __name__ == "__main__":
    run_static_cli(
        method="CNO",
        out_dir="CNO_{hidden}hidden_results",
        hidden=16,
        epochs=100000,
        lr=1e-2,
        patience=3000,
        in_channels=3,
        out_channels=2,
        ndim=3,
        batch_size=8,
        test_batch_size=8,
        scheduler_patience=500,
        grad_clip=1.0,
        print_every=200,
        modes=6,
    )

