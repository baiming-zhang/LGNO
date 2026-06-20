# Baseline wrapper for the 1D Diffusion case.

import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))
os.chdir(CASE_DIR)

from aligned_baseline_common import run_diffusion_cli


if __name__ == "__main__":
    run_diffusion_cli("OneShotPDE", "OneShotPDE_{hidden}hidden_results")

