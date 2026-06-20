"""Compatibility launcher for the 3D Houdini LGNO implementation."""

import sys
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))

from lgno_common import run_case_entry


if __name__ == "__main__":
    raise SystemExit(run_case_entry("houdini-3d"))
