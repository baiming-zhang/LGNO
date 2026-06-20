"""Generate all robustness datasets and train LGNO at every noise level."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOISE_LEVELS = [round(0.01 * index, 2) for index in range(11)]
REL_RE = re.compile(r"Test Relative L2 error:\s*([0-9.eE+-]+)")
MSE_RE = re.compile(r"Test MSE error\s*:\s*([0-9.eE+-]+)")


def latest_run(result_root: Path, started_at: float) -> Path:
    candidates = [
        path
        for path in result_root.glob("run_*")
        if path.is_dir() and path.stat().st_mtime >= started_at - 2.0
    ]
    if not candidates:
        raise RuntimeError(f"Cannot find the newly created run in {result_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_last_metric(log_path: Path, pattern: re.Pattern[str]) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = pattern.findall(text)
    return matches[-1] if matches else ""


def main() -> int:
    subprocess.run(
        [sys.executable, "-u", "generate_data.py"],
        cwd=ROOT,
        check=True,
    )

    summary_path = ROOT / "lgno_noise_training_summary.csv"
    fields = [
        "noise",
        "run_dir",
        "relative_test_l2_error",
        "test_mse_error",
        "exit_code",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    failures = 0
    for noise in NOISE_LEVELS:
        label = f"noise_{noise:.2f}"
        train_dir = f"dataset/{label}/train_data"
        test_dir = f"dataset/{label}/test_data"
        print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] training LGNO for {label}")

        started_at = datetime.now().timestamp()
        completed = subprocess.run(
            [
                sys.executable,
                "-u",
                "LGNO.py",
                "--train_dir",
                train_dir,
                "--test_dir",
                test_dir,
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode != 0:
            failures += 1

        run_dir = latest_run(ROOT / "LGNO_8hidden_results", started_at)
        log_path = run_dir / "train_log.txt"
        row = {
            "noise": f"{noise:.2f}",
            "run_dir": str(run_dir.relative_to(ROOT)),
            "relative_test_l2_error": read_last_metric(log_path, REL_RE),
            "test_mse_error": read_last_metric(log_path, MSE_RE),
            "exit_code": completed.returncode,
        }
        with summary_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow(row)

    print(f"\nSummary written to {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
