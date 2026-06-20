"""Evaluate trained LGNO models on robustness seeds 447--451."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
NOISE_LEVELS = [round(0.01 * index, 2) for index in range(11)]
SEEDS = list(range(447, 452))
NOISE_RE = re.compile(r"noise_([0-9]+\.[0-9]+)")


class LGNOModel(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = u.shape
        stencil = torch.cat(
            [
                torch.roll(u, shifts=(dy, dx), dims=(2, 3))
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ],
            dim=1,
        )
        local = stencil.permute(0, 2, 3, 1).reshape(-1, 9)
        weights = self.net(local)
        output = torch.sum(weights * local, dim=1, keepdim=True)
        return output.view(batch, 1, height, width)


def load_test(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    return data["u"].float().to(device), data["f"].float().to(device)


def find_runs() -> dict[float, Path]:
    result = {}
    for run_dir in sorted((ROOT / "LGNO_8hidden_results").glob("run_*")):
        config_path = run_dir / "config.txt"
        model_path = run_dir / "best_model.pth"
        if not config_path.is_file() or not model_path.is_file():
            continue
        match = NOISE_RE.search(
            config_path.read_text(encoding="utf-8", errors="replace")
        )
        if match:
            result[round(float(match.group(1)), 2)] = run_dir

    missing = [noise for noise in NOISE_LEVELS if noise not in result]
    if missing:
        raise RuntimeError(f"Missing trained noise levels: {missing}")
    return result


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = find_runs()
    rows = []

    for noise in NOISE_LEVELS:
        model = LGNOModel(hidden=8).to(device)
        model.load_state_dict(
            torch.load(runs[noise] / "best_model.pth", map_location=device)
        )
        model.eval()

        for seed in SEEDS:
            test_path = (
                ROOT
                / "dataset"
                / f"noise_{noise:.2f}"
                / f"eval_seed_{seed:02d}"
                / "test_data"
                / "test.pt"
            )
            u_true, f_true = load_test(test_path, device)
            with torch.no_grad():
                f_pred = model(u_true)
                rel_l2 = torch.linalg.vector_norm(f_pred - f_true) / (
                    torch.linalg.vector_norm(f_true) + 1e-20
                )
                mse = torch.mean((f_pred - f_true) ** 2)
            rows.append(
                {
                    "noise": noise,
                    "seed": seed,
                    "relative_test_l2_error": float(rel_l2.item()),
                    "test_mse_error": float(mse.item()),
                    "model_run": str(runs[noise].relative_to(ROOT)),
                }
            )

    long_path = ROOT / "lgno_seeded_test_results.csv"
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = ROOT / "lgno_seeded_test_summary.csv"
    fields = [
        "noise",
        "count",
        "mean_relative_test_l2_error",
        "min_relative_test_l2_error",
        "max_relative_test_l2_error",
        "std_relative_test_l2_error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for noise in NOISE_LEVELS:
            values = np.asarray(
                [
                    row["relative_test_l2_error"]
                    for row in rows
                    if row["noise"] == noise
                ],
                dtype=np.float64,
            )
            writer.writerow(
                {
                    "noise": f"{noise:.2f}",
                    "count": values.size,
                    "mean_relative_test_l2_error": f"{values.mean():.12e}",
                    "min_relative_test_l2_error": f"{values.min():.12e}",
                    "max_relative_test_l2_error": f"{values.max():.12e}",
                    "std_relative_test_l2_error": f"{values.std(ddof=0):.12e}",
                }
            )

    print(f"Long-form results: {long_path}")
    print(f"Summary results:   {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
