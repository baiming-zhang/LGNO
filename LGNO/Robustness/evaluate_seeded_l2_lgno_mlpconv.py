from pathlib import Path
import csv
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import torch
import torch.nn as nn


FONT_SIZE = 36
LEGEND_FONT_SIZE = 23
ERROR_BAR_STD_SCALE = 0.30
METHODS = ["LGNO", "MLPConv"]
NOISES = [round(i / 100, 2) for i in range(11)]
SEEDS = list(range(447, 452))
NOISE_PATTERN = re.compile(r"noise_([0-9]+\.[0-9]+)")

METHOD_STYLE = {
    "LGNO": {
        "color": "#2F5385",
        "marker": "o",
        "linestyle": "-",
    },
    "MLPConv": {
        "color": "#86ABD8",
        "marker": "s",
        "linestyle": "--",
    },
}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.titlesize"] = FONT_SIZE
plt.rcParams["axes.labelsize"] = FONT_SIZE
plt.rcParams["xtick.labelsize"] = FONT_SIZE
plt.rcParams["ytick.labelsize"] = FONT_SIZE
plt.rcParams["legend.fontsize"] = FONT_SIZE - 2
plt.rcParams["figure.titlesize"] = FONT_SIZE
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.10
plt.rcParams["svg.fonttype"] = "none"


class LGNOModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, u):
        batch, _, height, width = u.shape
        stencil = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                stencil.append(torch.roll(u, shifts=(dy, dx), dims=(2, 3)))

        stencil = torch.cat(stencil, dim=1)
        local = stencil.permute(0, 2, 3, 1).reshape(-1, 9)
        weights = self.net(local)
        out = torch.sum(weights * local, dim=1, keepdim=True)
        return out.view(batch, 1, height, width)


class MLPConvModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, u):
        batch, _, height, width = u.shape
        stencil = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                stencil.append(torch.roll(u, shifts=(dy, dx), dims=(2, 3)))

        stencil = torch.cat(stencil, dim=1)
        local = stencil.permute(0, 2, 3, 1).reshape(-1, 9)
        out = self.net(local)
        return out.view(batch, 1, height, width)


def load_dataset(test_dir, device):
    data_path = Path(test_dir) / "test.pt"
    data = torch.load(data_path, map_location=device)
    u = data["u"].float()
    f = data["f"].float()

    if u.dim() == 3:
        u = u.unsqueeze(1)
        f = f.unsqueeze(1)
    elif u.dim() == 2:
        u = u.unsqueeze(0).unsqueeze(0)
        f = f.unsqueeze(0).unsqueeze(0)
    elif u.dim() != 4:
        raise ValueError(f"Unsupported tensor shape: {tuple(u.shape)}")

    return u.to(device), f.to(device)


def read_noise_from_config(config_path):
    text = config_path.read_text(encoding="utf-8", errors="replace")
    match = NOISE_PATTERN.search(text)
    if not match:
        return None
    return round(float(match.group(1)), 2)


def find_trained_runs(result_root):
    runs = {}
    for run_dir in sorted(result_root.glob("run_*")):
        config_path = run_dir / "config.txt"
        model_path = run_dir / "best_model.pth"
        if not config_path.is_file() or not model_path.is_file():
            continue
        noise = read_noise_from_config(config_path)
        if noise is not None:
            runs[noise] = run_dir

    missing = [noise for noise in NOISES if noise not in runs]
    if missing:
        missing_text = ", ".join(f"{noise:.2f}" for noise in missing)
        raise RuntimeError(f"Missing trained runs in {result_root}: {missing_text}")

    return {noise: runs[noise] for noise in NOISES}


def make_model(method, device):
    if method == "LGNO":
        return LGNOModel(hidden=8).to(device)
    if method == "MLPConv":
        return MLPConvModel(hidden=8).to(device)
    raise ValueError(method)


def evaluate_all(script_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = script_dir / "dataset"
    run_roots = {
        "LGNO": script_dir / "LGNO_8hidden_results",
        "MLPConv": script_dir / "MLPConv_8hidden_results",
    }
    trained_runs = {method: find_trained_runs(root) for method, root in run_roots.items()}

    rows = []
    for method in METHODS:
        for noise in NOISES:
            model = make_model(method, device)
            model_path = trained_runs[method][noise] / "best_model.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            for seed in SEEDS:
                test_dir = dataset_root / f"noise_{noise:.2f}" / f"eval_seed_{seed:02d}" / "test_data"
                u_test, f_test = load_dataset(test_dir, device)
                with torch.no_grad():
                    f_pred = model(u_test)
                    rel_l2 = torch.norm(f_pred - f_test) / (torch.norm(f_test) + 1e-20)
                    mse = torch.mean((f_pred - f_test) ** 2)

                row = {
                    "method": method,
                    "noise": noise,
                    "seed": seed,
                    "relative_test_l2_error": float(rel_l2.item()),
                    "test_mse_error": float(mse.item()),
                    "model_run": str(trained_runs[method][noise]),
                    "test_dir": str(test_dir),
                }
                rows.append(row)
                print(
                    f"{method:7s} noise={noise:.2f} seed={seed:02d} "
                    f"rel_l2={row['relative_test_l2_error']:.6e}"
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows


def summarize(rows):
    summary = []
    for method in METHODS:
        for noise in NOISES:
            values = np.array(
                [
                    row["relative_test_l2_error"]
                    for row in rows
                    if row["method"] == method and row["noise"] == noise
                ],
                dtype=np.float64,
            )
            summary.append(
                {
                    "method": method,
                    "noise": noise,
                    "count": values.size,
                    "mean_relative_test_l2_error": values.mean(),
                    "min_relative_test_l2_error": values.min(),
                    "max_relative_test_l2_error": values.max(),
                    "std_relative_test_l2_error": values.std(ddof=0),
                }
            )
    return summary


def write_long_csv(rows, output_path):
    fieldnames = [
        "method",
        "noise",
        "seed",
        "relative_test_l2_error",
        "test_mse_error",
        "model_run",
        "test_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "noise": f"{row['noise']:.2f}",
                    "relative_test_l2_error": f"{row['relative_test_l2_error']:.12e}",
                    "test_mse_error": f"{row['test_mse_error']:.12e}",
                }
            )


def write_summary_csv(summary, output_path):
    fieldnames = [
        "method",
        "noise",
        "count",
        "mean_relative_test_l2_error",
        "min_relative_test_l2_error",
        "max_relative_test_l2_error",
        "std_relative_test_l2_error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    **row,
                    "noise": f"{row['noise']:.2f}",
                    "mean_relative_test_l2_error": f"{row['mean_relative_test_l2_error']:.12e}",
                    "min_relative_test_l2_error": f"{row['min_relative_test_l2_error']:.12e}",
                    "max_relative_test_l2_error": f"{row['max_relative_test_l2_error']:.12e}",
                    "std_relative_test_l2_error": f"{row['std_relative_test_l2_error']:.12e}",
                }
            )


def plot_summary(summary, output_stem):
    fig, ax = plt.subplots(figsize=(13.8, 7.0))

    for method in METHODS:
        method_rows = [row for row in summary if row["method"] == method]
        noises = np.array([row["noise"] for row in method_rows], dtype=np.float64)
        mean = np.array([row["mean_relative_test_l2_error"] for row in method_rows], dtype=np.float64)
        std = np.array([row["std_relative_test_l2_error"] for row in method_rows], dtype=np.float64)

        style = METHOD_STYLE[method]
        ax.errorbar(
            noises,
            mean,
            yerr=ERROR_BAR_STD_SCALE * std,
            color=style["color"],
            linewidth=4.2,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=14,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=2.6,
            elinewidth=2.6,
            capsize=6.0,
            capthick=2.6,
            label=method,
            zorder=3,
        )

    ax.set_xlabel("Noise level")
    ax.set_ylabel(r"Relative test $L^2$ error")
    ax.set_xlim(min(NOISES) - 0.003, max(NOISES) + 0.003)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(NOISES)
    ax.set_xticklabels([f"{noise:.2f}" for noise in NOISES], rotation=45, ha="right")
    x_label_offset = ScaledTranslation(8 / 72, 0, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + x_label_offset)
    ax.grid(False)

    legend_x0 = 1.04
    legend_x1 = 1.19
    legend_xm = 0.5 * (legend_x0 + legend_x1)
    legend_err = 0.055
    legend_cap = 0.020

    def draw_legend_item(method, y_line, y_text):
        style = METHOD_STYLE[method]
        color = style["color"]
        ax.plot(
            [legend_x0, legend_x1],
            [y_line, y_line],
            color=color,
            linewidth=4.2,
            linestyle=style["linestyle"],
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(
            [legend_xm, legend_xm],
            [y_line - legend_err, y_line + legend_err],
            color=color,
            linewidth=2.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(
            [legend_xm - legend_cap, legend_xm + legend_cap],
            [y_line - legend_err, y_line - legend_err],
            color=color,
            linewidth=2.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(
            [legend_xm - legend_cap, legend_xm + legend_cap],
            [y_line + legend_err, y_line + legend_err],
            color=color,
            linewidth=2.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.plot(
            [legend_xm],
            [y_line],
            linestyle="none",
            marker=style["marker"],
            markersize=14,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2.6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            legend_xm,
            y_text,
            method,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONT_SIZE,
            color="black",
            clip_on=False,
        )

    draw_legend_item("LGNO", y_line=0.52, y_text=0.36)
    draw_legend_item("MLPConv", y_line=0.18, y_text=0.03)

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.svg")
    fig.savefig(f"{output_stem}.pdf")
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    rows = evaluate_all(script_dir)
    summary = summarize(rows)

    long_csv = script_dir / "lgno_mlpconv_seeded_test_l2_long.csv"
    summary_csv = script_dir / "lgno_mlpconv_seeded_test_l2_summary.csv"
    output_stem = script_dir / "lgno_mlpconv_seeded_test_l2_vs_noise"

    write_long_csv(rows, long_csv)
    write_summary_csv(summary, summary_csv)
    plot_summary(summary, output_stem)

    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {output_stem}.png")
    print(f"Saved: {output_stem}.svg")
    print(f"Saved: {output_stem}.pdf")


if __name__ == "__main__":
    main()
