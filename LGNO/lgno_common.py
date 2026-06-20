"""Shared launcher for the case-specific LGNO implementations.

The numerical implementations intentionally remain case-specific. This module
only resolves a case name, translates a small set of uniform CLI aliases, and
starts the original implementation in its original working directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LGNOCase:
    key: str
    case_dir: str
    implementation: str
    description: str
    aliases: tuple[str, ...] = ()
    train_flag: str | None = None
    test_flag: str | None = None
    output_flag: str | None = None
    option_aliases: tuple[tuple[str, str], ...] = ()

    @property
    def directory(self) -> Path:
        return ROOT / self.case_dir

    @property
    def script(self) -> Path:
        return self.directory / self.implementation


CASES: tuple[LGNOCase, ...] = (
    LGNOCase(
        key="diffusion-1d",
        aliases=("diffusion", "1d-diffusion"),
        case_dir="1D Diffusion",
        implementation="_lgno_impl.py",
        description="1D periodic diffusion time marching",
        train_flag="--data_dir",
        test_flag="--test_path",
        output_flag="--save_dir",
        option_aliases=(
            ("--max-rollout-steps", "--max_rollout_steps"),
            ("--num-rollout-starts", "--num_rollout_starts"),
            ("--warmup-ratio", "--warmup_ratio"),
            ("--lambda-deriv", "--lambda_deriv"),
            ("--lambda-roll", "--lambda_roll"),
            ("--grad-clip", "--grad_clip"),
            ("--learn-coeff", "--learn_coeff"),
        ),
    ),
    LGNOCase(
        key="burgers-2d",
        aliases=("burgers", "2d-burgers"),
        case_dir="2D Burgers",
        implementation="_lgno_impl.py",
        description="2D periodic static scalar operator",
        train_flag="--train_dir",
        test_flag="--test_dir",
        output_flag="--run_dir",
        option_aliases=(("--eval-only", "--eval_only"),),
    ),
    LGNOCase(
        key="navier-stokes-2d",
        aliases=("navier", "navier-stokes", "2d-navier-stokes"),
        case_dir="2D Navier-Stokes",
        implementation="_lgno_impl.py",
        description="2D periodic Navier-Stokes derivative and rollout",
        train_flag="--train_dir",
        test_flag="--test_dir",
        output_flag="--run_dir",
        option_aliases=(
            ("--batch-size", "--batch_size"),
            ("--save-stride", "--save_stride"),
            ("--eval-only", "--eval_only"),
        ),
    ),
    LGNOCase(
        key="gpe-2d-folded",
        aliases=("gpe", "gpe-folded", "2d-gpe-folded"),
        case_dir="2D Gross-Pitaevskii",
        implementation="_lgno_impl.py",
        description="2D Gross-Pitaevskii folded LGNO",
        train_flag="--train_dir",
        test_flag="--test_dir",
        output_flag="--out_dir",
        option_aliases=(
            ("--min-lr", "--min_lr"),
            ("--weight-decay", "--weight_decay"),
            ("--grad-clip", "--grad_clip"),
            ("--last-scale", "--last_scale"),
            ("--scheduler-patience", "--scheduler_patience"),
            ("--log-every", "--log_every"),
            ("--save-weights", "--save_weights"),
        ),
    ),
    LGNOCase(
        key="gpe-2d-unfolded",
        aliases=("gpe-unfolded", "2d-gpe-unfolded"),
        case_dir="2D Gross-Pitaevskii",
        implementation="_lgno_unfolded_impl.py",
        description="2D Gross-Pitaevskii unfolded LGNO",
        train_flag="--train_dir",
        test_flag="--test_dir",
        output_flag="--out_dir",
        option_aliases=(
            ("--min-lr", "--min_lr"),
            ("--weight-decay", "--weight_decay"),
            ("--grad-clip", "--grad_clip"),
            ("--last-scale", "--last_scale"),
            ("--scheduler-patience", "--scheduler_patience"),
            ("--log-every", "--log_every"),
            ("--save-weights", "--save_weights"),
        ),
    ),
    LGNOCase(
        key="perturbed-harmonic-3d",
        aliases=("harmonic-3d", "schrodinger-3d"),
        case_dir="3D Schrodinger (Perturbed Harmonic)",
        implementation="_lgno_impl.py",
        description="3D perturbed-harmonic Schrodinger operator",
        train_flag="--train_dir",
        test_flag="--test_dir",
        option_aliases=(
            ("--batch-size", "--batch_size"),
            ("--test-batch-size", "--test_batch_size"),
        ),
    ),
    LGNOCase(
        key="houdini-3d",
        aliases=("houdini", "smoke-3d"),
        case_dir="3D Houdini (Smoke collision)",
        implementation="_lgno_impl.py",
        description="3D Houdini smoke rollout",
    ),
    LGNOCase(
        key="robust",
        aliases=("robustness", "noise"),
        case_dir="Robustness",
        implementation="_lgno_impl.py",
        description="2D robustness/noise benchmark",
        train_flag="--train_dir",
        test_flag="--test_dir",
    ),
)


def _normalized_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "-")
    )


def get_case(name: str) -> LGNOCase:
    wanted = _normalized_name(name)
    for spec in CASES:
        names = (spec.key, *spec.aliases, spec.case_dir)
        if wanted in {_normalized_name(item) for item in names}:
            return spec
    available = ", ".join(spec.key for spec in CASES)
    raise ValueError(f"Unknown LGNO case '{name}'. Available cases: {available}")


def _replace_option(argument: str, source: str, target: str) -> str | None:
    if argument == source:
        return target
    prefix = source + "="
    if argument.startswith(prefix):
        return target + "=" + argument[len(prefix) :]
    return None


def translate_uniform_args(spec: LGNOCase, arguments: Sequence[str]) -> list[str]:
    """Translate documented uniform aliases without changing legacy arguments."""

    mappings: list[tuple[str, str | None]] = [
        ("--train-data", spec.train_flag),
        ("--test-data", spec.test_flag),
        ("--output", spec.output_flag),
        *spec.option_aliases,
    ]
    translated: list[str] = []
    for argument in arguments:
        replacement = None
        matched_source = None
        for source, target in mappings:
            candidate = _replace_option(argument, source, target or source)
            if candidate is not None:
                matched_source = source
                if target is None:
                    raise ValueError(
                        f"{source} is not supported by case '{spec.key}'. "
                        "Use that implementation's native configuration."
                    )
                replacement = candidate
                break
        translated.append(replacement if matched_source is not None else argument)
    return translated


def build_command(
    case: str | LGNOCase,
    arguments: Sequence[str] = (),
) -> tuple[LGNOCase, list[str]]:
    spec = get_case(case) if isinstance(case, str) else case
    if not spec.script.is_file():
        raise FileNotFoundError(f"Missing LGNO implementation: {spec.script}")
    child_args = translate_uniform_args(spec, arguments)
    return spec, [sys.executable, "-u", str(spec.script), *child_args]


def format_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command))


def run_case(
    case: str | LGNOCase,
    arguments: Sequence[str] = (),
    *,
    dry_run: bool = False,
) -> int:
    spec, command = build_command(case, arguments)
    if dry_run:
        print(f"cwd: {spec.directory}")
        print(f"command: {format_command(command)}")
        return 0
    completed = subprocess.run(command, cwd=os.fspath(spec.directory), check=False)
    return int(completed.returncode)


def run_case_entry(case: str, arguments: Sequence[str] | None = None) -> int:
    return run_case(case, sys.argv[1:] if arguments is None else arguments)


def print_case_list() -> None:
    width = max(len(spec.key) for spec in CASES)
    for spec in CASES:
        print(f"{spec.key:<{width}}  {spec.description}")


def print_usage() -> None:
    print(
        "Usage:\n"
        "  python run_lgno.py --list\n"
        "  python run_lgno.py [--dry-run] CASE [LGNO arguments...]\n\n"
        "Uniform aliases (translated only when supported):\n"
        "  --train-data PATH   training data directory/path\n"
        "  --test-data PATH    test data directory/path\n"
        "  --output PATH       result directory\n\n"
        "All original case-specific arguments remain valid and are forwarded unchanged.\n"
        "Put --help after CASE to display the original implementation's help."
    )


def main(arguments: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if not args or args[0] in {"-h", "--help"}:
        print_usage()
        return 0
    if args[0] == "--list":
        print_case_list()
        return 0

    dry_run = False
    if args[0] == "--dry-run":
        dry_run = True
        args.pop(0)
    if not args:
        print_usage()
        return 2

    case = args.pop(0)
    try:
        return run_case(case, args, dry_run=dry_run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
