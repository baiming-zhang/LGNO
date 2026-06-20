import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = PROJECT_ROOT / "_run_logs" / f"robust_noise_{TIMESTAMP}"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

NOISE_LEVELS = [round(0.01 * i, 2) for i in range(11)]
METHODS = [
    ("LGNO", "LGNO.py"),
    ("MLP", "MLP.py"),
    ("MLPConv", "MLPConv.py"),
    ("CNN", "CNN.py"),
    ("OneShotPDE", "oneshot.py"),
    ("DeepONet", "DeepONet.py"),
    ("FNO", "FNO.py"),
    ("CNO", "CNO.py"),
    ("GNO", "GNO.py"),
]

REL_RE = re.compile(r"Test Relative L2 error:\s*([+-]?(?:inf|nan|\d+(?:\.\d*)?(?:e[+-]?\d+)?))", re.I)
MSE_RE = re.compile(r"Test MSE error\s*:\s*([+-]?(?:inf|nan|\d+(?:\.\d*)?(?:e[+-]?\d+)?))", re.I)

MASTER_LOG = LOG_ROOT / "master.log"
SUMMARY_CSV = LOG_ROOT / "relative_l2_summary.csv"
CURRENT_TXT = LOG_ROOT / "current.txt"
STATUS_JSON = LOG_ROOT / "status.json"


def now_iso():
    return datetime.now().astimezone().isoformat()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def write_status(state, message, **extra):
    data = {
        "state": state,
        "message": message,
        "log_root": str(LOG_ROOT),
        "updated_at": now_iso(),
    }
    data.update(extra)
    STATUS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_master(text):
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(text)


def run_command(label, cmd, log_name):
    log_path = LOG_ROOT / log_name
    start = time.time()
    start_iso = now_iso()
    CURRENT_TXT.write_text(f"running | {label}\n", encoding="utf-8")
    write_status("running", label, command=cmd, log=str(log_path))
    append_master(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] START {label}\n")
    append_master("COMMAND: " + " ".join(cmd) + "\n")

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            append_master(line)
        exit_code = proc.wait()

    duration = time.time() - start
    end_iso = now_iso()
    status = "OK" if exit_code == 0 else "FAIL"
    append_master(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END {label} | "
        f"status={status} | exit={exit_code} | duration_sec={duration:.3f}\n"
    )
    return {
        "status": status,
        "exit_code": exit_code,
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_sec": f"{duration:.3f}",
        "log": str(log_path),
    }


def parse_metric(log_path, regex):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    matches = regex.findall(text)
    return matches[-1] if matches else ""


def main():
    (LOG_ROOT / "runner.pid").write_text(str(os.getpid()), encoding="utf-8")
    (PROJECT_ROOT / "_run_logs" / "latest_robust_noise_path.txt").write_text(
        str(LOG_ROOT), encoding="utf-8"
    )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise",
                "method",
                "status",
                "exit_code",
                "relative_test_l2_error",
                "test_mse_error",
                "start_time",
                "end_time",
                "duration_sec",
                "log",
            ],
        )
        writer.writeheader()

    gen_result = run_command(
        "generate_data noise_0.00_to_0.10",
        [sys.executable, "-u", "generate_data.py"],
        "00_generate_data.log",
    )
    if gen_result["status"] != "OK":
        CURRENT_TXT.write_text("failed | generate_data.py\n", encoding="utf-8")
        write_status("failed", "generate_data.py failed", **gen_result)
        return 1

    total = len(NOISE_LEVELS) * len(METHODS)
    done = 0
    for noise in NOISE_LEVELS:
        noise_label = f"noise_{noise:.2f}"
        train_dir = f"dataset/{noise_label}/train_data"
        test_dir = f"dataset/{noise_label}/test_data"

        for method, script in METHODS:
            done += 1
            label = f"{done}/{total} | {noise_label} | {method}"
            log_name = f"{done:02d}_{safe_name(noise_label)}_{safe_name(method)}.log"
            result = run_command(
                label,
                [
                    sys.executable,
                    "-u",
                    script,
                    "--train_dir",
                    train_dir,
                    "--test_dir",
                    test_dir,
                ],
                log_name,
            )

            rel_l2 = parse_metric(result["log"], REL_RE)
            mse = parse_metric(result["log"], MSE_RE)
            with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "noise",
                        "method",
                        "status",
                        "exit_code",
                        "relative_test_l2_error",
                        "test_mse_error",
                        "start_time",
                        "end_time",
                        "duration_sec",
                        "log",
                    ],
                )
                writer.writerow(
                    {
                        "noise": f"{noise:.2f}",
                        "method": method,
                        "status": result["status"],
                        "exit_code": result["exit_code"],
                        "relative_test_l2_error": rel_l2,
                        "test_mse_error": mse,
                        "start_time": result["start_time"],
                        "end_time": result["end_time"],
                        "duration_sec": result["duration_sec"],
                        "log": result["log"],
                    }
                )

    CURRENT_TXT.write_text("complete | all robust noise runs finished\n", encoding="utf-8")
    write_status("complete", "all robust noise runs finished", summary=str(SUMMARY_CSV))
    append_master(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Robust noise run complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
