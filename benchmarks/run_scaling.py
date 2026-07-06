"""
Benchmark FMM vs direct O(N^2) summation and plot runtime scaling.

Builds the C++ benchmark driver if needed, runs it across a range of N,
and produces docs/scaling.png (log-log runtime plot) plus a printed table
including max relative error of FMM vs direct at each N.

Run from the repo root:
    python3 benchmarks/run_scaling.py
"""

import csv
import io
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def ensure_built():
    subprocess.run(
        ["cmake", "-B", str(BUILD), "-DFMM_BUILD_TESTS=OFF", "-DFMM_BUILD_BENCH=ON"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--build", str(BUILD), "-j4", "--target", "fmm_bench"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_bench():
    out = subprocess.run(
        [str(BUILD / "fmm_bench")], check=True, capture_output=True, text=True
    ).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    for r in rows:
        r["N"] = int(r["N"])
        r["t_fmm_ms"] = float(r["t_fmm_ms"])
        r["t_direct_ms"] = float(r["t_direct_ms"])
        r["max_rel_err"] = float(r["max_rel_err"])
    return rows


def plot(rows, out_path: Path):
    N = [r["N"] for r in rows]
    t_fmm = [r["t_fmm_ms"] for r in rows]
    t_dir = [r["t_direct_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N, t_dir, marker="o", label="Direct $O(N^2)$", color="#e07a5f")
    ax.loglog(N, t_fmm, marker="o", label="FMM (p=4)", color="#3d5a80")

    # Reference slopes: anchor each to that method's LAST data point, where
    # it is actually in its asymptotic regime. Anchoring to the first point
    # is misleading for FMM, whose small-N runtime is dominated by fixed
    # per-level overhead rather than the O(N) term.
    n_end, t_end = N[-1], t_dir[-1]
    ax.loglog(
        N,
        [t_end * (n / n_end) ** 2 for n in N],
        "--",
        lw=1,
        color="#e07a5f",
        alpha=0.5,
        label="$\\propto N^2$",
    )
    n_endf, t_endf = N[-1], t_fmm[-1]
    ax.loglog(
        N,
        [t_endf * (n / n_endf) for n in N],
        "--",
        lw=1,
        color="#3d5a80",
        alpha=0.5,
        label="$\\propto N$",
    )

    ax.set_xlabel("Number of particles N")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("FMM vs direct summation: runtime scaling")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ensure_built()
    rows = run_bench()
    print(
        f"{'N':>8} {'FMM (ms)':>12} {'Direct (ms)':>12} {'speedup':>9} {'max rel err':>12}"
    )
    for r in rows:
        speedup = r["t_direct_ms"] / r["t_fmm_ms"]
        print(
            f"{r['N']:>8} {r['t_fmm_ms']:>12.1f} {r['t_direct_ms']:>12.1f} "
            f"{speedup:>8.1f}x {r['max_rel_err']:>12.2e}"
        )
    plot(rows, ROOT / "docs" / "scaling.png")
