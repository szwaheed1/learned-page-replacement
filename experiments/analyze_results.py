#!/usr/bin/env python3
"""
analyze_results.py
==================
Reads the raw results produced by run_experiments.py and generates
publication-quality figures for the term paper.

Outputs are saved to results/figures/.

Usage:
    python experiments/analyze_results.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def load_json(filename):
    with open(os.path.join(RAW_DIR, filename)) as f:
        return json.load(f)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    main_results = load_json("main_results.json")
    sensitivity = load_json("sensitivity_results.json")
    fault_logs = load_json("fault_logs.json")
    trace_info = load_json("trace.json")
    shift_index = trace_info["shift_index"]

    plot_hit_ratio_comparison(main_results)
    plot_fault_count_comparison(main_results)
    plot_phase_comparison(main_results)
    plot_rolling_fault_rate(fault_logs, shift_index)
    plot_sensitivity(sensitivity)

    print(f"Figures saved to {os.path.abspath(FIG_DIR)}/")

    # Also print a nicely formatted table for the report
    print_results_table(main_results)


# -------------------------------------------------------------------- #
#  Plot 1: Overall hit-ratio bar chart
# -------------------------------------------------------------------- #
def plot_hit_ratio_comparison(results):
    algos = list(results.keys())
    hit_ratios = [results[a]["phase_metrics"]["overall"]["hit_ratio"] for a in algos]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(algos, hit_ratios, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Hit Ratio")
    ax.set_title("Overall Hit Ratio by Algorithm")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    for bar, val in zip(bars, hit_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "hit_ratio_comparison.png"), dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------- #
#  Plot 2: Fault count bar chart
# -------------------------------------------------------------------- #
def plot_fault_count_comparison(results):
    algos = list(results.keys())
    faults = [results[a]["faults"] for a in algos]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(algos, faults, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Page Faults")
    ax.set_title("Total Page Faults by Algorithm")

    for bar, val in zip(bars, faults):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(val), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fault_count_comparison.png"), dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------- #
#  Plot 3: Phase 1 vs Phase 2 grouped bar chart
# -------------------------------------------------------------------- #
def plot_phase_comparison(results):
    algos = list(results.keys())
    p1_hr = [results[a]["phase_metrics"]["phase1"]["hit_ratio"] for a in algos]
    p2_hr = [results[a]["phase_metrics"]["phase2"]["hit_ratio"] for a in algos]

    x = np.arange(len(algos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, p1_hr, width, label="Phase 1 (Locality)",
                   color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, p2_hr, width, label="Phase 2 (Random/Bursty)",
                   color="#DD8452", edgecolor="white")

    ax.set_ylabel("Hit Ratio")
    ax.set_title("Hit Ratio Before and After Workload Shift")
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase_comparison.png"), dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------- #
#  Plot 4: Rolling fault rate over time (shows the shift)
# -------------------------------------------------------------------- #
def plot_rolling_fault_rate(fault_logs, shift_index, window=200):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"FIFO": "#4C72B0", "LRU": "#55A868",
              "Optimal": "#C44E52", "Learned (DT)": "#8172B2"}

    for algo, log in fault_logs.items():
        arr = np.array(log, dtype=float)
        # Compute rolling average fault rate
        kernel = np.ones(window) / window
        rolling = np.convolve(arr, kernel, mode="valid")
        ax.plot(rolling, label=algo, color=colors.get(algo, None), linewidth=1.2)

    ax.axvline(x=shift_index - window // 2, color="red", linestyle="--",
               linewidth=1.5, label="Workload shift")
    ax.set_xlabel("Access Index")
    ax.set_ylabel(f"Fault Rate (rolling window = {window})")
    ax.set_title("Rolling Fault Rate Over Time")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rolling_fault_rate.png"), dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------- #
#  Plot 5: Sensitivity to frame count
# -------------------------------------------------------------------- #
def plot_sensitivity(sensitivity):
    frame_counts = sorted(int(k) for k in sensitivity.keys())
    algos = list(sensitivity[str(frame_counts[0])].keys())

    colors = {"FIFO": "#4C72B0", "LRU": "#55A868",
              "Optimal": "#C44E52", "Learned (DT)": "#8172B2"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in algos:
        hr = [sensitivity[str(nf)][algo]["overall"]["hit_ratio"] for nf in frame_counts]
        ax.plot(frame_counts, hr, marker="o", label=algo,
                color=colors.get(algo, None), linewidth=1.8)

    ax.set_xlabel("Number of Frames")
    ax.set_ylabel("Overall Hit Ratio")
    ax.set_title("Hit Ratio vs. Frame Count")
    ax.set_xticks(frame_counts)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sensitivity_frame_count.png"), dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------- #
#  Console table
# -------------------------------------------------------------------- #
def print_results_table(results):
    print("\n" + "=" * 72)
    print(f"{'Algorithm':<16} {'Faults':>8} {'Hits':>8} {'Hit Ratio':>10} "
          f"{'P1 HR':>8} {'P2 HR':>8}")
    print("-" * 72)
    for algo, r in results.items():
        pm = r["phase_metrics"]
        print(f"{algo:<16} {r['faults']:>8} {r['hits']:>8} "
              f"{pm['overall']['hit_ratio']:>10.4f} "
              f"{pm['phase1']['hit_ratio']:>8.4f} "
              f"{pm['phase2']['hit_ratio']:>8.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
