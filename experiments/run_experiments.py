#!/usr/bin/env python3
"""
run_experiments.py
==================
Runs all four page-replacement policies (FIFO, LRU, Optimal, Learned)
on the same synthetic workload and writes raw results to results/raw/.

Usage:
    python experiments/run_experiments.py
"""

import json
import os
import sys

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.page_replacement import (
    FIFOPageReplacement,
    LRUPageReplacement,
    OptimalPageReplacement,
    LearnedPageReplacement,
)
from src.workload import WorkloadGenerator
from src.utils import compute_phase_metrics


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #
NUM_FRAMES = 8
NUM_PAGES = 50
WORKING_SET = 10
TRACE_LENGTH = 10000
SEED = 42
TRAINING_TRACE_LENGTH = 8000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "raw")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Generate workload ----------------------------------------- #
    gen = WorkloadGenerator(
        num_pages=NUM_PAGES,
        working_set_size=WORKING_SET,
        trace_length=TRACE_LENGTH,
        seed=SEED,
    )
    trace, shift_index = gen.generate()
    training_trace = gen.generate_training_trace(length=TRAINING_TRACE_LENGTH)

    print(f"Trace length       : {len(trace)}")
    print(f"Shift point        : {shift_index}")
    print(f"Training trace len : {len(training_trace)}")
    print(f"Frame count        : {NUM_FRAMES}")
    print(f"Page space         : 0 .. {NUM_PAGES - 1}")
    print()

    # ---- Classical algorithms -------------------------------------- #
    algorithms = {
        "FIFO": FIFOPageReplacement(NUM_FRAMES),
        "LRU": LRUPageReplacement(NUM_FRAMES),
        "Optimal": OptimalPageReplacement(NUM_FRAMES),
    }

    all_results = {}

    for name, algo in algorithms.items():
        result = algo.run(trace)
        phase_metrics = compute_phase_metrics(result["fault_log"], shift_index)
        result["phase_metrics"] = phase_metrics
        all_results[name] = result
        _print_result(name, phase_metrics)

    # ---- Learned (Decision Tree) ----------------------------------- #
    # Train on a separate trace using Optimal's eviction decisions
    opt_trainer = OptimalPageReplacement(NUM_FRAMES)
    decisions = opt_trainer.get_eviction_decisions(training_trace, NUM_FRAMES)
    print(f"Training samples from Optimal: {len(decisions)} eviction events")

    learned = LearnedPageReplacement(NUM_FRAMES)
    learned.train(decisions)
    result = learned.run(trace)
    phase_metrics = compute_phase_metrics(result["fault_log"], shift_index)
    result["phase_metrics"] = phase_metrics
    all_results["Learned (DT)"] = result
    _print_result("Learned (DT)", phase_metrics)

    # ---- Sensitivity: vary frame count ----------------------------- #
    sensitivity_results = {}
    for nf in [4, 6, 8, 10, 12]:
        row = {}
        for name_cls, cls in [
            ("FIFO", FIFOPageReplacement),
            ("LRU", LRUPageReplacement),
            ("Optimal", OptimalPageReplacement),
        ]:
            r = cls(nf).run(trace)
            pm = compute_phase_metrics(r["fault_log"], shift_index)
            row[name_cls] = pm

        # Retrain learned for this frame count
        dec = OptimalPageReplacement(nf).get_eviction_decisions(training_trace, nf)
        lp = LearnedPageReplacement(nf)
        lp.train(dec)
        r = lp.run(trace)
        pm = compute_phase_metrics(r["fault_log"], shift_index)
        row["Learned (DT)"] = pm

        sensitivity_results[nf] = row

    # ---- Save raw results ------------------------------------------ #
    # Remove fault_log from saved results (too large) but keep phase metrics
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {
            k: v for k, v in r.items() if k != "fault_log"
        }

    with open(os.path.join(RESULTS_DIR, "main_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "sensitivity_results.json"), "w") as f:
        json.dump(sensitivity_results, f, indent=2)

    # Save the trace itself for reference
    with open(os.path.join(RESULTS_DIR, "trace.json"), "w") as f:
        json.dump({"trace": trace, "shift_index": shift_index}, f)

    # Save fault logs separately for plotting
    fault_logs = {name: r["fault_log"] for name, r in all_results.items()}
    with open(os.path.join(RESULTS_DIR, "fault_logs.json"), "w") as f:
        json.dump(fault_logs, f)

    print(f"\nRaw results saved to {os.path.abspath(RESULTS_DIR)}/")


def _print_result(name, pm):
    """Pretty-print phase metrics for one algorithm."""
    p1 = pm["phase1"]
    p2 = pm["phase2"]
    ov = pm["overall"]
    print(f"--- {name} ---")
    print(f"  Overall   : faults={ov['faults']:5d}  hit_ratio={ov['hit_ratio']:.4f}")
    print(f"  Phase 1   : faults={p1['faults']:5d}  hit_ratio={p1['hit_ratio']:.4f}")
    print(f"  Phase 2   : faults={p2['faults']:5d}  hit_ratio={p2['hit_ratio']:.4f}")
    print()


if __name__ == "__main__":
    main()
