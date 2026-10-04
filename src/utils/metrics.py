"""
Metric helpers for page-replacement experiments.
"""


def compute_metrics(fault_log):
    """
    Given a fault_log (list of 0/1 per access), return summary stats.
    """
    total = len(fault_log)
    faults = sum(fault_log)
    hits = total - faults
    return {
        "total_accesses": total,
        "faults": faults,
        "hits": hits,
        "hit_ratio": hits / total if total > 0 else 0.0,
        "fault_rate": faults / total if total > 0 else 0.0,
    }


def compute_phase_metrics(fault_log, shift_index):
    """
    Split the fault log at shift_index and compute metrics
    for Phase 1 (before shift) and Phase 2 (after shift) separately.
    """
    phase1_log = fault_log[:shift_index]
    phase2_log = fault_log[shift_index:]
    return {
        "phase1": compute_metrics(phase1_log),
        "phase2": compute_metrics(phase2_log),
        "overall": compute_metrics(fault_log),
    }
