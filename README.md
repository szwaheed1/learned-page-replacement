# Learned Page Replacement: Classical Algorithms Meet Adaptive Prediction

**CSE-307: Operating Systems — Term Paper (Track 1)**

## Overview

This project implements and compares three classical page-replacement algorithms (FIFO, LRU, and Belady's Optimal) against a **learned replacement policy** that uses a Decision Tree classifier trained on Optimal's eviction decisions. The key experiment introduces a deliberate **workload shift** — from a locality-heavy access pattern to a random/bursty one — to study how each policy's performance degrades when the underlying pattern changes.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── page_replacement/
│   │   ├── fifo.py          # FIFO (First-In, First-Out)
│   │   ├── lru.py           # LRU (Least Recently Used)
│   │   ├── optimal.py       # Belady's Optimal algorithm
│   │   └── learned.py       # Decision Tree–based learned policy
│   ├── workload/
│   │   └── generator.py     # Synthetic two-phase workload generator
│   └── utils/
│       └── metrics.py       # Hit/fault metric computation
├── experiments/
│   ├── run_experiments.py    # Run all algorithms, collect raw data
│   └── analyze_results.py   # Generate figures and tables
├── results/
│   ├── raw/                  # JSON outputs from experiments
│   └── figures/              # Generated plots (PNG)
├── paper/
│   └── term_paper.tex        # LaTeX source for the report
└── tests/
    └── test_algorithms.py    # Unit tests (pytest)
```

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Run the experiments

```bash
python experiments/run_experiments.py
```

This runs all four replacement policies on a 10,000-access synthetic trace (with the shift at access 5,000) and saves raw results to `results/raw/`.

### 2. Generate figures

```bash
python experiments/analyze_results.py
```

Produces five plots in `results/figures/`:
- `hit_ratio_comparison.png` — overall hit ratio bar chart
- `fault_count_comparison.png` — total page faults bar chart
- `phase_comparison.png` — phase 1 vs. phase 2 hit ratio (grouped bars)
- `rolling_fault_rate.png` — rolling fault rate over time (shows the shift)
- `sensitivity_frame_count.png` — hit ratio vs. number of frames

### 3. Run tests

```bash
python -m pytest tests/ -v
```

Includes textbook-verified test cases for FIFO (15 faults), LRU (12 faults), and Optimal (9 faults) on the standard 20-element reference string with 3 frames.

## Summary of Results

| Algorithm    | Phase 1 (Locality) | Phase 2 (Random) | Overall HR |
|-------------|-------------------|------------------|------------|
| FIFO        | Good              | Sharp drop       | 64.46%     |
| LRU         | Best among practical | Sharp drop  | 65.16%     |
| Optimal     | Theoretical best   | Best overall     | 77.84%     |
| Learned (DT)| Close to LRU      | Sharp drop       | 65.08%     |

Key findings:
- All practical policies see a substantial hit-ratio drop post-shift, converging to similar performance levels under random access.
- **LRU** achieves the highest overall hit ratio among the non-optimal algorithms.
- The **Learned policy** successfully approximates eviction signals and performs comparably to LRU (and occasionally beats it across different memory sizes), but does not outperform it under the default 8-frame workload.

## AI Disclosure

Claude Code was used for implementation assistance, debugging, code organization, documentation, and drafting support. The experimental design, workload parameters, analysis, interpretation, and final submission were verified to ensure they are truthful, reproducible, technically correct, and understandable.
