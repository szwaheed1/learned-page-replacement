#!/usr/bin/env python3
"""
Unit tests for page-replacement algorithms and workload generator.

Run with:
    python -m pytest tests/test_algorithms.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.page_replacement import (
    FIFOPageReplacement,
    LRUPageReplacement,
    OptimalPageReplacement,
    LearnedPageReplacement,
)
from src.workload import WorkloadGenerator
from src.utils import compute_metrics, compute_phase_metrics


# -------------------------------------------------------------------- #
#  Helpers
# -------------------------------------------------------------------- #
# Classic textbook example:
#   Reference string: 7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1
#   3 frames
TEXTBOOK_TRACE = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


class TestFIFO:
    def test_textbook_example(self):
        """FIFO with 3 frames on the standard textbook trace should produce 15 faults."""
        fifo = FIFOPageReplacement(3)
        result = fifo.run(TEXTBOOK_TRACE)
        assert result["faults"] == 15

    def test_no_faults_if_enough_frames(self):
        trace = [1, 2, 3, 1, 2, 3]
        fifo = FIFOPageReplacement(3)
        result = fifo.run(trace)
        # First 3 are compulsory faults, rest are hits
        assert result["faults"] == 3
        assert result["hits"] == 3

    def test_single_frame(self):
        trace = [1, 2, 1, 2]
        fifo = FIFOPageReplacement(1)
        result = fifo.run(trace)
        assert result["faults"] == 4  # every access is a fault

    def test_repeated_page(self):
        trace = [5, 5, 5, 5, 5]
        fifo = FIFOPageReplacement(2)
        result = fifo.run(trace)
        assert result["faults"] == 1
        assert result["hits"] == 4


class TestLRU:
    def test_textbook_example(self):
        """LRU with 3 frames on the textbook trace should produce 12 faults."""
        lru = LRUPageReplacement(3)
        result = lru.run(TEXTBOOK_TRACE)
        assert result["faults"] == 12

    def test_no_faults_if_enough_frames(self):
        trace = [1, 2, 3, 1, 2, 3]
        lru = LRUPageReplacement(3)
        result = lru.run(trace)
        assert result["faults"] == 3

    def test_single_frame(self):
        trace = [1, 2, 1, 2]
        lru = LRUPageReplacement(1)
        result = lru.run(trace)
        assert result["faults"] == 4

    def test_lru_evicts_correct_page(self):
        """With 2 frames and trace [1,2,3,1], after loading 1,2 then
        accessing 3 should evict 1 (least recently used), then
        accessing 1 should evict 2."""
        lru = LRUPageReplacement(2)
        result = lru.run([1, 2, 3, 1])
        assert result["faults"] == 4  # 1(F) 2(F) 3(F,evict1) 1(F,evict2)


class TestOptimal:
    def test_textbook_example(self):
        """Optimal with 3 frames on the textbook trace should produce 9 faults."""
        opt = OptimalPageReplacement(3)
        result = opt.run(TEXTBOOK_TRACE)
        assert result["faults"] == 9

    def test_optimal_is_best(self):
        """Optimal should never have more faults than FIFO or LRU."""
        trace = TEXTBOOK_TRACE
        for nf in [2, 3, 4]:
            opt_r = OptimalPageReplacement(nf).run(trace)
            fifo_r = FIFOPageReplacement(nf).run(trace)
            lru_r = LRUPageReplacement(nf).run(trace)
            assert opt_r["faults"] <= fifo_r["faults"]
            assert opt_r["faults"] <= lru_r["faults"]

    def test_eviction_decisions_not_empty(self):
        opt = OptimalPageReplacement(3)
        decisions = opt.get_eviction_decisions(TEXTBOOK_TRACE)
        assert len(decisions) > 0
        # Each decision should have frame_features with exactly one evict=1
        for d in decisions:
            evict_count = sum(1 for f in d["frame_features"] if f["evict"] == 1)
            assert evict_count == 1


class TestLearned:
    def test_train_and_run(self):
        """Learned model should train without errors and produce valid output."""
        opt = OptimalPageReplacement(3)
        decisions = opt.get_eviction_decisions(TEXTBOOK_TRACE)
        learned = LearnedPageReplacement(3)
        learned.train(decisions)
        result = learned.run(TEXTBOOK_TRACE)
        assert result["faults"] > 0
        assert result["total_accesses"] == len(TEXTBOOK_TRACE)
        assert 0.0 <= result["hit_ratio"] <= 1.0

    def test_untrained_raises(self):
        learned = LearnedPageReplacement(3)
        with pytest.raises(RuntimeError):
            learned.run([1, 2, 3])


class TestWorkloadGenerator:
    def test_trace_length(self):
        gen = WorkloadGenerator(trace_length=1000, seed=123)
        trace, shift = gen.generate()
        assert len(trace) == 1000
        assert shift == 500

    def test_reproducibility(self):
        gen1 = WorkloadGenerator(seed=99)
        gen2 = WorkloadGenerator(seed=99)
        t1, _ = gen1.generate()
        t2, _ = gen2.generate()
        assert t1 == t2

    def test_pages_in_range(self):
        gen = WorkloadGenerator(num_pages=30, trace_length=2000, seed=7)
        trace, _ = gen.generate()
        assert all(0 <= p < 30 for p in trace)

    def test_training_trace(self):
        gen = WorkloadGenerator(trace_length=1000, seed=42)
        tt = gen.generate_training_trace(length=500)
        assert len(tt) == 500


class TestMetrics:
    def test_compute_metrics(self):
        log = [1, 0, 1, 0, 0]  # 2 faults, 3 hits
        m = compute_metrics(log)
        assert m["faults"] == 2
        assert m["hits"] == 3
        assert abs(m["hit_ratio"] - 0.6) < 1e-9

    def test_phase_metrics(self):
        log = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
        pm = compute_phase_metrics(log, 5)
        assert pm["phase1"]["faults"] == 2
        assert pm["phase2"]["faults"] == 4
