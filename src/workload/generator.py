"""
Synthetic page-access workload generator.

Produces a page reference string in two distinct phases:

    Phase 1 (locality-heavy / sequential)
        A small working set is accessed repeatedly with high temporal
        locality, mixed with occasional sequential sweeps.

    Phase 2 (random / bursty)
        References are drawn uniformly from a much larger page space,
        with intermittent bursts on random hot pages.

The deliberate shift between phases is the "stress test" required
by the assignment: it lets us observe how each replacement policy
reacts when the workload character changes abruptly.
"""

import numpy as np


class WorkloadGenerator:
    """
    Parameters
    ----------
    num_pages : int
        Total number of distinct pages in the virtual address space.
    working_set_size : int
        Number of pages that form the "hot" working set in Phase 1.
    trace_length : int
        Total number of page accesses to generate (split 50/50).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_pages=50,
        working_set_size=10,
        trace_length=10000,
        seed=42,
    ):
        self.num_pages = num_pages
        self.working_set_size = working_set_size
        self.trace_length = trace_length
        self.rng = np.random.default_rng(seed)

    def generate(self):
        """
        Return (trace, shift_index) where shift_index marks the point
        where Phase 1 ends and Phase 2 begins.
        """
        half = self.trace_length // 2
        phase1 = self._locality_phase(half)
        phase2 = self._random_bursty_phase(half)
        trace = np.concatenate([phase1, phase2]).astype(int).tolist()
        return trace, half

    # ------------------------------------------------------------------ #
    #  Phase generators
    # ------------------------------------------------------------------ #
    def _locality_phase(self, length):
        """
        Heavy temporal locality within a tight working set.

        ~95 % of accesses are drawn from the working-set pages with a
        skewed distribution (lower-numbered pages are hotter).  The
        remaining ~5 % are short sequential sweeps that stay within or
        just outside the working set, simulating a process that mostly
        re-visits cached data with occasional short scans.
        """
        ws = list(range(self.working_set_size))
        trace = []
        i = 0
        while i < length:
            if self.rng.random() < 0.95:
                # Pick from the working set with a Zipf-like skew
                weights = np.array(
                    [1.0 / (j + 1) for j in range(self.working_set_size)],
                    dtype=float,
                )
                weights /= weights.sum()
                page = self.rng.choice(ws, p=weights)
                trace.append(page)
                i += 1
            else:
                # Short sequential sweep within the working set range
                sweep_len = int(self.rng.integers(2, 5))
                start = int(self.rng.integers(0, self.working_set_size))
                for s in range(sweep_len):
                    if i >= length:
                        break
                    trace.append((start + s) % self.working_set_size)
                    i += 1
        return np.array(trace[:length])

    def _random_bursty_phase(self, length):
        """
        Uniform random accesses across the full page space,
        with occasional short bursts on a random page.

        ~85 % of accesses are uniformly random across all pages; ~15 %
        are short bursts (3–6 repeats) on a randomly chosen page.
        """
        trace = []
        i = 0
        while i < length:
            if self.rng.random() < 0.85:
                # Uniform random over full page space
                page = int(self.rng.integers(0, self.num_pages))
                trace.append(page)
                i += 1
            else:
                # Short burst on a random page
                hot_page = int(self.rng.integers(0, self.num_pages))
                burst_len = int(self.rng.integers(3, 7))
                for _ in range(burst_len):
                    if i >= length:
                        break
                    trace.append(hot_page)
                    i += 1
        return np.array(trace[:length])

    def generate_training_trace(self, length=5000, seed_offset=100):
        """
        Generate a separate trace for training the learned model,
        with a mix of both locality and random patterns.
        """
        rng_backup = self.rng
        self.rng = np.random.default_rng(self.rng.integers(0, 2**31) + seed_offset)

        half = length // 2
        phase1 = self._locality_phase(half)
        phase2 = self._random_bursty_phase(half)
        trace = np.concatenate([phase1, phase2]).astype(int).tolist()

        self.rng = rng_backup
        return trace
