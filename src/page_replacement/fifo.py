"""
FIFO (First-In, First-Out) Page Replacement Algorithm.

Evicts the page that has been in memory the longest,
regardless of how recently or frequently it was accessed.
"""

from collections import deque


class FIFOPageReplacement:
    """
    Simulates the FIFO page replacement policy.

    Maintains a queue of pages currently loaded in frames.
    On a page fault, the oldest page (front of the queue) is evicted.
    """

    def __init__(self, num_frames):
        if num_frames <= 0:
            raise ValueError("Number of frames must be positive")
        self.num_frames = num_frames
        self.frames = set()
        self.queue = deque()
        self.fault_log = []     # 1 = fault, 0 = hit for each access
        self.eviction_log = []  # which page was evicted at each fault

    def access(self, page):
        """
        Process a single page access.

        Returns True if it was a hit, False if a page fault occurred.
        """
        if page in self.frames:
            self.fault_log.append(0)
            return True

        # Page fault
        evicted = None
        if len(self.frames) >= self.num_frames:
            evicted = self.queue.popleft()
            self.frames.remove(evicted)

        self.frames.add(page)
        self.queue.append(page)
        self.fault_log.append(1)
        self.eviction_log.append(evicted)
        return False

    def run(self, page_trace):
        """
        Run the algorithm over an entire page reference string.

        Returns a dict with total faults, hits, hit ratio, and per-access log.
        """
        self.frames = set()
        self.queue = deque()
        self.fault_log = []
        self.eviction_log = []

        for page in page_trace:
            self.access(page)

        total = len(page_trace)
        faults = sum(self.fault_log)
        hits = total - faults
        return {
            "algorithm": "FIFO",
            "total_accesses": total,
            "faults": faults,
            "hits": hits,
            "hit_ratio": hits / total if total > 0 else 0.0,
            "fault_log": list(self.fault_log),
        }
