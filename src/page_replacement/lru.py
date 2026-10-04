"""
LRU (Least Recently Used) Page Replacement Algorithm.

Evicts the page that has not been accessed for the longest period.
Uses an ordered dictionary to track recency efficiently.
"""

from collections import OrderedDict


class LRUPageReplacement:
    """
    Simulates the LRU page replacement policy.

    Uses an OrderedDict so that moving a page to the end on every
    access keeps the least-recently-used page at the front.
    """

    def __init__(self, num_frames):
        if num_frames <= 0:
            raise ValueError("Number of frames must be positive")
        self.num_frames = num_frames
        self.cache = OrderedDict()
        self.fault_log = []
        self.eviction_log = []

    def access(self, page):
        """
        Process a single page access.

        Returns True on hit, False on fault.
        """
        if page in self.cache:
            # Move to end to mark as most recently used
            self.cache.move_to_end(page)
            self.fault_log.append(0)
            return True

        # Page fault
        evicted = None
        if len(self.cache) >= self.num_frames:
            # Pop the least recently used (front of OrderedDict)
            evicted, _ = self.cache.popitem(last=False)

        self.cache[page] = True
        self.fault_log.append(1)
        self.eviction_log.append(evicted)
        return False

    def run(self, page_trace):
        """
        Run LRU over the full page reference string.
        """
        self.cache = OrderedDict()
        self.fault_log = []
        self.eviction_log = []

        for page in page_trace:
            self.access(page)

        total = len(page_trace)
        faults = sum(self.fault_log)
        hits = total - faults
        return {
            "algorithm": "LRU",
            "total_accesses": total,
            "faults": faults,
            "hits": hits,
            "hit_ratio": hits / total if total > 0 else 0.0,
            "fault_log": list(self.fault_log),
        }
