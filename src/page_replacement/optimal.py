"""
Optimal (Belady's) Page Replacement Algorithm.

Evicts the page that will not be used for the longest time in the
future.  This is provably optimal but requires advance knowledge of
the entire reference string, so it serves only as a theoretical
lower bound on page faults.
"""


class OptimalPageReplacement:
    """
    Simulates Belady's optimal page replacement.

    Before the run starts the full trace is scanned so that, at each
    fault, we know which loaded page has the farthest (or no) future use.
    """

    def __init__(self, num_frames):
        if num_frames <= 0:
            raise ValueError("Number of frames must be positive")
        self.num_frames = num_frames
        self.fault_log = []
        self.eviction_log = []

    def _build_next_use(self, page_trace):
        """
        Pre-compute, for every position i, the next position where each
        page will be referenced.  Returns a list of length len(page_trace);
        next_use[i] = {page: next_index, ...}.
        """
        n = len(page_trace)
        # last_seen[page] = index of the *next* future use seen so far
        # (built by scanning right to left)
        last_seen = {}
        next_use = [None] * n
        for i in range(n - 1, -1, -1):
            last_seen[page_trace[i]] = i
            next_use[i] = dict(last_seen)
        return next_use

    def run(self, page_trace):
        """
        Run optimal replacement over the full trace.
        """
        self.fault_log = []
        self.eviction_log = []
        frames = set()
        next_use = self._build_next_use(page_trace)

        for i, page in enumerate(page_trace):
            if page in frames:
                self.fault_log.append(0)
                continue

            # Page fault
            evicted = None
            if len(frames) >= self.num_frames:
                # Pick the page whose next use is farthest away
                # (or never used again — use infinity)
                farthest_page = None
                farthest_dist = -1
                for p in frames:
                    nxt = next_use[i].get(p, float("inf"))
                    if nxt == float("inf"):
                        farthest_page = p
                        break
                    if nxt > farthest_dist:
                        farthest_dist = nxt
                        farthest_page = p
                evicted = farthest_page
                frames.remove(evicted)

            frames.add(page)
            self.fault_log.append(1)
            self.eviction_log.append(evicted)

        total = len(page_trace)
        faults = sum(self.fault_log)
        hits = total - faults
        return {
            "algorithm": "Optimal",
            "total_accesses": total,
            "faults": faults,
            "hits": hits,
            "hit_ratio": hits / total if total > 0 else 0.0,
            "fault_log": list(self.fault_log),
        }

    def get_eviction_decisions(self, page_trace, num_frames=None):
        """
        Run optimal and return structured eviction decision records
        that the learned component can use as training labels.

        Each record contains:
          - position in trace
          - current frame contents
          - the page Optimal chose to evict
          - features for every page in the frame at that moment
        """
        if num_frames is None:
            num_frames = self.num_frames

        frames = set()
        next_use = self._build_next_use(page_trace)
        access_count = {}     # page -> total accesses so far
        last_access = {}      # page -> last access timestamp
        load_time = {}        # page -> when it was loaded into frame

        decisions = []

        for i, page in enumerate(page_trace):
            access_count[page] = access_count.get(page, 0) + 1
            last_access[page] = i

            if page in frames:
                continue

            # Page fault — record decision if we actually need to evict
            if len(frames) >= num_frames:
                farthest_page = None
                farthest_dist = -1
                for p in frames:
                    nxt = next_use[i].get(p, float("inf"))
                    if nxt == float("inf"):
                        farthest_page = p
                        break
                    if nxt > farthest_dist:
                        farthest_dist = nxt
                        farthest_page = p

                # Build a feature record for each page in the frame
                frame_features = []
                for p in frames:
                    recency = i - last_access.get(p, 0)
                    freq = access_count.get(p, 0)
                    age = i - load_time.get(p, 0)
                    frame_features.append({
                        "page": p,
                        "recency": recency,
                        "frequency": freq,
                        "age": age,
                        "evict": 1 if p == farthest_page else 0,
                    })

                decisions.append({
                    "time": i,
                    "requested_page": page,
                    "frame_features": frame_features,
                })

                frames.remove(farthest_page)
                load_time[page] = i

            else:
                load_time[page] = i

            frames.add(page)

        return decisions
