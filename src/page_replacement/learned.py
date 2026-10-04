"""
Learned Page Replacement using a Decision Tree Classifier.

The idea: instead of following a fixed eviction rule, we train a
lightweight model on the decisions that Belady's optimal algorithm
would make.  At each page fault the model scores every resident page
and evicts the one it considers the best candidate.

Features per candidate page:
    - recency        : accesses elapsed since the page was last referenced
    - frequency      : cumulative access count for the page
    - age            : accesses elapsed since the page was loaded into frame
    - recency_rank   : rank of this page by recency among all resident pages
                       (0 = most recently used, num_frames-1 = least recently used)
    - freq_rank      : rank of this page by frequency (0 = most frequent)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict


class LearnedPageReplacement:
    """
    A page-replacement policy driven by a scikit-learn DecisionTreeClassifier.

    Workflow
    --------
    1. Call ``train(training_decisions)`` with labelled data from the
       Optimal algorithm (produced by
       ``OptimalPageReplacement.get_eviction_decisions``).
    2. Call ``run(page_trace)`` to simulate replacements on any trace.
    """

    def __init__(self, num_frames, max_depth=8, random_state=42):
        if num_frames <= 0:
            raise ValueError("Number of frames must be positive")
        self.num_frames = num_frames
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",   # handle the imbalanced labels
        )
        self.is_trained = False
        self.fault_log = []
        self.eviction_log = []

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train(self, decisions):
        """
        Train the classifier from Optimal's labelled eviction decisions.

        Parameters
        ----------
        decisions : list[dict]
            Each dict has ``frame_features`` — a list of per-page dicts
            with keys ``recency``, ``frequency``, ``age``, ``evict``.
        """
        X, y = [], []
        for dec in decisions:
            entries = dec["frame_features"]

            # Compute ranks within this eviction event
            recencies = [e["recency"] for e in entries]
            freqs = [e["frequency"] for e in entries]
            recency_ranks = _rank_descending(recencies)   # higher recency -> higher rank
            freq_ranks = _rank_ascending(freqs)            # lower freq -> higher rank

            for idx, entry in enumerate(entries):
                X.append([
                    entry["recency"],
                    entry["frequency"],
                    entry["age"],
                    recency_ranks[idx],
                    freq_ranks[idx],
                ])
                y.append(entry["evict"])

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int32)

        self.model.fit(X, y)
        self.is_trained = True

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #
    def run(self, page_trace):
        """
        Simulate the learned replacement policy on the given trace.

        At each page fault the model predicts an eviction probability
        for every resident page and evicts the one with the highest
        predicted eviction score.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained — call train() first")

        self.fault_log = []
        self.eviction_log = []

        frames = OrderedDict()   # page -> load_time
        access_count = {}
        last_access = {}

        for i, page in enumerate(page_trace):
            access_count[page] = access_count.get(page, 0) + 1
            last_access[page] = i

            if page in frames:
                self.fault_log.append(0)
                continue

            # Page fault
            evicted = None
            if len(frames) >= self.num_frames:
                evicted = self._pick_eviction(frames, access_count, last_access, i)
                del frames[evicted]

            frames[page] = i   # load time
            self.fault_log.append(1)
            self.eviction_log.append(evicted)

        total = len(page_trace)
        faults = sum(self.fault_log)
        hits = total - faults
        return {
            "algorithm": "Learned (DT)",
            "total_accesses": total,
            "faults": faults,
            "hits": hits,
            "hit_ratio": hits / total if total > 0 else 0.0,
            "fault_log": list(self.fault_log),
        }

    def _pick_eviction(self, frames, access_count, last_access, current_time):
        """
        Query the decision-tree model for each resident page and return
        the page with the highest predicted eviction probability.
        """
        pages = list(frames.keys())
        raw_features = []
        for p in pages:
            recency = current_time - last_access.get(p, 0)
            freq = access_count.get(p, 0)
            age = current_time - frames[p]
            raw_features.append([recency, freq, age])

        # Compute ranks
        recencies = [f[0] for f in raw_features]
        freqs = [f[1] for f in raw_features]
        recency_ranks = _rank_descending(recencies)
        freq_ranks = _rank_ascending(freqs)

        features = []
        for idx, (rec, frq, ag) in enumerate(raw_features):
            features.append([rec, frq, ag, recency_ranks[idx], freq_ranks[idx]])

        X = np.array(features, dtype=np.float64)

        # Use predict_proba to get a confidence-ranked eviction
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Column index for class "1" (evict)
            if probs.shape[1] == 2:
                evict_scores = probs[:, 1]
            else:
                evict_scores = probs[:, 0]
        else:
            evict_scores = self.model.predict(X).astype(float)

        best_idx = int(np.argmax(evict_scores))
        return pages[best_idx]


# -------------------------------------------------------------------- #
#  Ranking helpers
# -------------------------------------------------------------------- #
def _rank_descending(values):
    """Rank values so that the largest gets rank 0."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _rank_ascending(values):
    """Rank values so that the smallest gets rank 0."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks
