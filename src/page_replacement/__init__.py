from .fifo import FIFOPageReplacement
from .lru import LRUPageReplacement
from .optimal import OptimalPageReplacement
from .learned import LearnedPageReplacement

__all__ = [
    "FIFOPageReplacement",
    "LRUPageReplacement",
    "OptimalPageReplacement",
    "LearnedPageReplacement",
]
