import itertools
from heapq import heappop, heappush, heapreplace
from typing import Iterator, List, Tuple

from numpy.typing import NDArray


class MaxHeap:
    """
    Basic max heap implementation using heapq

    Python does not support max heap operations
    (up to version 3.14, which I did not have when writing this)

    This implementation uses a min heap but pushes and pops
    negative elements, resulting in max heap
    """

    heap: List
    counter: Iterator

    def __init__(self):
        self.heap = []
        self.counter = itertools.count()

    def __len__(self):
        return len(self.heap)

    def push(self, priority: float, value: NDArray):
        count = next(self.counter)
        entry = (-priority, count, value)
        heappush(self.heap, entry)

    def pop(self) -> Tuple[int, NDArray]:
        priority, _, value = heappop(self.heap)
        return -priority, value

    def peek(self) -> Tuple[int, NDArray]:
        priority, _, value = self.heap[0]
        return -priority, value

    def replace(self, priority: float, value: NDArray):
        count = next(self.counter)
        entry = (-priority, count, value)
        priority, _, value = heapreplace(self.heap, entry)
        return priority, value

    def get_all_values(self) -> List:
        return [values for _, _, values in self.heap]
