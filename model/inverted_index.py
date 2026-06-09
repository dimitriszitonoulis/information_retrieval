from numpy import float32, int64
from numpy.typing import NDArray


class InvertedIndex:
    centroid: NDArray[float32]
    cluster_member_indices: NDArray[int64]

    def __init__(
        self,
        centroid: NDArray[float32],
        cluster_member_indices: NDArray[int64],
    ):
        self.centroid = centroid
        self.cluster_member_indices = cluster_member_indices
