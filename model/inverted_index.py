from numpy import float32, int64
from numpy.typing import NDArray
from sklearn.neighbors import KDTree


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

    def _set_kd_tree(self, matrix: NDArray[float32]):
        cluster_members = matrix[self.cluster_member_indices]
        self.kd_tree = KDTree(cluster_members, metric="euclidean")
