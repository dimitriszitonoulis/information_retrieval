from numpy import float32, int64
from numpy.typing import NDArray
from sklearn.neighbors import KDTree


class InvertedIndex:
    centroid: NDArray[float32]
    cluster_member_indices: NDArray[int64]
    kd_tree: KDTree

    def __init__(
        self,
        centroid: NDArray[float32],
        cluster_member_indices: NDArray[int64],
        matrix: NDArray[float32],
    ):
        self.centroid = centroid
        self.cluster_member_indices = cluster_member_indices
        self._set_kd_tree(matrix)

    def _set_kd_tree(self, matrix: NDArray[float32]):
        cluster_members = matrix[self.cluster_member_indices]
        self.kd_tree = KDTree(cluster_members, metric="euclidean")

    def get_nearest_members_indices(
        self,
        query_vector: NDArray[float32],
        n_nearest_neighbors: int = 1,
        return_distance: bool = True,
    ):
        """
        Finds the indices of the nearest neigbors to the the
        query. The indices of each member refer to the indices
        of `self.matrix`.

        Args:
            query_vector (NDArray[float32]): The query vector
                for which to find the indices of the nearest members
            n_nearest_neighbors (int): . Defaults to 1
            return_distance (bool): . Defaults to True

        Returns:
            If `return_distane` is True:

            else:
            -NDArray[int64]: Array containing the indices of the nearest
                members

        """

        k = min(n_nearest_neighbors, len(self.cluster_member_indices))

        if not return_distance:
            nearest_member_indices = self.kd_tree.query(
                X=query_vector, k=k, return_distance=return_distance
            )
            return self.cluster_member_indices[nearest_member_indices]

        distances, nearest_member_indices = self.kd_tree.query(
            X=query_vector, k=k, return_distance=return_distance
        )
        return distances, self.cluster_member_indices[nearest_member_indices]

    def get_distances_calculated(self) -> int:
        n_calls: int = self.kd_tree.get_n_calls()
        self.kd_tree.reset_n_calls()
        return n_calls
