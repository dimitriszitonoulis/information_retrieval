import matplotlib.pyplot as plt
from numpy import typing as nptyping
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils.dataset_loader import read_fvecs


def get_silhouette_score(matrix, max_k=10):
    best_k = 2
    best_score = -1

    scores = []

    for k in range(2, max_k + 1):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
        )
        labels = kmeans.fit_predict(matrix)
        print(f"Calculating silhouette for {k} clusters")
        score = silhouette_score(matrix, labels, sample_size=2000, random_state=42)
        # score = silhouette_score(matrix, labels, random_state=42)

        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

    # return best_k, scores
    return scores, best_score, best_k


def elbow_score(matrix: nptyping.NDArray, max_k: int):
    inertias = []

    for k in range(2, max_k + 1):
        # kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=5)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
        kmeans.fit(matrix)

        inertias.append(kmeans.inertia_)

    return inertias


SIFT_LEARN = "sift/sift_learn.fvecs"
max_k = 100
n_components = 32
matrix = read_fvecs(SIFT_LEARN)
matrix_pca = PCA(n_components=n_components).fit_transform(matrix)


scores, best_score, best_k = get_silhouette_score(matrix_pca, max_k)

print(f"{best_score=}, {best_k=}")

plt.plot(range(2, max_k + 1), scores, marker="o")
plt.xlabel("k")
plt.ylabel("Scores")
plt.title("Silhouette score")
plt.show()
