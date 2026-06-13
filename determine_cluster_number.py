import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from config import SIFT_LEARN

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


max_k = 1000
matrix = read_fvecs(SIFT_LEARN)

scores, best_score, best_k = get_silhouette_score(matrix, max_k)

print(f"{best_score=}, {best_k=}")

plt.plot(range(2, max_k + 1), scores, marker="o")
plt.xlabel("k")
plt.ylabel("Scores")
plt.title("Silhouette score")
plt.show()
