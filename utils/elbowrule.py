import kneed as kn
import matplotlib.pyplot as plt
import sklearn.cluster as skc

from config import SIFT_LEARN
from utils.dataset_loader import read_fvecs


def determineNumberOfClusters(clusterRange, initializationStrategy, numberOfInitializations, data):

    # (Within-Cluster Sum of Square)
    wcss = []

    for k in clusterRange:
        kmeans = skc.KMeans(
            init=initializationStrategy,
            n_clusters=k,
            n_init=numberOfInitializations,
            random_state=0,
        )
        # Υπολογισμός συστάδων με βάση τον K-Means
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # Auto einai pou ypologizei SSE
    # Αυτόματη επιλογή του αριθμού των συστάδων βασισμένη στο διάγραμμα μιας μετρικής συναρτήσει του πλήθους συστάδων
    # Πριν απο την απόφαση για το βέλτιστο πλήθος συστάδων θα πρέπει να διερευνάται και η φύση του προβλήματος.
    kneeData = kn.KneeLocator(
        clusterRange,
        wcss,
        curve="convex",
        direction="decreasing",
        S=1.0,
        online=False,
        interp_method="polynomial",
    )
    kneeData.plot_knee()

    return wcss, kneeData.elbow


cluster_range = range(2, 100)
initialization_strategy = "k-means++"
n_init = 3
matrix = read_fvecs(SIFT_LEARN)


determineNumberOfClusters(cluster_range, initialization_strategy, n_init, matrix)


scoreData, elbowData = determineNumberOfClusters(
    cluster_range, initialization_strategy, n_init, matrix
)
plt.figure(plt.gcf().number).suptitle("Original Data")
plt.show()
