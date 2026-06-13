SIFT_BASE = "sift/sift_base.fvecs"
SIFT_GROUNDTRUTH = "sift/sift_groundtruth.ivecs"
SIFT_LEARN = "sift/sift_learn.fvecs"
SIFT_QUERY = "sift/sift_query.fvecs"

CLUSTER_NUMBER = 1000  # this is set to sqrt(len(dataset))
N_NEAREST_CENTROIDS = 50  # number of the nearest centroids to the query
N_NEAREST_NEIGHBORS = 40  # number of nearest neighbors to search for the query
CLUSTERER_PATH = "./clusterer_model/kmeans"  # path to save the clusterer model
