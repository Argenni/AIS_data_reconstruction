# ----------- Library of functions used in clustering phase of AIS message reconstruction ----------
from sklearn import cluster, metrics
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from utils.miscellaneous import count_number

class Clustering:
    """
    Class that introduces clustering using either k-means or DBSCAN
    """
    _epsilon = 3.16
    _minpts = 1

    def __init__(self):
        """
        Class initialization
        """
        pass

    def run_kmeans(self, X, K):
        """
        Runs k-means algorithm on a given dataset with Euclidean distance metric
        Arguments: 
        - X - numpy array with dataset to cluster, shape = (num_message, num_features (115))
        - K - integer scalar, number of clusters
        Returns:
        - idx - list of indices of clusters assigned to each message, len = num_messages
        - centroids - numpy array with centers of each cluster, shape = (K, num_features (115))
        """
        kmeans_model = cluster.KMeans(n_clusters=K, n_init=200, max_iter=100, tol=0.001, random_state=0).fit(X)
        idx = kmeans_model.labels_
        centroids = kmeans_model.cluster_centers_
        return idx, centroids

    def run_DBSCAN(self, X, distance, optimize=None):
        """
        Runs DBSCAN algorithm on a given dataset with given distance metric
        Arguments: 
        - X - numpy array with dataset to cluster, shape = (num_message, num_features (115))
        - distance - single string, distance metric, preferably 'euclidean'
        Returns:
        - idx - list of indices of clusters assigned to each message, len = num_messages
        - K -integer scalar, number og clusters created by DBSCAN
        """
        # Optimize hyperparametres if allowed
        if optimize == 'epsilon': self.optimize(X, distance, parameter='epsilon')
        elif optimize == 'minpts': self.optimize(X, distance, parameter='minpts')
        # Cluster using DBSCAN
        DBSCAN_model = cluster.DBSCAN(eps = self._epsilon, min_samples = self._minpts, metric = distance).fit(X)
        idx = DBSCAN_model.labels_
        K, _ = count_number(idx)
        return idx, K

    def optimize(self, X, distance, parameter):
        """
        Search for optimal epsilon and 
        """
        params = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
        silhuettes = []
        print(" Search for optimal " + parameter + "...")
        for param in params:
            if parameter=='epsilon':
                DBSCAN_model = cluster.DBSCAN(
                    eps = param, 
                    min_samples = self._minpts, 
                    metric = distance).fit(X)
            elif parameter=='minpts':
                DBSCAN_model = cluster.DBSCAN(
                    eps = self._epsilon, 
                    min_samples = param, 
                    metric = distance).fit(X)
            idx = DBSCAN_model.labels_
            if count_number(idx)[0] == 1: silhuettes.append(0)
            else: silhuettes.append(metrics.silhouette_score(X,idx))
        # Plot
        fig, ax = plt.subplots()
        ax.plot(params, silhuettes, color='k')
        ax.set_title("Average silhouettes vs " + parameter)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Average silhouette")
        fig.show()
        # Save the optimal value
        if parameter=='epsilon': self._epsilon = int(input("Choose the optimal epsilon: "))
        elif parameter=='minpts': self._epsilon = int(input("Choose the optimal minpts: "))


def calculate_CC(idx,MMSI,MMSI_vec):
    """
    Calculate correctness coefficient - indicator of to which extent:
    1. each cluster consists of messages from one vessel 
    2. messages from one vessel are not divided between several clusters
    Arguments:
    - idx - list of indices of clusters assigned to each message, len = num_messages
    - MMSI - list of MMSI identifier from each AIS message, len = num_messages
    - MMSI_vec - list of unique MMSIs in MMSI list
    Returns: CC - float scalar, computed correctness coefficient
    """
    if idx.shape[0]==0 or len(MMSI)==0 or len(MMSI_vec)==0:
        return 0
    # Compute clusters' homogeneity coefficient
    accuracy_MMSI_vec1 = []
    accuracy_MMSI_vec2 = []
    for i in MMSI_vec:  # For each MMSI value
        same = idx[np.array(MMSI)==i]  # check which clusters consists of data from that MMSI
        _, clusters = count_number(same)  # count how many such clusters there is
        volume = []
        for j in clusters:  # count the volume of each of those clusters
            volume.append(np.sum(np.where(same==j,1,0)))
        accuracy_MMSI_vec1.append(np.max(volume)/np.sum(volume))  # find the modal cluster and count the fraction of messages from that MMSI in that cluster
        accuracy_MMSI_vec2.append(np.sum(volume))
    accuracy_MMSI = np.sum(
        np.multiply(accuracy_MMSI_vec1,accuracy_MMSI_vec2))/np.sum(accuracy_MMSI_vec2
        )  # Calculate the weighted average
    # Compute vessels' homogeneity coefficient
    accuracy_clust_vec1 = []
    accuracy_clust_vec2 = []
    for i in range(np.min(idx),np.max(idx)):  # For each cluster
        same = np.array(MMSI)[idx==i]  # check which MMSI that cluster consists of 
        _, MMSIs = count_number(same)  # count how many such MMSIs there is
        volume = []
        for j in MMSIs:  # count the volume of each of those MMSIs
            volume.append(np.sum(np.where(same==j,1,0)))
        accuracy_clust_vec1.append(np.max(volume)/np.sum(volume))  # find the modal MMSI and count the fraction of messages from that MMSI in that cluster
        accuracy_clust_vec2.append(np.sum(volume))
    accuracy_clust = np.sum(
        np.multiply(accuracy_clust_vec1,accuracy_clust_vec2))/np.sum(accuracy_clust_vec2
        )  # Calculate the weighted average   
    # Compute correctness coefficient as a F1 score
    CC = 2*accuracy_clust*accuracy_MMSI/(accuracy_clust+accuracy_MMSI)
    return CC