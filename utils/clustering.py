"""
Functions and classes used in clustering stage of AIS message reconstruction
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from utils.miscellaneous import count_number

class Clustering:
    """
    Class that introduces AIS data clustering using either k-means or DBSCAN.
    """
    _epsilon = 3.16
    _minpts = 1
    _verbose = []

    def __init__(self, verbose=False):
        """
        Class initialization (class object creation). \n
        Argument: verbose (optional) - Boolean, whether to print running logs or not, default=False
        """
        self._verbose = verbose

    def run_kmeans(self, X, K):
        """
        Runs k-means algorithm on a given dataset with Euclidean distance metric. \n
        Arguments: 
        - X - numpy array with dataset to cluster, shape=(num_message, num_features (115)),
        - K - scalar, int, desired number of clusters. \n
        Returns:
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - centroids - numpy array with centers of each cluster, shape=(K, num_features (115)).
        """
        if self._verbose: print("Running k-means clustering...")
        kmeans_model = KMeans(n_clusters=K, n_init=10, max_iter=100, tol=0.001, random_state=0).fit(X)
        idx = kmeans_model.labels_
        centroids = kmeans_model.cluster_centers_
        if self._verbose: print(" Complete.")
        return idx, centroids

    def run_DBSCAN(self, X, distance='euclidean', optimize=None):
        """
        Runs DBSCAN algorithm on a given dataset with given distance metric. \n
        Arguments: 
        - X - numpy array with dataset to cluster, shape=(num_message, num_features (115)),
        - distance - (optional) string, name of distance metric, default='euclidean',
        - optimize - (optional) string, name of DBSCAN hyperparameter to optimize, 
            'epsilon' or 'minpts', default=None (no optimization). \n
        Returns:
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - K - scalar, int, number of clusters created by DBSCAN.
        """
        # Optimize hyperparametres if allowed
        if optimize == 'epsilon': self._optimize_DBSCAN(X, distance, hyperparameter='epsilon')
        elif optimize == 'minpts': self._optimize_DBSCAN(X, distance, hyperparameter='minpts')
        # Cluster using DBSCAN
        if self._verbose: print("Running DBSCAN clustering...")
        DBSCAN_model = DBSCAN(eps = self._epsilon, min_samples = self._minpts, metric = distance).fit(X)
        idx = DBSCAN_model.labels_
        K, _ = count_number(idx)
        if self._verbose:print(" Complete.")
        return idx, K

    def _optimize_DBSCAN(self, X, distance, hyperparameter):
        """
        Searches for optimal epsilon or minpts value for AIS data clustering using DBSCAN
        and stores it in self._epsilon or self._minpts. Arguments:
        - X - numpy array with training dataset, shape=(num_message, num_features (115)),
        - distance - string, name of distance metric, eg. 'euclidean',
        - hyperparameter - string, name of DBSCAN hyperparameter to optimize, 'epsilon' or 'minpts'.
        """
        params = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
        silhuettes = []
        print(" Search for optimal " + hyperparameter + "...")
        for param in params:
            if hyperparameter=='epsilon':
                DBSCAN_model = DBSCAN(
                    eps = param, 
                    min_samples = self._minpts, 
                    metric = distance).fit(X)
            elif hyperparameter=='minpts':
                DBSCAN_model = DBSCAN(
                    eps = self._epsilon, 
                    min_samples = param, 
                    metric = distance).fit(X)
            idx = DBSCAN_model.labels_
            if count_number(idx)[0] == 1: silhuettes.append(0)
            else: silhuettes.append(silhouette_score(X,idx))
        # Plot
        fig, ax = plt.subplots()
        ax.plot(params, silhuettes, color='k')
        ax.set_title("Average silhouettes vs " + hyperparameter)
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel("Average silhouette")
        fig.show()
        # Save the optimal value
        if hyperparameter=='epsilon': self._epsilon = int(input(" Choose the optimal epsilon: "))
        elif hyperparameter=='minpts': self._minpts = int(input(" Choose the optimal minpts: "))


def calculate_CC(idx, MMSI, MMSI_vec, if_all=False):
    """
    Calculates correctness coefficient - indicator of to what extent: \n
    1. each cluster consists of messages from one vessel (CHC),
    2. messages from one vessel are not divided between several clusters (VHC). \n
    Arguments:
    - idx - list of indices of clusters assigned to each message, len=num_messages,
    - MMSI - list of MMSI identifier from each AIS message, len=num_messages,
    - MMSI_vec - list of unique MMSIs in MMSI list,
    - if_all (optional) - Boolean, whether to return also VHC and CHC (default=False). \n
    Returns: 
    - CC - scalar, float, computed correctness coefficient,
    - CHC - scalar, float, computed clusters' homogeneity coefficient,
    - VHC - scalar, float, computed vessels' homogeneity coefficient.
    """
    if idx.shape[0]==0 or len(MMSI)==0 or len(MMSI_vec)==0:
        return 0
    # Compute vessels' homogeneity coefficient
    VHC_vec1 = []
    VHC_vec2 = []
    for id in MMSI_vec:  # For each MMSI value
        same = idx[np.array(MMSI)==id]  # check which clusters consists of data from that MMSI
        _, clusters = count_number(same)  # count how many such clusters there is
        volume = [] # count the volume of each of those clusters
        for cluster in clusters: volume.append(np.sum(np.where(same==cluster,1,0)))
        VHC_vec1.append(np.max(volume)/np.sum(volume))  # find the modal cluster and count the fraction of messages from that MMSI in that cluster
        VHC_vec2.append(np.sum(volume))
    VHC = np.sum(np.multiply(VHC_vec1,VHC_vec2))/np.sum(VHC_vec2) # Calculate the weighted average
    # Compute clusters' homogeneity coefficient
    CHC_vec1 = []
    CHC_vec2 = []
    for id in range(np.min(idx),np.max(idx)):  # For each cluster
        same = np.array(MMSI)[idx==id]  # check which MMSI that cluster consists of 
        _, MMSIs = count_number(same)  # count how many such MMSIs there is
        volume = [] # count the volume of each of those MMSIs
        for MMSI_ in MMSIs:  volume.append(np.sum(np.where(same==MMSI_,1,0)))
        CHC_vec1.append(np.max(volume)/np.sum(volume))  # find the modal MMSI and count the fraction of messages from that MMSI in that cluster
        CHC_vec2.append(np.sum(volume))
    # Calculate the weighted average
    CHC = np.sum(np.multiply(CHC_vec1, CHC_vec2))/np.sum(CHC_vec2)   
    # Compute correctness coefficient as a F1 score
    CC = 2*CHC*VHC/(CHC+VHC)
    if if_all: return CC, CHC, VHC
    else: return CC

def check_cluster_assignment(idx, idx_corr, message_idx):
    """
    Checks if the damaged message is assigned together with other messages from its vessel.
    Arguments:
    - idx - list of indices of clusters assigned to each message, len=num_messages,
    - idx_corr - list of indices of clusters assigned to each message in a dataset, len=num_messages,
    - message_idx - scalar, int, index of a message that was corrupted.
    """
    idx_before = idx[message_idx]
    idx_now = idx_corr[message_idx]
    # Find all messages originally clustered with the corrupted message
    indices_original = np.where(idx == idx_before)
    # Find a cluster that contains most of those messages after the corruption
    percentage = []
    _, idx_corr_vec = count_number(idx_corr)
    for i in idx_corr_vec:  # for each cluster in corrupted data
        indices_cluster = np.where(idx_corr == i)  # find messages from that cluster
        intersection = set(indices_original[0]).intersection(indices_cluster[0])  # find messages both in original cluster and examined cluster
        percentage.append(len(intersection)/len(indices_original[0]))  # calculate how many messages from the original cluster are in examined cluster
    idx_preferable = idx_corr_vec[percentage.index(max(percentage))]  # the cluster with the biggest percentage is probably the right one
    # Check if that cluster is the same as before
    result = idx_now == idx_preferable
    return result