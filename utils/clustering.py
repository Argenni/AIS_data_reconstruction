"""
Functions and classes used in clustering stage of AIS message reconstruction
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
params = {'axes.labelsize': 16,'axes.titlesize':16, 'font.size': 16, 'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(params)
import sys
sys.path.append('.')
from utils.miscellaneous import count_number
import math

class Clustering:
    """
    Class that introduces AIS data clustering using either k-means or DBSCAN.
    """
    _epsilon = 3.16
    _minpts = 1
    _language = [] # 'pl' or 'eng' - for graphics only
    _verbose = []

    def __init__(self, language='eng', verbose=False):
        """
        Class initialization (class object creation). \n
        Arguments: 
        - language - string, 'pl' for Polish or 'eng' for English (only for graphics text translation),
        - verbose (optional) - Boolean, whether to print running logs or not, default=False
        """
        self._verbose = verbose
        self._language = language

    def run_kmeans(self, X, K, optimize=None, MMSI=[]):
        """
        Runs k-means algorithm on a given dataset with Euclidean distance metric. \n
        Arguments: 
        - X - numpy array with dataset to cluster, shape=(num_message, num_features (115)),
        - K - scalar, int, desired number of clusters,
        - optimize - (optional) string, name of k-means hyperparameter to optimize ('K'), 
            default=None (no optimization), 
        - MMSI (optional) - list of MMSI identifiers from each AIS message, len=num_messages
            (for hyperparameter tuning; only required if optimize!=None). \n\n
        Returns:
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - centroids - numpy array with centers of each cluster, shape=(K, num_features (115)).
        """
        # Optimize hyperparametres if allowed
        if len(MMSI)>0: 
            if optimize=='K': K = self.optimize_kmeans(X, K, MMSI)
        else: 
            if optimize=="K": print(" Cannot perform k-means hyperparameter tuning: no MMSI provided.")
        # Cluster using k-means
        if self._verbose: print("Running k-means clustering...")
        kmeans_model = KMeans(n_clusters=K, n_init=10, max_iter=100, tol=0.001, random_state=0).fit(X)
        idx = kmeans_model.labels_
        centroids = kmeans_model.cluster_centers_
        if self._verbose: print(" Complete.")
        return idx, centroids
    
    def optimize_kmeans(self, X, K, MMSI):
        """
        Searches for optimal number of clusters for k-means to create in terms of
        silhouette, CC and desired number of clusters.
        Arguments: 
        - X - numpy array with dataset to cluster, shape=(num_message, num_features (115)),
        - K - scalar, int, number of ships in a dataset,
        - MMSI - list of MMSI identifiers from each AIS message, len=num_messages. \n
        Returns: Knew - optimal number of clusters.
        """
        Ks = range(2, int(K*1.5))
        silhouettes = []
        CCs = []
        costs = []
        print(" Search for optimal K...")
        for K_0 in Ks:
            kmeans_model = KMeans(n_clusters=K_0, n_init=10, max_iter=100, tol=0.001, random_state=0).fit(X)
            idx = kmeans_model.labels_
            centroids = kmeans_model.cluster_centers_
            if K==len(idx): silhouettes.append(1)
            else: silhouettes.append(silhouette_score(X,idx))
            CCs.append(calculate_CC(idx, MMSI, count_number(MMSI)[1]))
            cost = [math.dist(X[i,:],centroids[idx[i]]) for i in range(len(idx))]
            costs.append(np.mean(cost))
        # Plot
        fig, ax = plt.subplots(ncols=3)
        ax[0].plot(Ks[0:K-1], np.ones((K-1))*silhouettes[K-2], color='r', linestyle='dashed')
        ax[0].vlines(x=K, ymin=min(silhouettes), ymax=silhouettes[K-2], color='r', linestyle='dashed')
        ax[0].plot(Ks, silhouettes, color='k')
        ax[0].scatter(Ks, silhouettes, color='k', s=6)
        ax[0].set_xlabel("K")
        ax[0].set_ylabel("Silhouette")
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[1].plot(Ks[0:K-1], np.ones((K-1))*CCs[K-2], color='r', linestyle='dashed')
        ax[1].vlines(x=K, ymin=min(CCs), ymax=CCs[K-2], color='r', linestyle='dashed')
        ax[1].plot(Ks, CCs, color='k')
        ax[1].scatter(Ks, CCs, color='k', s=6)
        ax[1].set_xlabel("K")
        ax[1].set_ylabel("CC")
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[2].plot(Ks, costs, color='k')
        ax[2].scatter(Ks, costs, color='k', s=6)
        ax[2].set_xlabel("K")
        if self._language == 'eng': ax[2].set_ylabel("Average cost")
        elif self._language == 'pl': ax[2].set_ylabel("Średni koszt")
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        fig.show()
        # Save the optimal value
        Knew = int(input(" Choose the optimal K: "))
        return Knew

    def run_DBSCAN(self, X, distance='euclidean', optimize=None, MMSI=[]):
        """
        Runs DBSCAN algorithm on a given dataset with given distance metric. \n
        Arguments: 
        - X - numpy array with dataset to cluster, shape=(num_message, num_features (115)),
        - distance - (optional) string, name of distance metric, default='euclidean',
        - optimize - (optional) string, name of DBSCAN hyperparameter to optimize, 
            'epsilon' or 'minpts', default=None (no optimization),
        - MMSI (optional) - list of MMSI identifiers from each AIS message, len=num_messages
            (for hyperparameter tuning; only required if optimize!=None). \n
        Returns:
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - K - scalar, int, number of clusters created by DBSCAN.
        """
        # Optimize hyperparametres if allowed
        if len(MMSI)>0: 
            if optimize == 'epsilon': self._optimize_DBSCAN(X, MMSI, distance, hyperparameter='epsilon')
            elif optimize == 'minpts': self._optimize_DBSCAN(X, MMSI, distance, hyperparameter='minpts')
        else: 
            if optimize=='epsilon' or optimize=='minpts':
                print(" Cannot perform DBSCAN hyperparameter tuning: no MMSI provided.")
        # Cluster using DBSCAN
        if self._verbose: print("Running DBSCAN clustering...")
        DBSCAN_model = DBSCAN(eps = self._epsilon, min_samples = self._minpts, metric = distance).fit(X)
        idx = DBSCAN_model.labels_
        K, _ = count_number(idx)
        if self._verbose:print(" Complete.")
        return idx, K

    def _optimize_DBSCAN(self, X, MMSI, distance, hyperparameter):
        """
        Searches for optimal epsilon or minpts value for AIS data clustering using DBSCAN
        and stores it in self._epsilon or self._minpts. Arguments:
        - X - numpy array with training dataset, shape=(num_message, num_features (115)),
        - MMSI - list of MMSI identifiers from each AIS message, len=num_messages
        - distance - string, name of distance metric, eg. 'euclidean',
        - hyperparameter - string, name of DBSCAN hyperparameter to optimize, 'epsilon' or 'minpts'.
        """
        params = [1, 2, 5, 10, 20, 50, 100]
        silhouettes = []
        CCs = []
        clusters = []
        print(" Search for optimal " + hyperparameter + "...")
        for param in params:
            if hyperparameter=='epsilon':
                DBSCAN_model = DBSCAN(
                    eps = np.sqrt(param), 
                    min_samples = self._minpts, 
                    metric = distance).fit(X)
            elif hyperparameter=='minpts':
                DBSCAN_model = DBSCAN(
                    eps = self._epsilon, 
                    min_samples = param, 
                    metric = distance).fit(X)
            idx = DBSCAN_model.labels_
            K = count_number(idx)[0]
            clusters.append(K)
            if K==1: silhouettes.append(0)
            elif K==len(idx): silhouettes.append(1)
            else: silhouettes.append(silhouette_score(X,idx))
            MMSIs, MMSI_vec = count_number(MMSI)
            CCs.append(calculate_CC(idx, MMSI, MMSI_vec))
        # Plot
        fig, ax = plt.subplots(ncols=3)
        ax[0].plot(params, silhouettes, color='k')
        ax[0].scatter(params, silhouettes, color='k', s=6)
        ax[0].set_xlabel(hyperparameter)
        ax[0].set_ylabel("Silhouette")
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[1].plot(params, CCs, color='k')
        ax[1].scatter(params, CCs, color='k', s=6)
        ax[1].set_xlabel(hyperparameter)
        ax[1].set_ylabel("CC")
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[2].plot(params, clusters, color='k')
        ax[2].scatter(params, clusters, color='k', s=6)
        ax[2].plot(params, np.ones((len(params)))*MMSIs, color='r')
        ax[2].set_xlabel(hyperparameter)
        if self._language=='eng': ax[2].set_ylabel("No. clusters")
        elif self._language=='pl': ax[2].set_ylabel("Liczba grup")
        ax[2].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False)
        fig.show()
        # Save the optimal value
        if hyperparameter=='epsilon': self._epsilon = np.round(np.sqrt(float(input(" Choose the optimal epsilon: "))),2)
        elif hyperparameter=='minpts': self._minpts = int(input(" Choose the optimal minpts: "))


def calculate_CC(idx, MMSI, MMSI_vec, if_all=False):
    """
    Calculates correctness coefficient - indicator of to what extent: \n
    1. each cluster consists of messages from one vessel (CHC),
    2. messages from one vessel are not divided between several clusters (VHC). \n
    Arguments:
    - idx - list of indices of clusters assigned to each message, len=num_messages,
    - MMSI - list of MMSI identifiers from each AIS message, len=num_messages,
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
    for id in range(np.min(idx),np.max(idx)+1):  # For each cluster
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