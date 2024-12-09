# Important imports
import numpy as np
import h5py
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.encoder import cluster_encoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
params = {'axes.labelsize': 16,'axes.titlesize':16, 'font.size': 16, 'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(params)
import os
import sys
sys.path.append('.')
from utils.initialization import Data # pylint: disable=import-error
from utils.clustering import calculate_CC
from utils.miscellaneous import count_number

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
language = 'pl' # 'pl' or 'eng' - for graphics only
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
# --------------------------------------------------------------------------------

# Decide what to do
precomputed = 'start'
while precomputed != '1' and precomputed != '2':
    precomputed = input("Choose: \n1 - Run computations from scratch \n2 - Load precomputed values \n")
    if precomputed != '1' and precomputed != '2':
        print("Unrecognizable answer.")

# Load data
print(" Initialization... ")
if precomputed == '2':  # Load file with precomputed values
    file = h5py.File(
        name='research_and_results/00_hyperparameters_clustering'+clustering_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations on the original data
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'hamming']
    num_algorithms = 1 if clustering_algorithm == 'DBSCAN' else 2 # k-means and k-medoids for 'kmeans'
    num_experiments = 2 # standardisation on/off
    num_quality_metrics = 3 if clustering_algorithm == 'DBSCAN' else 2  # silhouette and CC, for DBSCAN additionally number of clusters
    OK_vec = np.zeros((num_algorithms, len(metrics), num_experiments, num_quality_metrics))

    # Import the dataset
    file = h5py.File(name='data/Gdansk.h5', mode='r')
    data = Data(file)
    file.close()
    K, _ = count_number(data.MMSI)  # Count number of groups/ships

    for algorithm in range(num_algorithms):
        for metric in metrics:
            for experiment in range(num_experiments):
                # Do nothing for Hamming and standardization setup
                if metric=='hamming' and experiment==1: 
                    OK_vec[algorithm, metric, experiment, 0] = None
                    OK_vec[algorithm, metric, experiment, 1] = None
                    if clustering_algorithm=='DBSCAN': OK_vec[algorithm, metric, experiment, 2] = None
                else:
                    # Prepare data for clustering
                    if metric == 'hamming':
                        dist_metric = 'euclidean'
                        data_ = data.message_bits
                    else: 
                        dist_metric = metric
                        if experiment: data_= data.standardize(data.Xraw)[0]
                        else: data_ = data.Xraw

                    # Perform actual clustering
                    if clustering_algorithm == 'kmeans':  
                        initial_centers = random_center_initializer(data=data_, amount_centers=K, random_state=1).initialize()
                        if algorithm: # 0 -> k-means
                            km_model = kmeans(
                                data=data_, 
                                initial_centers=initial_centers, 
                                metric= distance_metric(metrics.index(dist_metric)+1))
                        else: # 1 -> k-medoids
                            km_model = kmedoids(
                                data=data_,
                                initial_centers=initial_centers, 
                                metric= distance_metric(metrics.index(dist_metric)+1))
                        km_model.process()
                        clusters = km_model.get_clusters()
                        type_repr = km_model.get_cluster_encoding()
                        encoder = cluster_encoder(type_repr, clusters, data_)
                        encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
                        idx = encoder.get_clusters()

                    elif clustering_algorithm == 'DBSCAN':
                        if metric == 'euclidean': epsilon = 3.16
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        else: pass # Choose optimal epsilon
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!! 
                        DBSCAN_model = DBSCAN(
                            eps = epsilon, 
                            min_samples = 1, 
                            metric = dist_metric)
                        idx = DBSCAN_model.labels_
                        K = count_number(idx)[0]
                        OK_vec[algorithm, metric, experiment, 1] = K

                # Compute quality measures
                if K==1: OK_vec[algorithm, metric, experiment, 0] = 0
                elif K==len(idx): OK_vec[algorithm, metric, experiment, 0] = 1
                else: OK_vec[algorithm, metric, experiment, 0] = silhouette_score(data_, idx)
                MMSIs, MMSI_vec = count_number(data.MMSI)
                OK_vec[algorithm, metric, experiment, 1] = calculate_CC(idx, data.MMSI, MMSI_vec)


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/00_hyperparameters_clustering'+clustering_algorithm+'.h5'):
        os.remove('research_and_results/00_hyperparameters_clustering'+clustering_algorithm+'.h5')
    File = h5py.File('research_and_results/00_hyperparameters_clustering'+clustering_algorithm+'.h5', mode='a')
    File.create_dataset('OK_vec', data=OK_vec)
    File.close()