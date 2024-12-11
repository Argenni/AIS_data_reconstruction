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
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data # pylint: disable=import-error
from utils.clustering import calculate_CC
from utils.miscellaneous import count_number

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
language = 'pl' # 'pl' or 'eng' - for graphics only
clustering_algorithm = 'kmeans'  # 'kmeans' or 'DBSCAN'
# --------------------------------------------------------------------------------

# Import the dataset
file = h5py.File(name='data/Gdansk.h5', mode='r')
data = Data(file)
file.close()
K, _ = count_number(data.MMSI)  # Count number of groups/ships

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
        name='research_and_results/00_hyperparameters_clustering_'+clustering_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    titles = np.array(file.get('titles'))
    file.close()

else:  # or run the computations on the original data
    distance_metrics_official = ['euclidean', 'manhattan', 'chebyshev']
    if language == 'eng':
        experiments = ['standardisation on', 'standardisation off']
        metrics = ['euclidean', 'manhattan', 'chebyshev', 'hamming']
    elif language == 'pl':
        experiments = ['standaryzacja', 'bez standaryzacji']
        metrics = ['euklidesowa', 'Manhattan', 'Czebyszewa', 'Hamminga']
    algorithms = ['DBSCAN, '] if clustering_algorithm == 'DBSCAN' else [', k-means, ', ', k-medoids, ']
    num_quality_metrics = 3 if clustering_algorithm == 'DBSCAN' else 2  # silhouette and CC, for DBSCAN additionally number of clusters
    OK_vec = np.zeros((len(algorithms), len(metrics), len(experiments), num_quality_metrics))
    titles = np.empty((len(algorithms), len(metrics), len(experiments)), dtype=object)

    print("Analysing...")
    for num_algorithm, algorithm in enumerate(algorithms):
        if clustering_algorithm=='kmeans' and algorithm==', k-medoids, ': 
            initial_index_medoids=np.random.choice(np.arange(len(data.MMSI)), K)
        for num_metric, metric in enumerate(metrics):
            for num_experiment, experiment in enumerate(experiments):
                if clustering_algorithm=='kmeans': title = metric + algorithm + experiment
                else: title = metric + ", " + experiment     
                titles[num_algorithm, num_metric, num_experiment] = title
                # Do nothing for Hamming and standardization setup
                if (metric=='hamming' or metric=='Hamminga') and (experiment=='standardisation on' or experiment=='standaryzacja'): 
                    OK_vec[num_algorithm, num_metric, num_experiment, 0] = None
                    OK_vec[num_algorithm, num_metric, num_experiment, 1] = None
                    if clustering_algorithm=='DBSCAN': OK_vec[num_algorithm, num_metric, num_experiment, 2] = None
                else: 
                    # Prepare data for clustering
                    print(title)
                    if metric == 'hamming' or metric == 'Hamminga':
                        dist_metric = 'manhattan'
                        data_ = data.message_bits
                    else: 
                        dist_metric = distance_metrics_official[metrics.index(metric)]
                        if experiment=='standaryzacja' or experiment=='standardisation on': data_= data.standardize(data.Xraw)[0]
                        else: data_ = copy.deepcopy(data.Xraw)

                    # Perform actual clustering
                    metric_idx = 0 if dist_metric=='euclidean' else distance_metrics_official.index(dist_metric)+1
                    if clustering_algorithm == 'kmeans':  
                        if algorithm == ', k-means, ':
                            initial_centers = random_center_initializer(data=data_, amount_centers=K, random_state=0).initialize()
                            km_model = kmeans(
                                data=data_, 
                                initial_centers=initial_centers, 
                                tolerance=0.001,
                                metric=distance_metric(metric_idx),
                                itermax=100)
                        else: # k-medoids
                            km_model = kmedoids(
                                data=data_,
                                initial_index_medoids=initial_index_medoids, 
                                metric= distance_metric(metric_idx))
                        km_model.process()
                        clusters = km_model.get_clusters()
                        type_repr = km_model.get_cluster_encoding()
                        encoder = cluster_encoder(type_repr, clusters, data_)
                        encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
                        idx = encoder.get_clusters()
                        K_new = K

                    elif clustering_algorithm == 'DBSCAN':
                        if (metric=='euclidean' or metric=='euklidesowa')and(experiment=='standardisation on' or experiment=='standaryzacja'): epsilon = 3.16
                        # ----------------------------
                        else: # Choose optimal epsilon
                            silhouettes = []
                            CCs = []
                            clusters = []
                            if experiment=='standardisation off' or experiment=='bez standaryzacji': params=[1,5,10,20,50,100,200,500]
                            else: params=[1,2,5,10,20,50,100]
                            if metric=='Hamminga' or metric=='hamming': params=[0.1, 0.5, 1, 10, 20, 50] #params=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
                            for param in params:
                                DBSCAN_model = DBSCAN(
                                    eps = np.sqrt(param) if (metric=='euclidean' or metric=='euklidesowa') else param,
                                    min_samples=1, 
                                    metric=dist_metric).fit(data_)
                                idx = DBSCAN_model.labels_
                                K_param = count_number(idx)[0]
                                clusters.append(abs(K_param-K)/K)
                                if K_param==1 or K_param==len(idx): silhouettes.append(0)
                                else: silhouettes.append(silhouette_score(data_,idx))
                                MMSI_vec = count_number(data.MMSI)[1]
                                CCs.append(calculate_CC(idx, data.MMSI, MMSI_vec))
                            epsilon = (params[np.argmax(silhouettes)] + params[np.argmax(CCs)] + params[np.argmin(clusters)])/3
                            print("Epsilon: " + str(epsilon))
                            if metric=='euclidean' or metric=='euklidesowa': epsilon = np.sqrt(epsilon)
                        # ----------------------------
                        DBSCAN_model = DBSCAN(
                            eps = epsilon, 
                            min_samples = 1, 
                            metric = dist_metric).fit(data_)
                        idx = DBSCAN_model.labels_
                        K_new = count_number(idx)[0]
                        OK_vec[num_algorithm, num_metric, num_experiment, 2] = K_new

                    # Compute quality measures
                    if K_new==1 or K_new==len(idx): OK_vec[num_algorithm, num_metric, num_experiment, 0] = 0
                    else: OK_vec[num_algorithm, num_metric, num_experiment, 0] = silhouette_score(data_, idx)
                    MMSI_vec = count_number(data.MMSI)[1]
                    OK_vec[num_algorithm, num_metric, num_experiment, 1] = calculate_CC(np.array(idx), data.MMSI, MMSI_vec)


# Visualize
titles_flat = titles.ravel()
titles_flat = np.delete(titles_flat, -2)
if clustering_algorithm == 'kmeans': titles_flat = np.delete(titles_flat, 6)

silh_flat = OK_vec[:,:,:,0].ravel()
silh_flat = np.delete(silh_flat, -2)
if clustering_algorithm == 'kmeans': silh_flat = np.delete(silh_flat, 6)
indices_silh = np.flip(np.argsort(silh_flat))
fig, ax = plt.subplots()
ax.bar(titles_flat[indices_silh], silh_flat[indices_silh], width=0.4)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Silhouette")
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.5, box.width, box.height*0.5])
fig.show()

cc_flat = OK_vec[:,:,:,1].ravel()
cc_flat = np.delete(cc_flat, -2)
if clustering_algorithm == 'kmeans': cc_flat = np.delete(cc_flat, 6)
indices_cc = np.flip(np.argsort(cc_flat))
fig, ax = plt.subplots()
ax.bar(titles_flat[indices_cc], cc_flat[indices_cc], width=0.4)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("CC")
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.5, box.width, box.height*0.5])
fig.show()

if clustering_algorithm == 'DBSCAN':
    K_flat = OK_vec[:,:,:,2].ravel()
    K_flat = np.delete(K_flat, -2)
    indices_K = np.flip(np.argsort(K_flat))
    fig, ax = plt.subplots()
    ax.bar(titles_flat[indices_K], K_flat[indices_K], width=0.4)
    ax.plot(np.ones_like(K_flat)*K, color='r')
    ax.tick_params(axis='x', labelrotation=90)
    if language=='pl': ax.set_ylabel("Liczba grup")
    elif language=='eng': ax.set_ylabel("Number of clusters")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+box.height*0.5, box.width, box.height*0.5])
    fig.show()


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/00_hyperparameters_clustering_'+clustering_algorithm+'.h5'):
        os.remove('research_and_results/00_hyperparameters_clustering_'+clustering_algorithm+'.h5')
    File = h5py.File('research_and_results/00_hyperparameters_clustering_'+clustering_algorithm+'.h5', mode='a')
    File.create_dataset('OK_vec', data=OK_vec)
    File.create_dataset('titles', data=titles)
    File.close()