""""
Loads prepared AIS data, preprocesses it and conducts the reconstruction process:
1. clustering with DBSCAN or kmeans to distinguish individual trajectories,
2. for each trajectory - detection of anomalies in it,
3. for each outlier found - prediction of its correct form. \n
Requires: Gdansk.h5 (from data_Gdansk.py), Baltic.h5 (from data_Baltic.py) or Gibraltar.h5 (from data_Gibralar.py)
with the following datasets:
 - message_bits - numpy array, AIS messages in binary form (1 column = 1 bit), shape=(num_mesages, num_bits (168)),
 - message_decoded - numpy array, AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifiers from each AIS message, len=num_messages,
 - timestamp - list of strings with timestamp of each message, len  um_messages. \n
Creates (saved as .txt):
 - idx - list of indices of clusters assigned to each message, len=num_messages (clustering.txt),
 - outliers - numpy array with anomaly detection information, shape=(num_messages, 3) (anomaly_detection.txt)
   (1. column - if a message is outlier, 2. column - proposed correct cluster, 3. column - possibly damaged field).
"""
print("\n----------- AIS Data Reconstruction ---------- ")

# ----------------------------------------------- AIS Data Reconstruction ---------------------------------------------
# ----------- Initialization ----------
print("\n----------- Initialization ---------- ")
# Important imports
import numpy as np
import h5py
from sklearn.metrics import silhouette_score
import sys
sys.path.append('.')
from utils.initialization import Data
from utils.clustering import Clustering, calculate_CC
from utils.anomaly_detection import AnomalyDetection
from utils.miscellaneous import count_number, visualize_trajectories, TimeWindow

# ----------------------------!!! EDIT HERE !!! --------------------------------- 
# Specify some important configuration
np.random.seed(1) #For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
wavelet = 'morlet' # 'morlet' or 'ricker'
#--------------------------------------------------------------------------------

# Load data
print(" Importing files... ")
file = h5py.File(name='data/Gdansk.h5', mode='r') # Gdansk.h5, Gibraltar.h5 or Baltic.h5
data = Data(file=file)
data.split(train_percentage=50, val_percentage=25) # split into train (50%), val (25%) and test (25%) set
file.close()

# Preprocess data
print(" Preprocessing data... ")
time_window = TimeWindow(0, 35)  # apply 35-min time window
data = time_window.use_time_window(data)
K_train, MMSI_vec_train = count_number(data.MMSI_train)
K_val, MMSI_vec_val = count_number(data.MMSI_val)
K, MMSI_vec = count_number(data.MMSI)  # count number of groups/ships
visualize_trajectories(
    X=data.Xraw,
    MMSI=data.MMSI,
    MMSI_vec=MMSI_vec,
    goal='data_visualization')
data.X_train, mu_train, sigma_train = data.standarize(data.Xraw_train)
data.X_val, mu_val, sigma_val = data.standarize(data.Xraw_val)
data.X, mu, sigma = data.standarize(data.Xraw)
print(" Complete.")


# -------------------------- Part 1 - Clustering -------------------------
print("\n----------- Part 1 - Clustering ---------- ")
clustering = Clustering()
if clustering_algorithm == 'kmeans':
    idx, centroids = clustering.run_kmeans(X=data.X, K=K)
elif clustering_algorithm == 'DBSCAN':
    idx, K = clustering.run_DBSCAN(X=data.X, distance=distance, optimize=None)
silhouette = silhouette_score(data.X,idx)
print("Average silhouette: " + str(round(silhouette,2)))
CC = calculate_CC(idx, data.MMSI, MMSI_vec)
print("Correctness coefficient: " + str(round(CC,4)))
visualize_trajectories(
    X=data.Xraw,
    MMSI=idx,
    MMSI_vec=range(-1, np.max(idx)+1),
    goal='clustering')


# ------------------------- Part 2 - Anomaly detection --------------------- 
print("\n----------- Part 2 - Anomaly detection ---------- ")
ad = AnomalyDetection(
    if_visualize=True,
    optimize=None, # 'max_depth', 'n_estimators', 'k', 'max_depth2', 'n_estimators2', None
    ad_algorithm=ad_algorithm,
    wavelet=wavelet)
ad.detect_in_1element_clusters(
    idx=idx,
    idx_vec=range(-1, np.max(idx)+1),
    X=data.X,
    message_decoded=data.message_decoded)
ad.detect_in_multielement_clusters(
    idx=idx, 
    message_decoded=data.message_decoded,
    timestamp=data.timestamp)
print("Anomalies found: " + str(np.sum(np.array(ad.outliers, dtype=object)[:,0])))
visualize_trajectories(
    X=data.Xraw,
    MMSI=np.array(ad.outliers, dtype=object)[:,0].tolist(),
    MMSI_vec=[0,1],
    goal='anomaly_detection')


# ------------------ Finalization --------------------
# Save results
input("\nPress Enter to save results and exit...")
np.savetxt(
    'output/clustering.txt',
    np.array(idx, dtype=object), 
    delimiter=',',
    fmt='%s',
    header="Cluster_id")
np.savetxt(
    'output/anomaly_detection.txt',
    np.array(ad.outliers, dtype=object), 
    delimiter=',',
    fmt='%s',
    header="If_outlier, Correct_cluster_id, Damage_fields")
