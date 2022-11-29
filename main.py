# ----------------------------------------------- AIS Data Reconstruction ---------------------------------------------
""""
Loads prepared AIS data, preprocesses it and starts the reconstruction process:
1. clustering with DBSCAN or kmeans to distinguish individual trajecotries
2. for each trajectory, detection of anomalies in it
3. for each outlier found - prediction of its correct form 
Requires: Gdansk.h5 (from data_Gdansk.py), Baltic.h5 (from data_Baltic.py) 
 or Gibraltar.h5 (from data_Gibralar.py) with the following datasets:
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_mesages, num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages, num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages
 - timestamp - list of strings with timestamp of each message, len = num_messages
Creates:
 - idx - list of indices of clusters assigned to each message, len = num_messages
 - outliers - numpy array with anomaly detection information, shape = (num_messages, 3)
   (1. column - if a message is outlier, 2. column - proposed correct cluster, 3. column - possibly damaged feature)
"""
print("\n----------- AIS Data Reconstruction ---------- ")

# ----------- Part 0 - Initialization ----------
print("\n----------- Part 0 - Initialization ---------- ")
# Important imports
import numpy as np
import h5py
from sklearn import metrics
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
#--------------------------------------------------------------------------------

# Load data
print(" Importing files... ")
file = h5py.File(name='data/Gdansk.h5', mode='r') # Gdansk.h5, Gibraltar.h5 or Baltic.h5
data = Data(file=file)
data.split(train_percentage=50, val_percentage=25) # split into train, val and test set
file.close()

# Preprocess data
print(" Preprocessing data... ")
time_window = TimeWindow(0, 35)  # apply 25-min time window
data = time_window.use_time_window(data)
K_train, MMSI_vec_train = count_number(data.MMSI_train)
K_val, MMSI_vec_val = count_number(data.MMSI_val)
K, MMSI_vec = count_number(data.MMSI)  # count number of groups/ships
visualize_trajectories(
    X=data.Xraw,
    MMSI=data.MMSI,
    MMSI_vec=MMSI_vec,
    goal='data_visualization'
    )
data.X_train, mu_train, sigma_train = data.normalize(data.Xraw_train)
data.X_val, mu_val, sigma_val = data.normalize(data.Xraw_val)
data.X, mu, sigma = data.normalize(data.Xraw)
print(" Complete.")


# ----------- Part 1 - Clustering ----------------
print("\n----------- Part 1 - Grouping ---------- ")
clustering = Clustering()
if clustering_algorithm == 'kmeans':
    print(" Running k-means clustering...")
    idx, centroids = clustering.run_kmeans(X=data.X, K=K)
elif clustering_algorithm == 'DBSCAN':
    print(" Running DBSCAN clustering...")
    idx, K = clustering.run_DBSCAN(X=data.X, distance=distance, optimize=None)
print(" Complete.")
silhouette = metrics.silhouette_score(data.X,idx)
print(" Average silhouette: " + str(round(silhouette,2)))
CC = calculate_CC(idx, data.MMSI, MMSI_vec)
print(" Correctness coefficient: " + str(round(CC,4)))
visualize_trajectories(
    X=data.Xraw,
    MMSI=idx,
    MMSI_vec=range(-1, np.max(idx)+1),
    goal='clustering'
)


# ----------- Part 2 - Anomaly detection ------- 
print("\n----------- Part 2 - Anomaly detection ---------- ")
print(" Looking for anomalies...")
outliers = AnomalyDetection(
    data=data,
    if_visualize=True,
    optimize=None # 'max_depth', 'n_estimators', 'k', None
    )
# Conduct anomaly detection - search for standalone clusters
outliers.detect_standalone_clusters(
    idx=idx,
    idx_vec=range(-1, np.max(idx)+1),
    X=data.X,
    message_decoded=data.message_decoded,
    )
# Conduct anomaly detection - search inside proper clusters
outliers.detect_inside(
    idx=idx, 
    message_decoded=data.message_decoded
    )
print(" Anomalies found: " + str(np.sum(np.array(outliers.outliers, dtype=object)[:,0])))
visualize_trajectories(
    X=data.Xraw,
    MMSI=np.array(outliers.outliers, dtype=object)[:,0].tolist(),
    MMSI_vec=[0,1],
    goal='anomaly_detection'
    )

# Save results
np.savetxt(
    'output/output.txt',
    np.array(outliers.outliers, dtype=object), 
    delimiter=',',
    fmt='%s',
    header="If_outlier, Correct_cluster_id, Damage_fields")
input("Press Enter to exit...")