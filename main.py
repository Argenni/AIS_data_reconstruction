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
Creates:
 - idx - list of indices of clusters assigned to each message, len=num_messages (clustering.txt),
 - outliers - numpy array with anomaly detection information, shape=(num_messages, 3) (anomaly_detection.txt)
   (1. column - if a message is outlier, 2. column - proposed correct cluster, 3. column - possibly damaged field).
 - predicitons - list of messages that underwent correction (in the form of dictionary with 
    'message_idx':index of reconstructed message, field_num:reconstructed value for that field) (prediction.txt),
 - data/reconstructed_Gdansk.h5 / data/reconstructed_Baltic.h5 / data/reconstructed_Gibraltar.h5 - corrected datasets 
    (message_bits and message_decoded).
   """
print("\n----------- AIS Data Reconstruction ---------- ")

# ----------------------------------------------- AIS Data Reconstruction ---------------------------------------------
# ------------------- Initialization ----------------------
print("\n----------- Initialization ---------- ")
# Important imports
import numpy as np
import h5py
from sklearn.metrics import silhouette_score
import sys
import os
sys.path.append('.')
from utils.initialization import Data
from utils.clustering import Clustering
from utils.anomaly_detection import AnomalyDetection
from utils.prediction import Prediction
from utils.miscellaneous import count_number, visualize_trajectories, TimeWindow

# ----------------------------!!! EDIT HERE !!! --------------------------------- 
# Specify some important configuration
np.random.seed(1) #For reproducibility
language = 'pl' # 'pl' or 'eng' - for graphics only
filename = 'Gdansk.h5' # Gdansk.h5, Gibraltar.h5 or Baltic.h5
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
prediction_algorithm = 'xgboost' # 'ar' or 'xgboost'
wavelet = 'morlet' # 'morlet' or 'ricker'
#--------------------------------------------------------------------------------

# Load data
print(" Importing files... ")
file = h5py.File(name='data/'+filename, mode='r')
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
    goal='data_visualization',
    language=language)
data.X_train, mu_train, sigma_train = data.standardize(data.Xraw_train)
data.X_val, mu_val, sigma_val = data.standardize(data.Xraw_val)
data.X, mu, sigma = data.standardize(data.Xraw)
print(" Complete.")


# -------------------------- Stage 1 - Clustering -------------------------
print("\n----------- Part 1 - Clustering ---------- ")
clustering = Clustering(language=language, verbose=True)
if clustering_algorithm == 'kmeans':
    idx, centroids = clustering.run_kmeans(X=data.X, K=K, optimize=None, MMSI=data.MMSI)
elif clustering_algorithm == 'DBSCAN':
    idx, K = clustering.run_DBSCAN(X=data.X, distance=distance, optimize=None, MMSI=data.MMSI)
silhouette = silhouette_score(data.X,idx)
print("Average silhouette: " + str(round(silhouette,2)))
visualize_trajectories(
    X=data.Xraw,
    MMSI=idx,
    MMSI_vec=range(-1, np.max(idx)+1),
    goal='clustering',
    language=language)


# ------------------------- Stage 2 - Anomaly detection --------------------- 
print("\n----------- Part 2 - Anomaly detection ---------- ")
ad = AnomalyDetection(
    verbose=True,
    optimize=None, # 'max_depth', 'n_estimators', 'k', 'max_depth2', 'n_estimators2', None
    ad_algorithm=ad_algorithm,
    wavelet=wavelet,
    language=language)
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
    goal='anomaly_detection',
    language=language)


# ------------------------- Stage 3 - Prediction --------------------- 
print("\n----------- Part 3 - Prediction ---------- ")
prediction = Prediction(
    verbose=True,
    optimize=None, # 'lags', 'max_depth' or 'n_estimators'
    prediction_algorithm=prediction_algorithm,
    language=language)
message_bits_new, message_decoded_new, idx_new =  prediction.find_and_reconstruct_data(
    message_decoded=data.message_decoded, 
    message_bits=data.message_bits,
    idx=idx,
    timestamp=data.timestamp,
    outliers=ad.outliers)
print("Messages corrected: " + str(len(prediction.predictions)))
reconstructed_idx = [prediction.predictions[i]['message_idx'] for i in range(len(prediction.predictions))]
visualize_trajectories(
    X=message_decoded_new[:,[7,8]],
    MMSI=message_decoded_new[:,2],
    MMSI_vec=count_number(message_decoded_new[:,2])[1],
    goal='prediction',
    language=language,
    reconstructed_idx=reconstructed_idx)


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
np.savetxt(
    'output/prediction.txt',
    np.array(prediction.predictions, dtype=object), 
    delimiter=',',
    fmt='%s',
    header="Predicted new values")
if os.path.exists('data/reconstructed_'+filename):
    os.remove('data/reconstructed_'+filename)
File = h5py.File('data/reconstructed_'+filename, mode='x')
File.create_dataset('message_bits', data=message_bits_new)
File.create_dataset('message_decoded', data=message_decoded_new)
File.close()