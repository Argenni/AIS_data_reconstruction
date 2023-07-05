# ------------------ Examine the impact of observation time on AIS message reconstruction --------------------
"""
Analyse the datasets using different time windows and check the performace
of all stages of AIS message reconstruction.
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_messages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_messages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (805)
Creates 00_timecomp_.h5 file, with OK_vec with average:
 - if clustering: silhouette and CC,
 - if anomaly detection: F1 score of detecting messages and fields
"""
print("\n----------- AIS message reconstruction - observation time comparison  --------- ")

# ----------- Part 0 - Initialization ----------
# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
plt.rcParams.update({'font.size': 16})
import datetime
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption, TimeWindow

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
stage = 'ad' # 'clustering', 'ad' or 'prediction'
filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
percentages_clust = [0, 5, 10, 20]
percentages_ad = [5, 10, 20]
windows = [5, 10, 15, 20, 30, 60, 120, 180, 360]
bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields

# --------------------------------------------------------------------------------

# Decide what to do
precomputed = 'start'
while precomputed != '1' and precomputed != '2':
    precomputed = input("Choose: \n1 - Run computations from scratch \n2 - Load precomputed values \n")
    if precomputed != '1' and precomputed != '2':
        print("Unrecognizable answer.")

# Load data
print(" Importing files... ")
if precomputed == '2':  # Load file with precomputed values
    if stage == 'clustering':
        file = h5py.File(name='research_and_results/00_timecomp_' + clustering_algorithm, mode='r')
    elif stage == 'ad':
        file = h5py.File(name='research_and_results/00_timecomp_' + ad_algorithm, mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()
else:  # or run the computations

    if stage == 'clustering': percentages = percentages_clust
    elif stage == 'ad': percentages = percentages_ad
    OK_vec = np.zeros((len(windows), len(percentages), 2)) # For computed quality measures for each time window length
    for file in filename:
        # Load the data from the right file
        file = h5py.File(name='data/' + filename[file], mode='r')
        data_original = Data(file)
        data_original.split(train_percentage=50, val_percentage=25)
        file.close()
        # Initialize some variables
        overall_time = datetime.timedelta(minutes=max(data_original.timestamp)-min(data_original.timestamp))
        for window in windows:
            start = 0
            stop = start + window
            while stop <= overall_time:
                # Select only messages from the given time window
                data = copy.deepcopy(data_original)
                time_window = TimeWindow(start, stop)
                time_window.use_time_window(data)
                OK_vec_per = []
                for percentage in percentages:
                    # Corrupt data
                    Xraw_corr = copy.deepcopy(data.Xraw)
                    MMSI_corr = copy.deepcopy(data.MMSI)
                    message_decoded_corr = copy.deepcopy(data.message_decoded)
                    corruption = Corruption(data.X,1)
                    outliers = AnomalyDetection(data=data, ad_algorithm=ad_algorithm)
                    messages = []
                    num_messages = int(len(data.MMSI)*percentage/100)
                    for n in range(num_messages):
                        # Choose 0.05 or 0.1 of all messages and corrupt 2 their random bits
                        bits_corr = np.random.choice(bits, size=2, replace=False)
                        fields = [sum(field_bits <= bit) for bit in np.sort(bits_corr)]
                        message_bits_corr, message_idx = corruption.corrupt_bits(message_bits=data.message_bits, bit_idx=bits_corr[0])
                        message_bits_corr, message_idx = corruption.corrupt_bits(message_bits_corr, message_idx=message_idx, bit_idx=bits_corr[1])
                        messages.append(message_idx)
                        # put it back to the dataset
                        X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
                        Xraw_corr[message_idx,:] = X_0
                        MMSI_corr[message_idx] = MMSI_0
                        message_decoded_corr[message_idx,:] = message_decoded_0
                    # Preprocess data                  
                    K, _ = count_number(data.MMSI)  # Count number of groups/ships
                    data.X_train, _, _ = data.normalize(data.Xraw_train) 
                    data.X_val, _, _ = data.normalize(data.Xraw_val)
                    data.X, _, _ = data.normalize(data.Xraw) 
                    clustering = Clustering() # perform clustering
                    if clustering_algorithm == 'kmeans':
                        print(" Running k-means clustering...")
                        idx, centroids = clustering.run_kmeans(X=data.X_val,K=K)
                    elif clustering_algorithm == 'DBSCAN':
                        print(" Running DBSCAN clustering...")
                        idx, K = clustering.run_DBSCAN(X=data.X_val,distance=distance)
                    # Run anomaly detection if needed 

                    # Compute quality measures
                
                # Slide the time window
                if window == 5: 
                    start = start + window
                    stop = stop + window
                else:
                    start = start + np.ceil(window/2)
                    stop = stop + np.ceil(window/2)

# Visualize

# Save the results
