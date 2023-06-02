# ------------------ Examine the anomaly detection of AIS message reconstruction --------------------
# ---------------------------------- Standalone clusters -------------------------------------------- 
"""
Artificially damages random bit of a randomly chosen AIS messages and compare the performace
of standalone clusters anomaly detection phase using Morlet and Ricker wavelet.
Requires: Gdansk/Baltic/Gibraltar.h5 files with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_messages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_messages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (805)
Creates 01c_anomaly_detection_standalone_clusters_ricker.h5 file, with OK_vec with classification f1 scores.
"""
print("\n---- AIS Anomaly detection - Standalone clusters accuracy part 3 - Morlet vs Ricker -------- ")

# ----------- Part 0 - Initialization ----------
# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from scipy import signal
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = ['rf', 'xgboost']
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
    file = h5py.File(
        name='research_and_results/01c_anomaly_detection_standalone_clusters_ricker.h5',
        mode='r'
        )
    OK_vec = np.array(file.get('OK_vec'))
    OK_vec_mor = np.array(file.get('OK_vec_mor'))
    file.close()

else:  # or run the computations on the original data
    # ----------- Part 1 - Computing accuracy ----------
    # Artificially corrupt the dataset
    num_experiments = 100 # number of messages to randomly choose and corrupt
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    OK_vec2 = np.zeros((num_experiments, len(ad_algorithm), len(filename)))
    OK_vec_mor = np.zeros((len(ad_algorithm), len(filename)))
   
    for k in range(len(filename)): # iterate for each dataset: 0-Gdansk, 1-Baltic, 2-Gibraltar
        print(" Import dataset...") 
        # Import the dataset
        file = h5py.File(name='data/' + filename[k], mode='r')
        data = Data(file)
        data.split(train_percentage=50, val_percentage=25)
        file.close()
        # Preprocess data
        print(" Preprocessing data... ")
        K, _ = count_number(data.MMSI)  # Count number of groups/ships
        data.X_train, _, _ = data.normalize(data.Xraw_train)
        data.X_val, _, _ = data.normalize(data.Xraw_val)
        data.X, _, _ = data.normalize(data.Xraw)  
        # First clustering
        clustering = Clustering()
        if clustering_algorithm == 'kmeans':
            print(" Running k-means clustering...")
            idx, centroids = clustering.run_kmeans(X=data.X,K=K)
        elif clustering_algorithm == 'DBSCAN':
            print(" Running DBSCAN clustering...")
            idx, K = clustering.run_DBSCAN(X=data.X,distance=distance)
        print(" Complete.")
        
        for j in range(len(ad_algorithm)): # iterate for rf (0) and xgboost (1)
            print(" Corrupting messages...") 
            corruption = Corruption(data.X,1)
            for i in range(num_experiments):  # For each of the randomly chosen AIS messages 
                stop = False
                while not stop:
                    # corrupt its random bit
                    Xraw_corr = copy.deepcopy(data.Xraw)
                    MMSI_corr = copy.deepcopy(data.MMSI)
                    message_decoded_corr = copy.deepcopy(data.message_decoded)
                    bit_idx = np.random.permutation(bits)[0:2].tolist()
                    message_bits_corr, message_idx = corruption.corrupt_bits(message_bits=data.message_bits, bit_idx=bit_idx[0])
                    message_bits_corr, message_idx = corruption.corrupt_bits(message_bits_corr, message_idx=message_idx, bit_idx=bit_idx[1])
                    # put it back to the dataset
                    X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
                    Xraw_corr[message_idx,:] = X_0
                    MMSI_corr[message_idx] = MMSI_0
                    message_decoded_corr[message_idx,:] = message_decoded_0
                    X_corr, _, _ = data.normalize(Xraw_corr)
                    # cluster again to find new cluster assignment
                    K_corr, MMSI_vec_corr = count_number(MMSI_corr)
                    if clustering_algorithm == 'kmeans':
                        idx_corr, _ = clustering.run_kmeans(X=X_corr,K=K_corr)
                    elif clustering_algorithm == 'DBSCAN':
                        idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr,distance=distance)
                    # Check if the cluster is a standalone cluster
                    outliers = AnomalyDetection(data=data, ad_algorithm=ad_algorithm[j], wavelet='ricker')
                    outliers.detect_standalone_clusters(
                        idx=idx_corr,
                        idx_vec=range(-1, np.max(idx_corr)+1),
                        X=X_corr,
                        message_decoded=message_decoded_corr
                        )
                    # if so, stop searching
                    stop = outliers.outliers[message_idx][0]
                    # if not, allow this message to be chosen again
                    corruption.indices_corrupted[message_idx] = outliers.outliers[message_idx][0]

                # and check which cluster that message should be assigned to
                idx_corr[message_idx] = outliers.outliers[message_idx][1]
                
                # Check which fields are damaged
                field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
                field = [sum(field_bits <= bit) for bit in np.sort(bit_idx)]
                accuracies = calculate_ad_accuracy(field, outliers.outliers[message_idx][2])
                OK_vec2[i,j,k] = accuracies["f1"]
            
            # Get the result with a Morlet wavelet
            file2 = h5py.File(
                name='research_and_results/01b_anomaly_detection_standalone_clusters_'+ad_algorithm[j]+'_'+filename[k],
                mode='r' )
            OK_vec_2 = np.array(file2.get('OK_vec_2'))
            file2.close()
            OK_vec_mor[j,k] = OK_vec_2[3]

    OK_vec = np.mean(OK_vec2, axis=0)*100

# ----------- Part 2 - Visualization ----------
print(" Complete.")
fig, ax = plt.subplots(ncols=len(ad_algorithm), sharey=True)
titles=[]
for x in range(len(filename)): titles.append(str(x+1))
x = np.arange(len(filename))
ax[0].bar(x-0.2,OK_vec_mor[0,:], width=0.3)
ax[0].bar(x+0.2,OK_vec[0,:], width=0.3)
ax[0].set_title("Random Forest")
ax[0].set_xlabel("Number of a dataset")
ax[0].set_ylabel("Average F1 score [%]")
ax[0].set_xticks(x)
ax[0].set_xticklabels(titles)
ax[0].legend(["Morlet wavelet", "Ricker wavelet"])
ax[1].bar(x-0.2,OK_vec_mor[1,:], width=0.3)
ax[1].bar(x+0.2,OK_vec[1,:], width=0.3)
ax[1].set_title("XGBoost")
ax[1].set_xlabel("Number of a dataset")
ax[1].set_ylabel("Average F1 score [%]")
ax[1].set_xticks(x)
ax[1].set_xticklabels(titles)
ax[1].legend(["Morlet wavelet", "Ricker wavelet"])
fig.show()

fig2, ax2 = plt.subplots(ncols=2, sharey=True)
ax2[0].plot(signal.morlet2(100,5))
ax2[0].set_title("Sample Morlet wavelet")
ax2[0].set_xlabel("Sample")
ax2[0].set_ylabel("Value")
ax2[1].plot(signal.ricker(100,5))
ax2[1].set_title("Sample Ricker wavelet")
ax2[1].set_xlabel("Sample")
ax2[1].set_ylabel("Value")
fig2.show()

if precomputed == '2':
    input("Press Enter to exit...")
else:
    # Save file
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/01c_anomaly_detection_standalone_clusters_ricker.h5'):
        os.remove('research_and_results/01c_anomaly_detection_standalone_clusters_ricker.h5')
    File = h5py.File(
        'research_and_results/01c_anomaly_detection_standalone_clusters_ricker.h5', 
        mode='a'
        )
    File.create_dataset('OK_vec', data=OK_vec)
    File.create_dataset('OK_vec_mor', data=OK_vec_mor)
    File.close()
