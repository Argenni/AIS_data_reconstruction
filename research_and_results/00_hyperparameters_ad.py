"""
Artificially damages random bit of a randomly chosen AIS messages and compare the performace
of 1-element-cluster anomaly detection stage on validation set using Morlet and Ricker wavelet. \n
Requires: Gdansk/Baltic/Gibraltar.h5 files with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifier from each AIS message, len=num_messages.
Creates 00_hyperparameters_ad_wavelet.h5 file, with OK_vec with classification f1 scores
    (for each wavelet, algorithm and dataset).
"""
print("\n---- Hyperparameter tuning - anomaly detection stage - wavelet -------- ")

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
from utils.anomaly_detection import AnomalyDetection, calculate_ad_metrics
from utils.miscellaneous import count_number, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = ['rf', 'xgboost']
wavelet = ['morlet', 'ricker']
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
        name='research_and_results/00_hyperparameters_ad_wavelet.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations on the original data
    num_experiments = 100 # number of messages to randomly choose and damage
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    OK_vec2 = np.zeros((num_experiments, len(wavelet), len(ad_algorithm), len(filename)))
   
    for file_num in range(len(filename)): # iterate for each dataset: 0-Gdansk, 1-Baltic, 2-Gibraltar
        # Import the dataset
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data = Data(file)
        data.split(train_percentage=50, val_percentage=25)
        file.close()
        # Preprocess data
        K, _ = count_number(data.MMSI_val)  # Count number of groups/ships
        data.X_train, _, _ = data.standardize(data.Xraw_train)
        data.X_val, _, _ = data.standardize(data.Xraw_val)
        # First clustering
        clustering = Clustering()
        if clustering_algorithm == 'kmeans': idx, centroids = clustering.run_kmeans(X=data.X_val,K=K)
        elif clustering_algorithm == 'DBSCAN': idx, K = clustering.run_DBSCAN(X=data.X_val,distance=distance)
        
        for wav_num in range(len(wavelet)): # iterate for morlet (0) and ricker (1)
            for alg_num in range(len(ad_algorithm)): # iterate for rf (0) and xgboost (1)
                print(" Analysing: dataset " + str(file_num+1)+"., " + wavelet[wav_num]+", " + ad_algorithm[alg_num]+"...") 
                corruption = Corruption(data.X_val)
                for i in range(num_experiments):  # For each of the randomly chosen AIS messages 
                    stop = False
                    while not stop:
                        # damage its random bit
                        Xraw_corr = copy.deepcopy(data.Xraw_val)
                        MMSI_corr = copy.deepcopy(data.MMSI_val)
                        message_decoded_corr = copy.deepcopy(data.message_decoded_val)
                        bit_idx = np.random.permutation(bits)[0:2].tolist()
                        message_bits_corr, message_idx = corruption.corrupt_bits(message_bits=data.message_bits_val, bit_idx=bit_idx[0])
                        message_bits_corr, message_idx = corruption.corrupt_bits(message_bits_corr, message_idx=message_idx, bit_idx=bit_idx[1])
                        # put it back to the dataset
                        X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
                        Xraw_corr[message_idx,:] = X_0
                        MMSI_corr[message_idx] = MMSI_0
                        message_decoded_corr[message_idx,:] = message_decoded_0
                        X_corr, _, _ = data.standardize(Xraw_corr)
                        # cluster again to find new cluster assignment
                        K_corr, MMSI_vec_corr = count_number(MMSI_corr)
                        if clustering_algorithm == 'kmeans':
                            idx_corr, _ = clustering.run_kmeans(X=X_corr,K=K_corr)
                        elif clustering_algorithm == 'DBSCAN':
                            idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr,distance=distance)
                        # Check if the cluster is a 1-element cluster
                        ad = AnomalyDetection( 
                            ad_algorithm=ad_algorithm[alg_num], 
                            wavelet=wavelet[wav_num])
                        ad.detect_in_1element_clusters(
                            idx=idx_corr,
                            idx_vec=range(-1, np.max(idx_corr)+1),
                            X=X_corr,
                            message_decoded=message_decoded_corr)
                        # if so, stop searching
                        stop = ad.outliers[message_idx][0]
                        # if not, allow this message to be chosen again
                        corruption.indices_corrupted[message_idx] = ad.outliers[message_idx][0]
                    # and check which cluster that message should be assigned to
                    idx_corr[message_idx] = ad.outliers[message_idx][1]               
                    # Check which fields are damaged
                    field = list(set([sum(field_bits <= bit) for bit in np.sort(bit_idx)]))
                    ad_metrics = calculate_ad_metrics(field, ad.outliers[message_idx][2])
                    OK_vec2[i,wav_num,alg_num,file_num] = ad_metrics["f1"]
    OK_vec = np.mean(OK_vec2, axis=0)*100


# Visualization
print(" Complete.")
fig, ax = plt.subplots(ncols=len(ad_algorithm), sharey=True)
titles=[]
for x in range(len(filename)): titles.append(str(x+1))
x = np.arange(len(filename))
ax[0].bar(x-0.2,OK_vec[0,0,:], width=0.3)
ax[0].bar(x+0.2,OK_vec[1,0,:], width=0.3)
ax[0].set_title("Random Forest")
ax[0].set_xlabel("Number of a dataset")
ax[0].set_ylabel("Average F1 score [%]")
ax[0].set_xticks(x)
ax[0].set_xticklabels(titles)
ax[0].legend(["Morlet wavelet", "Ricker wavelet"])
ax[1].bar(x-0.2,OK_vec[0,1,:], width=0.3)
ax[1].bar(x+0.2,OK_vec[1,1,:], width=0.3)
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


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/00_hyperparameters_ad_wavelet.h5'):
        os.remove('research_and_results/00_hyperparameters_ad_wavelet.h5')
    File = h5py.File('research_and_results/00_hyperparameters_ad_wavelet.h5', mode='a')
    File.create_dataset('OK_vec', data=OK_vec)
    File.close()
