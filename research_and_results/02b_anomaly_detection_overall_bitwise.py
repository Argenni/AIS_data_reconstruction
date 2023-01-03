# ------------------ Examine the anomaly detection of AIS message reconstruction --------------------
# ---------------------------------- Inside clusters -------------------------------------------- 
"""
Artificially damages each bit of a randomly chosen AIS message and checks the performace
of anomaly detection phase.
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_messages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_messages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (2805)
Creates 02b_anomaly_detection_overall_bitwise_Gdansk_.h5 file, with OK_vec with:
1. message indication recall,
2. fields to correct classification recall,
3. fields to correct classification precision,
4. fields to correct classification f1 score. 
"""
print("\n----------- AIS Anomaly detection - overall accuracy part 1 --------- ")

# ----------- Part 0 - Initialization ----------
# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption
from research import visualize_corrupted_bits

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
filename = 'Gdansk.h5' # 'Gdansk', 'Baltic', 'Gibraltar'
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
# --------------------------------------------------------------------------------
bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist())
mask = []
for bit in range(146):
    if bit in bits: mask.append(1)
    else: mask.append(0)
mask = np.array(mask)

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
        name='research_and_results/02b_anomaly_detection_overall_bitwise_' + filename,
        mode='r'
        )
    OK_vec = np.array(file.get('OK_vec'))
    file.close()
else:  # or run the computations on the original data
    file = h5py.File(name='data/' + filename, mode='r')
    data = Data(file)
    data.split(train_percentage=50, val_percentage=25)
    file.close()

    # Preprocess data
    print(" Preprocessing data... ")
    K, _ = count_number(data.MMSI)  # Count number of groups/ships
    data.X_train, _, _ = data.normalize(data.Xraw_train)
    data.X_val, _, _ = data.normalize(data.Xraw_val)
    data.X, _, _ = data.normalize(data.Xraw)  


    # ----------- Part 1 - Computing accuracy ----------
    print(" Corrupting bit by bit...") 
    # Artificially corrupt the dataset
    corruption = Corruption(data.X,1) 
    OK_vec = np.zeros((4,146))
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 148])  # range of fields
    for bit in bits:  # For each of AIS message bits
        np.random.seed(250)  # make numpy choose the same messages all the time
        corruption.reset()
        OK_vec2 = np.zeros((4,20))  # choose 20 messages
        mask[bit] = 1
        field = sum(field_bits <= bit)  # check to which field the corrupted bit belong to
        for j in range(20):  # for each chosen message:
            # corrupt its ith bit
            X_corr = copy.deepcopy(data.Xraw)
            MMSI_corr = copy.deepcopy(data.MMSI)
            message_decoded_corr = copy.deepcopy(data.message_decoded)
            message_bits_corr, message_idx = corruption.corrupt_bits(data.message_bits, bit)
            # put it back to the dataset
            X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
            X_corr[message_idx,:] = X_0
            MMSI_corr[message_idx] = MMSI_0
            message_decoded_corr[message_idx,:] = message_decoded_0
            X_corr, _, _ = data.normalize(X_corr)
            # cluster
            clustering = Clustering()
            if clustering_algorithm == 'kmeans':
                K_corr, _ = count_number(MMSI_corr)
                idx_corr, _ = clustering.run_kmeans(X=X_corr,K=K_corr)
            elif clustering_algorithm == 'DBSCAN':
                idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr,distance=distance)

            outliers = AnomalyDetection(data=data)
            outliers.detect_standalone_clusters(
                idx=idx_corr,
                idx_vec=range(-1, np.max(idx_corr)+1),
                X=X_corr,
                message_decoded=message_decoded_corr
                )
            outliers.detect_inside(
                idx=idx_corr,
                message_decoded=message_decoded_corr,
                timestamp=data.timestamp
                )
            OK_vec2[0,j] = outliers.outliers[message_idx][0] # save message indication recall
            accuracies = calculate_ad_accuracy([field], outliers.outliers[message_idx][2])
            OK_vec2[1,j] = accuracies["recall"] # save field indication recall
            OK_vec2[2,j] = accuracies["precision"] # save field indication precision
            OK_vec2[3,j] = accuracies["f1"] # save field indication f1 score
        OK_vec[:,bit] = np.mean(OK_vec2, axis=1)*100


# ----------- Part 2 - Visualization ----------
print(" Complete.")
titles = {
    '0':"Correctly detected damaged messages - recall [%]", 
    '1':"Correctly detected damaged fields - recall [%]", 
    '2':"Correctly detected damaged fields - precision [%]",
    '3':"Correctly detected damaged fields - f1 score [%]"
    }
visualize_corrupted_bits(OK_vec[0:4,:], titles)
print(" Message indication recall: " + str(round(np.mean(OK_vec[0,mask==1]),2)) + "%")
print(" Feature indication recall: " + str(round(np.mean(OK_vec[1,mask==1]),2)) + "%")
print(" Feature indication precision: " + str(round(np.mean(OK_vec[2,mask==1]),2)) + "%")
print(" Feature indication f1 score: " + str(round(np.mean(OK_vec[3,mask==1]),2)) + "%")

if precomputed == '2':
    input("Press Enter to exit...")
else:
    # Save file
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/02b_anomaly_detection_overall_bitwise_Gdansk.h5'):
        os.remove('research_and_results/02b_anomaly_detection_overall_bitwise_Gdansk.h5')
    File = h5py.File(
        'research_and_results/02b_anomaly_detection_overall_bitwise_Gdansk.h5', 
        mode='x'
        )
    File.create_dataset('OK_vec', data=OK_vec)
    File.close()