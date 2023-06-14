# ------------------ Examine the anomaly detection of AIS message reconstruction --------------------
# ---------------------------------- Standalone clusters -------------------------------------------- 
"""
Artificially damages each bit of a randomly chosen 20 AIS messages and check if the message is still
 assigned to the same cluster as before, if not, if it forms a standalone cluster (conducts then 
 a procedure to find a right cluster and damaged field to that message)
Requires: Gdansk.h5 file with the following datasets (created by data_Gdansk.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_messages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_messages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (2805)
Creates 01_anomaly_detection_standalone_clusters_Gdansk_.h5 file with OK_vec and baseline, shape = (4, 146), where:
 1. row - percentage of messages assigned to the same cluster, 2. row - percentage of standalone clusters,
 3. row - percentage of standalone clusters assigned to right cluster after correction, 
 4. row - percentage of correctly pointed damaged fields (recall) - also in baseline,
 5. row - mean Jaccard score of field multilabel classification - also in baseline,
 6. row - mean Hamming score of field multilabel classification - also in baseline.
"""
print("\n----------- AIS Anomaly detection - Standalone clusters accuracy --------- ")

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
from utils.clustering import Clustering, check_cluster_assignment
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption  
from research import visualize_corrupted_bits, check_cluster_assignment

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
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
        name='research_and_results/01_anomaly_detection_standalone_clusters_accuracy_Gdansk.h5',
        mode='r'
        )
    OK_vec = np.array(file.get('OK_vec'))
    baseline = np.array(file.get('baseline'))
    file.close()
else:  # or run the computations on the original data
    file = h5py.File(name='data\Gdansk.h5', mode='r')
    data = Data(file)
    data.split(train_percentage=50, val_percentage=25)
    file.close()

    # Preprocess data
    print(" Preprocessing data... ")
    data.X_train, _, _ = data.normalize(data.Xraw_train)
    data.X_val, _, _ = data.normalize(data.Xraw_val)
    data.X, _, _ = data.normalize(data.Xraw)  

    # First clustering
    clustering = Clustering()
    if clustering_algorithm == 'kmeans':
        print(" Running k-means clustering...")
        K, _ = count_number(data.MMSI)  # Count number of groups/ships
        idx, centroids = clustering.run_kmeans(X=data.X,K=K)
    elif clustering_algorithm == 'DBSCAN':
        print(" Running DBSCAN clustering...")
        idx, K = clustering.run_DBSCAN(X=data.X,distance=distance)
    print(" Complete.")


    # ----------- Part 1 - Computing accuracy ----------
    print(" Corrupting bit by bit...") 
    # Artificially corrupt the dataset
    corruption = Corruption(data.X,2) 
    OK_vec = np.zeros((8,146))
    baseline = np.zeros((5,146))
    bits = list(range(145))  # create a list of meaningful bits to examine
    bits.append(148)
    for bit in bits:  # For each of AIS message bits
        np.random.seed(250)  # make numpy choose the same messages all the time
        corruption.reset()
        OK_vec2 = np.zeros((8,20))  # choose 20 messages
        baseline2 = np.zeros((5,20))
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
            # cluster again to find new cluster assignment
            if clustering_algorithm == 'kmeans':
                K_corr, _ = count_number(MMSI_corr)
                idx_corr, _ = clustering.run_kmeans(X=X_corr,K=K_corr)
            elif clustering_algorithm == 'DBSCAN':
                idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr,distance=distance)
            # check if the cluster is the same despite corruption
            OK_vec2[0,j] = check_cluster_assignment(
                idx=idx, 
                idx_corr=idx_corr, 
                message_idx=message_idx
                )

            # Check if the cluster is a standalone cluster
            outliers = AnomalyDetection(data=data)
            outliers.detect_standalone_clusters(
                idx=idx_corr,
                idx_vec=range(-1, np.max(idx_corr)+1),
                X=X_corr,
                message_decoded=message_decoded_corr
                )
            OK_vec2[1,j] = outliers.outliers[message_idx][0]  

            # If so, check which cluster that message should be assigned to
            if OK_vec2[1,j]:
                idx_corr[message_idx] = outliers.outliers[message_idx][1]
                OK_vec2[2,j] = check_cluster_assignment(idx, idx_corr, message_idx)
                
                # Check which field is damaged
                field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 148])  # range of fields
                field = sum(field_bits <= bit)  # check to which field the corrupted bit belong to
                accuracies = calculate_ad_accuracy([field], outliers.outliers[message_idx][2])
                OK_vec2[3,j] = accuracies["recall"]
                OK_vec2[4,j] = accuracies["precision"]
                OK_vec2[5,j] = accuracies["f1"]
                OK_vec2[6,j] = accuracies["jaccard"]
                OK_vec2[7,j] = accuracies["hamming"]
                #Compare to baseline
                baseline_pred = []
                cwt_vec = []
                for f in outliers.fields:
                    field_diff = np.array(outliers.compute_fields_diff(message_decoded_corr, idx_corr, message_idx, f))
                    cwt_vec.append(field_diff[0])
                    if field_diff[0]>0.5 and field_diff[1]>0.5: baseline_pred.append(f)
                accuracies = calculate_ad_accuracy([field], baseline_pred)
                baseline2[0,j] = accuracies["recall"]
                baseline2[1,j] = accuracies["precision"]
                baseline2[2,j] = accuracies["f1"]
                baseline2[3,j] = accuracies["jaccard"]
                baseline2[4,j] = accuracies["hamming"]
            else:
                OK_vec2[2,j] = OK_vec2[0,j]
        if bit < 145: 
            OK_vec[:,bit] = np.mean(OK_vec2, axis=1)*100
            baseline[:,bit] = np.mean(baseline2, axis=1)*100
        else: 
            OK_vec[:,145] = np.mean(OK_vec2, axis=1)*100
            baseline[:,145] = np.mean(baseline2, axis=1)*100


# ----------- Part 2 - Visualization ----------
print(" Complete.")
titles = {
    '0':"Correctly assigned messages [%]", 
    '1':"Messages forming standalone clusters [%]", 
    '2':"Correctly assigned messages after correction [%]"
    }
visualize_corrupted_bits(OK_vec[0:3,:], titles)
print(" Final cluster assignment accuracy: " + str(round(np.mean(OK_vec[2,:]),2)) + "%")
print(" Standalone cluster assigment accuracy: " + str(round(np.mean(OK_vec[2,OK_vec[1,:]>0]),2)) + "%")
print(" Feature indication recall: " + str(round(np.mean(np.divide(OK_vec[3,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%,"
+ " baseline: " + str(round(np.mean(np.divide(baseline[0,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%")  # only for standalone clusters
print(" Feature indication precision: " + str(round(np.mean(np.divide(OK_vec[4,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%,"
+ " baseline: " + str(round(np.mean(np.divide(baseline[1,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%")
print(" Feature indication f1 score: " + str(round(np.mean(np.divide(OK_vec[5,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%,"
+ " baseline: " + str(round(np.mean(np.divide(baseline[2,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%")
print(" Feature indication Jaccard: " + str(round(np.mean(np.divide(OK_vec[6,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%,"
+ " baseline: " + str(round(np.mean(np.divide(baseline[3,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%")
print(" Feature indication Hamming: " + str(round(np.mean(np.divide(OK_vec[7,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%,"
+ " baseline: " + str(round(np.mean(np.divide(baseline[4,OK_vec[1,:]>0],OK_vec[1,OK_vec[1,:]>0]))*100,2)) + "%")
fig, ax = plt.subplots()
ax.bar(range(146), OK_vec[1,:])
ax.plot(OK_vec[4,:], color='r')
ax.plot(baseline[1,:], color='r', linestyle='dashed')
ax.plot(OK_vec[3,:], color='k')
ax.plot(baseline[0,:], color='k', linestyle='dashed')
ax.set_title("Feature indication accuracy")
ax.set_xlabel("Index of a damaged bit")
ax.set_ylabel("Feature indication quality measure [%]")
ax.legend(["Precision", "Precision - baseline", "Recall", "Recall - baseline"])
fig.show()

if precomputed == '2':
    input("Press Enter to exit...")
else:
    # Save file
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/01_anomaly_detection_standalone_clusters_accuracy_Gdansk.h5'):
        os.remove('research_and_results/01_anomaly_detection_standalone_clusters_accuracy_Gdansk.h5')
    File = h5py.File(
        'research_and_results/01_anomaly_detection_standalone_clusters_accuracy_Gdansk.h5', 
        mode='x'
        )
    File.create_dataset('OK_vec', data=OK_vec)
    File.create_dataset('baseline', data=baseline)
    File.close()
