# ------------------ Examine the anomaly detection of AIS message reconstruction --------------------
"""
Artificially damages random bits of randomly chosen AIS messages and checks the performace
of anomaly detection phase.
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_messages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_messages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (2805)
Creates 02c_anomaly_detection_standalone_clusters_Gdansk_.h5 file, with OK_vec with:
1. message indication recall,
2. message indication precision,
3. message indication accuracy,
4. fields to correct classification recall,
5. fields to correct classification precision.
"""
print("\n----------- AIS Anomaly detection - overall accuracy part 2 --------- ")

# ----------- Part 0 - Initialization ----------
# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
plt.rcParams.update({'font.size': 16})
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
filename = 'Baltic.h5' # 'Gdansk', 'Baltic', 'Gibraltar'
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
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
        name='research_and_results/02c_anomaly_detection_overall_percentage_' + ad_algorithm + '_' + filename,
        mode='r'
        )
    OK_vec_1 = np.array(file.get('OK_vec_1'))
    OK_vec_2 = np.array(file.get('OK_vec_2'))
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

    # First clustering
    clustering = Clustering()
    if clustering_algorithm == 'kmeans':
        print(" Running k-means clustering...")
        idx, centroids = clustering.run_kmeans(X=data.X,K=K)
    elif clustering_algorithm == 'DBSCAN':
        print(" Running DBSCAN clustering...")
        idx, K = clustering.run_DBSCAN(X=data.X,distance=distance)
    print(" Complete.")


    # ----------- Part 1 - Computing accuracy ----------
    print(" Corrupting messages...") 
    # Artificially corrupt the dataset
    num_experiments = 10
    percentages = [5, 10]
    num_metrics = 5 # number of quality metrics to compute
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    for percentage in percentages:
        OK_vec2 = np.zeros((num_experiments, num_metrics))
        np.random.seed(1)
        for i in range(num_experiments):  # For each of the randomly chosen AIS messages
            Xraw_corr = copy.deepcopy(data.Xraw)
            MMSI_corr = copy.deepcopy(data.MMSI)
            message_decoded_corr = copy.deepcopy(data.message_decoded)
            corruption = Corruption(data.X,1)
            outliers = AnomalyDetection(data=data, ad_algorithm=ad_algorithm)
            fields = []
            messages = []
            num_messages = int(len(data.MMSI)*percentage/100)
            for n in range(num_messages):
                # Choose 0.05 or 0.1 of all messages and corrupt 2 their random bits
                field = np.random.choice(outliers.inside_fields, size=2, replace=False)
                fields.append(field)
                bit_idx = np.random.randint(field_bits[field[0]-1], field_bits[field[0]]-1)
                message_bits_corr, message_idx = corruption.corrupt_bits(message_bits=data.message_bits, bit_idx=bit_idx)
                new_bit_idx = np.random.randint(field_bits[field[1]-1], field_bits[field[1]]-1)
                message_bits_corr, message_idx = corruption.corrupt_bits(message_bits_corr, message_idx=message_idx, bit_idx=new_bit_idx)
                messages.append(message_idx)
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
            
            # Perform anomaly detection
            outliers.detect_standalone_clusters(
                idx=idx_corr,
                idx_vec=range(-1, np.max(idx_corr)+1),
                X=X_corr,
                message_decoded=message_decoded_corr,
                )
            outliers.detect_inside(
                idx=idx_corr,
                message_decoded=message_decoded_corr,
                timestamp=data.timestamp
                )
            # Compute accuracy
            pred = np.array([outliers.outliers[n][0] for n in range(len(outliers.outliers))], dtype=int)
            true = np.array(corruption.indices_corrupted, dtype=int)
            OK_vec2[i,0] = recall_score(true, pred)
            OK_vec2[i,1] = precision_score(true, pred)
            OK_vec2[i,2] = np.mean(true == pred)
            recall = []
            precision =[]
            for n in range(num_messages):
                accuracy = calculate_ad_accuracy(fields[n], outliers.outliers[messages[n]][2])
                recall.append(accuracy["recall"])
                precision.append(accuracy["precision"])
            OK_vec2[i,3] = np.mean(recall)
            OK_vec2[i,4] = np.mean(precision)
        
        if percentage==percentages[0]:
            OK_vec_1 = np.mean(OK_vec2, axis=0)*100
        else:
            OK_vec_2 = np.mean(OK_vec2, axis=0)*100


# ----------- Part 2 - Visualization ----------
print(" Complete.")
print(" With 0.05 messages corrupted:")
print(" - Message indication recall: " + str(round(OK_vec_1[0],2)) + "%")
print(" - Message indication precision: " + str(round(OK_vec_1[1],2)) + "%")
print(" - Message indication accuracy: " + str(round(OK_vec_1[2],2)) + "%")
print(" - Feature indication recall: " + str(round(OK_vec_1[3],2)) + "%")
print(" - Feature indication precision: " + str(round(OK_vec_1[4],2)) + "%")
print(" With 0.1 messages corrupted corrupted:")
print(" - Message indication recall: " + str(round(OK_vec_2[0],2)) + "%")
print(" - Message indication precision: " + str(round(OK_vec_2[1],2)) + "%")
print(" - Message indication accuracy: " + str(round(OK_vec_2[2],2)) + "%")
print(" - Feature indication recall: " + str(round(OK_vec_2[3],2)) + "%")
print(" - Feature indication precision: " + str(round(OK_vec_2[4],2)) + "%")

if precomputed == '2':
    input("Press Enter to exit...")
else:
    # Save file
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/02c_anomaly_detection_overall_percentage_'+ad_algorithm+'_'+filename):
        os.remove('research_and_results/02c_anomaly_detection_overall_percentage_'+ad_algorithm+'_'+filename)
    File = h5py.File(
        'research_and_results/02c_anomaly_detection_overall_percentage_'+ad_algorithm+'_'+filename, 
        mode='a'
        )
    File.create_dataset('OK_vec_1', data=OK_vec_1)
    File.create_dataset('OK_vec_2', data=OK_vec_2)
    File.close()
