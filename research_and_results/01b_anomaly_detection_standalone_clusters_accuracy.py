# ------------------ Examine the anomaly detection of AIS message reconstruction --------------------
# ---------------------------------- Standalone clusters -------------------------------------------- 
"""
Artificially damages random bit of a randomly chosen AIS messages and check the performace
of standalone clusters anomaly detection phase.
Requires: Gdansk.h5 file with the following datasets (created by data_Gdansk.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_mes33ages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (2805)
Creates 01b_anomaly_detection_standalone_clusters_Gdansk_.h5 file, with OK_vec with:
1. cluster assignment accuracy,
2. fields to correct classification recall,
3. fields to correct classification Jaccard score,
4. fields to correct classification Hamming score. 
"""
print("\n----------- AIS Anomaly detection - Standalone clusters accuracy part 2 --------- ")

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
from research import check_cluster_assignment 

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
filename = 'Gdansk.h5' # 'Gdansk', 'Baltic', 'Gibraltar'
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
        name='research_and_results/01b_anomaly_detection_standalone_clusters_accuracy_' + filename,
        mode='r'
        )
    OK_vec_1 = np.array(file.get('OK_vec_1'))
    baseline_1 = np.array(file.get('baseline_1'))
    OK_vec_2 = np.array(file.get('OK_vec_2'))
    baseline_2 = np.array(file.get('baseline_2'))
    file.close()
else:  # or run the computations on the original data
    file = h5py.File(name='data\\' + filename, mode='r')
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
    num_experiments = 100 # number of messages to randomly choose and corrupt
    num_metrics = 6 # number of quality metrics to compute
    bits = np.array(np.arange(8,60).tolist() + np.arange(61,137).tolist() + np.arange(143,145).tolist())
    for j in range(2): # iterate 2 times: for 1 and 2 bits corrupted
        corruption = Corruption(data.X,j+1)
        OK_vec2 = np.zeros((num_experiments, num_metrics))
        baseline2 = np.zeros((num_experiments, num_metrics-1))
        for i in range(num_experiments):  # For each of the randomly chosen AIS messages 
            stop = False
            while not stop:
                # corrupt its random bit
                Xraw_corr = copy.deepcopy(data.Xraw)
                MMSI_corr = copy.deepcopy(data.MMSI)
                message_decoded_corr = copy.deepcopy(data.message_decoded)
                bit_idx = np.random.permutation(bits)[0:j+1].tolist()
                message_bits_corr, message_idx = corruption.corrupt_bits(message_bits=data.message_bits, bit_idx=bit_idx[0])
                if j: # if two bits must be corrupted
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
                outliers = AnomalyDetection(data=data)
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
            OK_vec2[i,0] = check_cluster_assignment(idx, idx_corr, message_idx)
            
            # Check which fields are damaged
            field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 148])  # range of fields
            field = [sum(field_bits <= bit) for bit in np.sort(bit_idx)]
            #print(field)
            #print(outliers.outliers[message_idx][2])
            #print("\n")
            accuracies = calculate_ad_accuracy(field, outliers.outliers[message_idx][2])
            OK_vec2[i,1] = accuracies["recall"]
            OK_vec2[i,2] = accuracies["precision"]
            OK_vec2[i,3] = accuracies["f1"]
            OK_vec2[i,4] = accuracies["jaccard"]
            OK_vec2[i,5] = accuracies["hamming"]
            # Compare to baseline
            baseline_pred = []
            cwt_vec = []
            for f in outliers.fields:
                field_diff = np.array(outliers.compute_fields_diff(message_decoded_corr, idx_corr, message_idx, f))
                cwt_vec.append(field_diff[0])
                if field_diff[0]>0.5 and field_diff[1]>0.5: baseline_pred.append(f)
            accuracies = calculate_ad_accuracy(field, baseline_pred)
            baseline2[i,0] = accuracies["recall"]
            baseline2[i,1] = accuracies["precision"]
            baseline2[i,2] = accuracies["f1"]
            baseline2[i,3] = accuracies["jaccard"]
            baseline2[i,4] = accuracies["hamming"]

            # Plot
            if j==0 and field[0]==7: # when longitude field is damaged
                fields = [
                    "MMSI","Navigational\nstatus", "Rate of turns","Speed over\nground",
                    "Longitude", "Latitude","Course over\nground","True\nheading", 
                    "Special\nmanouvre\nindicator"]
                fig, ax = plt.subplots(ncols=2, nrows=2)
                # Plot corrupted fields message by message         
                messages = np.zeros_like(corruption.indices_corrupted)
                messages[message_idx] = 1
                messages = messages[idx_corr==outliers.outliers[message_idx][1]]
                messages = np.multiply(messages, message_decoded_corr[idx_corr==outliers.outliers[message_idx][1],field[0]])
                messages[messages == 0] = None
                ax[0,0].scatter(range(messages.shape[0]), message_decoded_corr[idx_corr==outliers.outliers[message_idx][1],field[0]])
                ax[0,0].scatter(range(messages.shape[0]), messages, color='r')
                ax[0,0].set_title("Damaged field: Longitude")
                ax[0,0].set_xlabel("Number of a consecutive message")
                ax[0,0].legend(["Original values", "Corrupted value"])
                # Plot the result of a wavelet transform
                with_ = message_decoded_corr[idx_corr==outliers.outliers[message_idx][1],7] # from all messages from the determined cluster
                without = np.delete(message_decoded_corr,message_idx,axis=0)  # and without the corrupted message
                without = without[np.delete(idx_corr,message_idx,axis=0) == outliers.outliers[message_idx][1],7]
                if len(with_)>1: with_ = abs(with_[1:len(with_)]-with_[0:len(with_)-1]) # compute the derivative
                scale = max(with_) # normalize
                if scale and sum(with_): with_ = with_/scale 
                with_cwt = abs(signal.cwt(with_, signal.morlet2, np.array([1,3])))
                if len(without)>1: without = abs(without[1:len(without)]-without[0:len(without)-1])
                if scale and sum(with_): without = without/scale
                without_cwt = abs(signal.cwt(without, signal.morlet2, np.array([1,3])))
                ax[0,1].plot(with_cwt[0,:])
                ax[0,1].plot(without_cwt[0,:])
                ax[0,1].set_title("Wavelet transform of a damaged field")
                ax[0,1].set_xlabel("Time dimension")
                ax[0,1].legend(["With corrupted value", "Without corrupted value"])
                # Plot relative differences
                gs = ax[1, 0].get_gridspec()
                for ax in ax[1, 0:2]: # remove the underlying axes
                    ax.remove()
                axbig = fig.add_subplot(gs[1, 0:2])
                axbig.bar(fields, cwt_vec, width=0.5)
                axbig.set_title("Relative difference in wavelet transform's max of each field")
                axbig.set_xlabel("Fields")
                axbig.set_xticks(range(len(fields)))
                axbig.set_xticklabels(fields, rotation=45)
                real = np.zeros_like(cwt_vec)
                f = outliers.fields.index(field[0])
                real[f] = 1
                axbig.text(f-0.075, cwt_vec[f]+0.01*cwt_vec[f], str(round(cwt_vec[f],3)), color='r')
                axbig.bar(fields, np.multiply(cwt_vec,real), width=0.5, color='r')
                fig.set_tight_layout(True)
                fig.show()
        
        if j==0:
            OK_vec_1 = np.mean(OK_vec2, axis=0)*100
            baseline_1 = np.mean(baseline2, axis=0)*100
        else:
            OK_vec_2 = np.mean(OK_vec2, axis=0)*100
            baseline_2 = np.mean(baseline2, axis=0)*100


# ----------- Part 2 - Visualization ----------
print(" Complete.")
print(" With 1 bit corrupted:")
print(" - Standalone cluster assigment accuracy: " + str(round(OK_vec_1[0],2)) + "%")
print(" - Feature indication recall: " + str(round(OK_vec_1[1],2)) + "%,"
+ " baseline: " + str(round(baseline_1[0],2)) + "%")
print(" - Feature indication precision: " + str(round(OK_vec_1[2],2)) + "%,"
+ " baseline: " + str(round(baseline_1[1],2)) + "%")
print(" - Feature indication f1 score: " + str(round(OK_vec_1[3],2)) + "%,"
+ " baseline: " + str(round(baseline_1[2],2)) + "%")
print(" - Feature indication Jaccard: " + str(round(OK_vec_1[4],2)) + "%,"
+ " baseline: " + str(round(baseline_1[3],2)) + "%")
print(" - Feature indication Hamming: " + str(round(OK_vec_1[5],2)) + "%,"
+ " baseline: " + str(round(baseline_1[4],2)) + "%")
print(" With 2 bits corrupted:")
print(" - Standalone cluster assigment accuracy: " + str(round(OK_vec_2[0],2)) + "%")
print(" - Feature indication recall: " + str(round(OK_vec_2[1],2)) + "%,"
+ " baseline: " + str(round(baseline_2[0],2)) + "%")
print(" - Feature indication precision: " + str(round(OK_vec_2[2],2)) + "%,"
+ " baseline: " + str(round(baseline_2[1],2)) + "%")
print(" - Feature indication f1 score: " + str(round(OK_vec_2[3],2)) + "%,"
+ " baseline: " + str(round(baseline_2[2],2)) + "%")
print(" - Feature indication Jaccard: " + str(round(OK_vec_2[4],2)) + "%,"
+ " baseline: " + str(round(baseline_2[3],2)) + "%")
print(" - Feature indication Hamming: " + str(round(OK_vec_2[5],2)) + "%,"
+ " baseline: " + str(round(baseline_2[4],2)) + "%")

if precomputed == '2':
    input("Press Enter to exit...")
else:
    # Save file
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/01b_anomaly_detection_standalone_clusters_accuracy_'+filename):
        os.remove('research_and_results/01b_anomaly_detection_standalone_clusters_accuracy_'+filename)
    File = h5py.File(
        'research_and_results/01b_anomaly_detection_standalone_clusters_accuracy_'+filename, 
        mode='a'
        )
    File.create_dataset('OK_vec_1', data=OK_vec_1)
    File.create_dataset('baseline_1', data=baseline_1)
    File.create_dataset('OK_vec_2', data=OK_vec_2)
    File.create_dataset('baseline_2', data=baseline_2)
    File.close()
