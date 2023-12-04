"""
Analyses the datasets by damaging each bit and checks the performace of all stages of AIS message reconstruction. \n
Requires: Gdansk.h5 file with the following datasets (created by data_Gdansk.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages (805), num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages (805), num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages (805), num_features (115)),
 - MMSI - list of MMSI identifiers from each AIS message, len=num_messages (805). \n
Creates 02_bitwise_.h5 file, with OK_vec with average percentage of: 
 - if clustering: correctly clustered messages, messages forming 1-element clusters, correctly assigned at the end,
 - if anomaly detection: recall of detecting messages and fields, precision of detecting fields.
"""
print("\n----------- The impact of the position of damagedd bit on AIS message reconstruction  --------- ")

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

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
stage = 'ad' # 'clustering', 'ad' or 'prediction'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'rf' # 'rf' or 'xgboost'
# --------------------------------------------------------------------------------

# Create a list of meaningful bits to examine
field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
mask = []
if stage=='clustering':
    bits = list(range(145))  
    bits.append(148)
else:
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
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
    if stage == 'clustering':
        file = h5py.File(name='research_and_results/02_bitwise_'+clustering_algorithm+'.h5', mode='r')
    elif stage == 'ad':
        file = h5py.File(name='research_and_results/02_bitwise_'+ad_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations on the original data
    file = h5py.File(name='data\Gdansk.h5', mode='r')
    data = Data(file)
    data.split(train_percentage=50, val_percentage=25)
    file.close()

   # Preprocess data
    print(" Preprocessing data... ")
    data.X_train, _, _ = data.standarize(data.Xraw_train)
    data.X_val, _, _ = data.standarize(data.Xraw_val)
    data.X, _, _ = data.standarize(data.Xraw)  

    # First clustering
    clustering = Clustering()
    if clustering_algorithm == 'kmeans':
        K, _ = count_number(data.MMSI)  # Count number of groups/ships
        idx, centroids = clustering.run_kmeans(X=data.X,K=K)
    elif clustering_algorithm == 'DBSCAN':
        idx, K = clustering.run_DBSCAN(X=data.X,distance=distance)
    print(" Complete.")

    print(" Damaging bit by bit...") 
    corruption = Corruption(data.X) 
    OK_vec = np.zeros((3,146))
    for bit in bits:  # For each of AIS message bits
        np.random.seed(250)  # make numpy choose the same messages all the time
        corruption.reset()
        OK_vec2 = np.zeros((3,20))  # choose 20 messages
        if stage=="ad": field = sum(field_bits <= bit)  # check to which field the damaged bit belong to
        for j in range(20):  # for each chosen message:
            # damage its bit
            X_corr = copy.deepcopy(data.Xraw)
            MMSI_corr = copy.deepcopy(data.MMSI)
            message_decoded_corr = copy.deepcopy(data.message_decoded)
            message_bits_corr, message_idx = corruption.corrupt_bits(data.message_bits, bit)
            # put it back to the dataset
            X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
            X_corr[message_idx,:] = X_0
            MMSI_corr[message_idx] = MMSI_0
            message_decoded_corr[message_idx,:] = message_decoded_0
            X_corr, _, _ = data.standarize(X_corr)
            # cluster again to find new cluster assignment
            if clustering_algorithm == 'kmeans':
                K_corr, _ = count_number(MMSI_corr)
                idx_corr, _ = clustering.run_kmeans(X=X_corr,K=K_corr)
            elif clustering_algorithm == 'DBSCAN':
                idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr,distance=distance)
            # search for 1-element clusters
            ad = AnomalyDetection(ad_algorithm=ad_algorithm)
            ad.detect_in_1element_clusters(
                idx=idx_corr,
                idx_vec=range(-1, np.max(idx_corr)+1),
                X=X_corr,
                message_decoded=message_decoded_corr)
            
            if stage == 'clustering': # metrics after clustering
                OK_vec2[0,j] = check_cluster_assignment(idx=idx, idx_corr=idx_corr, message_idx=message_idx)
                OK_vec2[1,j] = ad.outliers[message_idx][0]
                if OK_vec2[1,j]:
                    idx_corr[message_idx] = ad.outliers[message_idx][1]
                    OK_vec2[2,j] = check_cluster_assignment(idx, idx_corr, message_idx)
                else: OK_vec2[2,j] = OK_vec2[0,j]

            else: # perform further anomaly detection
                ad.detect_in_multielement_clusters(
                idx=idx_corr,
                message_decoded=message_decoded_corr,
                timestamp=data.timestamp)
            
                if stage == 'ad': # metrics after anomaly detection
                    OK_vec2[0,j] = ad.outliers[message_idx][0] # save message indication recall
                    accuracies = calculate_ad_accuracy([field], ad.outliers[message_idx][2])
                    OK_vec2[1,j] = accuracies["recall"] # save field indication recall
                    OK_vec2[2,j] = accuracies["precision"] # save field indication precision

        if bit < 145: OK_vec[:,bit] = np.mean(OK_vec2, axis=1)*100
        else: OK_vec[:,145] = np.mean(OK_vec2, axis=1)*100


# Visualization
print(" Complete.")
if stage == 'clustering': 
    titles = {
        '0':"Correctly assigned messages [%]", 
        '1':"Messages forming standalone clusters [%]", 
        '2':"Correctly assigned messages after correction [%]"}
    print(" Correctly assigned messages: " + str(round(np.mean(OK_vec[0,:]),2)) + "%")
    print(" Messages forming standalone clusters: " + str(round(np.mean(OK_vec[1,:]),2)) + "%")
    print(" Correctly assigned messages after correction: " + str(round(np.mean(OK_vec[2,:]),2)) + "%")
elif stage == 'ad': 
    titles = {
    '0':"Correctly detected damaged messages - recall [%]", 
    '1':"Correctly detected damaged fields - recall [%]", 
    '2':"Correctly detected damaged fields - precision [%]"}
    print(" Message indication recall: " + str(round(np.mean(OK_vec[0,mask==1]),2)) + "%")
    print(" Feature indication recall: " + str(round(np.mean(OK_vec[1,mask==1]),2)) + "%")
    print(" Feature indication precision: " + str(round(np.mean(OK_vec[2,mask==1]),2)) + "%")

bits = list(range(145))  # create a list of meaningful bits to visualize
bits.append(148)
fig, ax = plt.subplots(OK_vec.shape[0], sharex=True, sharey=True)
for i in range(OK_vec.shape[0]):
    ax[i].set_title(titles[str(i)])  # get titles from the dictionary
    # Plot each meesage fields with a different color - other bits are 0s
    ax[i].bar(bits, np.concatenate((OK_vec[i,0:6], np.zeros((140))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((6)), OK_vec[i,6:8], np.zeros((138))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((8)), OK_vec[i,8:38], np.zeros((108))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((38)), OK_vec[i,38:42], np.zeros((104))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((42)), OK_vec[i,42:50], np.zeros((96))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((50)), OK_vec[i,50:60], np.zeros((86))), axis=0))
    temp = np.zeros(146)
    temp[60] = OK_vec[i,60]
    ax[i].bar(bits, temp)
    ax[i].bar(bits, np.concatenate((np.zeros((61)), OK_vec[i,61:89], np.zeros((57))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((89)), OK_vec[i,89:116], np.zeros((30))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((116)), OK_vec[i,116:128], np.zeros((18))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((128)), OK_vec[i,128:137], np.zeros((9))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((137)), OK_vec[i,137:143], np.zeros((3))), axis=0))
    ax[i].bar(bits, np.concatenate((np.zeros((143)), OK_vec[i,143:145], np.zeros((1))), axis=0))
    temp = np.zeros(146)
    temp[145] = OK_vec[i,145]
    ax[i].bar(bits, temp)
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0, box.width * 0.85, box.height])
ax[i].set_xlabel("Index of a damaged bit")
fig.legend([
            "Message type","Repeat indicator","MMSI","Navigational status", 
            "Rate of turns","Speed over ground","Position accuracy","Longitude", 
            "Latitude","Course over ground","True heading","Time stamp", 
            "Special manouvre indicator", "RAIM-flag"], loc=7)
fig.show()


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if stage == 'clustering':
        if os.path.exists('research_and_results/02_bitwise_'+clustering_algorithm+'.h5'):
            os.remove('research_and_results/02_bitwise_'+clustering_algorithm+'.h5')
        file = h5py.File('research_and_results/02_bitwise_'+clustering_algorithm+'.h5', mode='a')
    elif stage == 'ad':
        if os.path.exists('research_and_results/02_bitwise_'+ad_algorithm+'.h5'):
            os.remove('research_and_results/02_bitwise_'+ad_algorithm+'.h5')
        file = h5py.File('research_and_results/02_bitwise_'+ad_algorithm+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()