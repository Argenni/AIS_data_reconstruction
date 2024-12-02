"""
Analyses the datasets by damaging each bit and checks the performace of all stages of AIS message reconstruction. \n
Requires: Gdansk.h5 file with the following datasets (created by data_Gdansk.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages (805), num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages (805), num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages (805), num_features (115)),
 - MMSI - list of MMSI identifiers from each AIS message, len=num_messages (805). \n
Creates 02_bitwise_.h5 file, with OK_vec with average: 
 - if clustering: percentage of correctly clustered messages, messages forming 1-element clusters, correctly assigned at the end,
 - if anomaly detection: recall of detecting messages and fields, precision of detecting fields,
 - if prediction: SMAE of damaged dataset, pure prediction and prediction after anomaly detection.
"""
print("\n----------- The impact of the position of damaged bit on AIS message reconstruction  --------- ")

# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
params = {'axes.labelsize': 16,'axes.titlesize':16, 'font.size': 16, 'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(params)
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering, check_cluster_assignment
from utils.anomaly_detection import AnomalyDetection, calculate_ad_metrics
from utils.prediction import Prediction, calculate_SMAE
from utils.miscellaneous import count_number, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
language = 'pl' # 'pl' or 'eng' - for graphics only
distance = 'euclidean'
stage = 'prediction' # 'clustering', 'ad' or 'prediction'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
prediction_algorithm = 'xgboost' # 'ar' or 'xgboost'
num_metrics = 3
num_experiments = 20
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
print(" Initialization... ")
if precomputed == '2':  # Load file with precomputed values
    if stage == 'clustering':
        file = h5py.File(name='research_and_results/02_bitwise_'+clustering_algorithm+'.h5', mode='r')
    elif stage == 'ad':
        file = h5py.File(name='research_and_results/02_bitwise_'+ad_algorithm+'.h5', mode='r')
    elif stage == 'prediction':
        file = h5py.File(name='research_and_results/02_bitwise_prediction_'+prediction_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations on the original data
    file = h5py.File(name='data\Gdansk.h5', mode='r')
    data = Data(file)
    data.split(train_percentage=50, val_percentage=25)
    file.close()
   # Preprocess data
    data.X_train, _, _ = data.standardize(data.Xraw_train)
    data.X_val, _, _ = data.standardize(data.Xraw_val)
    data.X, _, _ = data.standardize(data.Xraw)  
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
    OK_vec = np.zeros((num_metrics,146))
    for bit in bits:  # For each of AIS message bits
        np.random.seed(250)  # make numpy choose the same messages all the time
        corruption.reset()
        OK_vec2 = np.zeros((num_metrics, num_experiments))  # choose messages
        if stage != "clustering": field = sum(field_bits <= bit)  # check to which field the damaged bit belong to
        for j in range(num_experiments):  # for each chosen message:
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
            X_corr, _, _ = data.standardize(X_corr)
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
                    ad_metrics = calculate_ad_metrics([field], ad.outliers[message_idx][2])
                    OK_vec2[1,j] = ad_metrics["recall"] # save field indication recall
                    OK_vec2[2,j] = ad_metrics["precision"] # save field indication precision

                elif stage == 'prediction': # perform prediction stage 
                    prediction = Prediction(prediction_algorithm=prediction_algorithm)
                    OK_vec2[0,j] = calculate_SMAE(
                        prediction=message_decoded_corr[message_idx,field],
                        real=data.message_decoded[message_idx,field],
                        field=field)
                    pred = prediction.reconstruct_data(
                            message_decoded=data.message_decoded, 
                            timestamp=data.timestamp,
                            idx=data.MMSI,
                            message_idx=message_idx,
                            field=field)
                    OK_vec2[1,j] = calculate_SMAE(
                        prediction=pred if pred is not None else message_decoded_corr[message_idx,field],
                        real=data.message_decoded[message_idx,field],
                        field=field)
                    message_decoded_new, idx_new = prediction.find_and_reconstruct_data(
                        message_decoded=message_decoded_corr, 
                        message_bits=[],
                        idx=idx_corr,
                        timestamp=data.timestamp,
                        outliers=ad.outliers,
                        if_bits=False)
                    OK_vec2[2,j] = calculate_SMAE(
                        prediction=message_decoded_new[message_idx,field],
                        real=data.message_decoded[message_idx,field],
                        field=field)
                    
        if stage != 'prediction':
            if bit < 145: OK_vec[:,bit] = np.mean(OK_vec2, axis=1)*100
            else: OK_vec[:,145] = np.mean(OK_vec2, axis=1)*100
        else:
            if bit < 145: OK_vec[:,bit] = np.mean(OK_vec2, axis=1)
            else: OK_vec[:,145] = np.mean(OK_vec2, axis=1)


# Visualization
print(" Complete.")
if stage == 'clustering':
    if language=='eng': titles = {
        '0':"Correctly assigned messages [%]", 
        '1':"Messages forming standalone clusters [%]", 
        '2':"Correctly assigned messages after correction [%]"}  
    elif language=='pl': titles = {
        '0':"Wiadomości poprawnie przypisane [%]", 
        '1':"Wiadomości tworzące jednoelementowe grupy [%]", 
        '2':"Wiadomości poprawnie przypisane po klasyfikacji [%]"} 
    print(" Correctly assigned messages: " + str(round(np.mean(OK_vec[0,:]),2)) + "%")
    print(" Messages forming standalone clusters: " + str(round(np.mean(OK_vec[1,:]),2)) + "%")
    print(" Correctly assigned messages after correction: " + str(round(np.mean(OK_vec[2,:]),2)) + "%")
elif stage == 'ad': 
    if language=='eng':  titles = {
    '0':"Correctly detected damaged messages - recall [%]", 
    '1':"Correctly detected damaged fields - recall [%]", 
    '2':"Correctly detected damaged fields - precision [%]"}
    elif language=='pl':  titles = {
    '0':"Poprawnie wykryte uszkodzone wiadomości - czułość [%]", 
    '1':"Poprawnie wykryte uszkodzone pola - czułość  [%]", 
    '2':"Poprawnie wykryte uszkodzone pola - precyzja [%]"}
    print(" Recall (message): " + str(round(np.mean(OK_vec[0,mask==1]),2)) + "%")
    print(" Recall (field): " + str(round(np.mean(OK_vec[1,mask==1]),2)) + "%")
    print(" Precision (field): " + str(round(np.mean(OK_vec[2,mask==1]),2)) + "%")
elif stage == 'prediction': 
    if language=='eng': titles = {
    '0':"SMAE (orignal vs damaged field value)", 
    '1':"SMAE (pure prediction, no anomaly detection)",
    '2':"SMAE (prediction after anomaly detection)"}
    elif language=='pl': titles = {
    '0':"SMAE po uszkodzeniu", 
    '1':"SMAE po etapie predykcji",
    '2':"SMAE po całym procesie rekonstrukcji"}
    print(" SMAE (original vs damaged): " + str(round(np.mean(OK_vec[0,mask==1]),6)))
    print(" SMAE (pure prediction): " + str(round(np.mean(OK_vec[1,mask==1]),6)))
    print(" SMAE (for anomalies): " + str(round(np.mean(OK_vec[2,mask==1]),6)))

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
if language=='eng': ax[i].set_xlabel("Index of a damaged bit")
elif language=='pl': ax[i].set_xlabel("Lokalizacja (indeks) uszkodzonego bitu")
fig.legend([
            "Message type","Repeat indicator","MMSI","Navigational status", 
            "Rate of turns","Speed over ground","Position accuracy","Longitude", 
            "Latitude","Course over ground","True heading","Time stamp", 
            "Special manoeuvre indicator", "RAIM-flag"], loc=7)
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
    elif stage == 'prediction':
        if os.path.exists('research_and_results/02_bitwise_prediction_'+prediction_algorithm+'.h5'):
            os.remove('research_and_results/02_bitwise_prediction_'+prediction_algorithm+'.h5')
        file = h5py.File('research_and_results/02_bitwise_prediction_'+prediction_algorithm+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()