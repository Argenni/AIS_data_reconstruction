"""
Checks which stage of AIS data reconstruction is most important. \n
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifier from each AIS message, len=num_messages. \n
Creates 04_ablation_.h5 file, with OK_vec with SMAE of prediction (for each dataset) when each stage
  receives ideal results.
"""
print("\n----------- The importance of each stage on AIS data reconstruction --------- ")

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
from utils.anomaly_detection import AnomalyDetection
from utils.prediction import Prediction, calculate_SMAE
from utils.miscellaneous import count_number, visualize_trajectories, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
prediction_algorithm = 'xgboost' # 'ar' or 'xgboost'
num_metrics = 3
num_experiments = 10
percentage = 10
# ------------------------------------------------------------------------------

# Decide what to do
precomputed = 'start'
while precomputed != '1' and precomputed != '2':
    precomputed = input("Choose: \n1 - Run computations from scratch \n2 - Load precomputed values \n")
    if precomputed != '1' and precomputed != '2':
        print("Unrecognizable answer.")

# Load data
print(" Importing files... ")
if precomputed == '2':  # Load file with precomputed values
    file = h5py.File(name='research_and_results/04_ablation_'+prediction_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations
    filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    OK_vec = np.zeros((len(filename), num_metrics, num_experiments))
    for file_num in range(len(filename)):
        print(" Analysing " + str(file_num+1) + ". dataset...")
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data = Data(file)
        file.close()
        data.split(train_percentage=50, val_percentage=25)
        data.X_train, _, _ = data.standardize(data.Xraw_train)
        data.X_val, _, _ = data.standardize(data.Xraw_val)
        data.X, _, _ = data.standardize(data.Xraw)

        for stage_num in range(num_metrics):           
            # Damage selected messages 
            np.random.seed(1)
            for i in range(num_experiments):  # For each of the randomly chosen AIS messages
                Xraw_corr = copy.deepcopy(data.Xraw)
                MMSI_corr = copy.deepcopy(data.MMSI)
                message_decoded_corr = copy.deepcopy(data.message_decoded)
                corruption = Corruption(data.X)
                ad = AnomalyDetection(ad_algorithm=ad_algorithm)
                fields = []
                messages = []
                num_messages = int(len(data.MMSI)*percentage/100)
                for n in range(num_messages):
                    # Choose 0.05 or 0.1 of all messages and damage 2 their random bits from different fields
                    field = np.random.choice(ad.fields, size=2, replace=False)
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
                X_corr, _, _ = data.standardize(Xraw_corr)

                # Perform clustering
                if stage_num != 0:
                    clustering = Clustering()
                    if clustering_algorithm == 'kmeans': 
                        K_corr, _ = count_number(MMSI_corr)
                        idx_corr, _ = clustering.run_kmeans(X=X_corr, K=K_corr)
                    elif clustering_algorithm == 'DBSCAN':
                        idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr, distance=distance)
                else: # if clustering must be ideal, give ideal results
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    idx_corr = []
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # Perform anomaly detection
                if stage_num != 1:
                    ad.detect_in_1element_clusters(
                        idx=idx_corr,
                        idx_vec=range(-1, np.max(idx_corr)+1),
                        X=X_corr,
                        message_decoded=message_decoded_corr)
                    ad.detect_in_multielement_clusters(
                        idx=idx_corr,
                        message_decoded=message_decoded_corr,
                        timestamp=data.timestamp)
                else: # if ad must be ideal, give ideal results
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    outliers = []
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
                # Perform prediction
                prediction = Prediction(prediction_algorithm=prediction_algorithm)
                if stage_num != 2:
                    message_decoded_new, idx_new = prediction.find_and_reconstruct_data(
                        message_decoded=message_decoded_corr, 
                        message_bits=[],
                        idx=idx_corr,
                        timestamp=data.timestamp,
                        outliers=ad.outliers,
                        if_bits=False)
                else: # if prediction must be ideal, give ideal results
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    predictions = []
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # Calculate final results
                mae = []
                for n in range(num_messages):
                    mae.append(calculate_SMAE(
                        prediction=message_decoded_new[messages[n],fields[n][0]],
                        real = data.message_decoded[messages[n],fields[n][0]],
                        field=fields[n][0]))
                    mae.append(calculate_SMAE(
                        prediction=message_decoded_new[messages[n],fields[n][1]],
                        real = data.message_decoded[messages[n],fields[n][1]],
                        field=fields[n][1]))
                OK_vec[file_num, stage_num, i] = np.mean(mae)
    OK_vec = np.mean(OK_vec, axis=2)


# Visualisation
print(" Complete.")
print("For "+ prediction_algorithm + ", with 10% messages damaged:")
print("- SMAE after ideal clustering - Gdansk: " + str(round(OK_vec[0,0],6)) + ", Baltic: " + str(round(OK_vec[1,0],6)) + ", Gibraltar: "
+ str(round(OK_vec[2,0],6)))
print("- SMAE after ideal anom. det. - Gdansk: " + str(round(OK_vec[0,1],6)) + ", Baltic: " + str(round(OK_vec[1,1],6)) + ", Gibraltar: "
+ str(round(OK_vec[2,1],6)))
print("- SMAE after ideal prediction - Gdansk: " + str(round(OK_vec[0,2],6)) + ", Baltic: " + str(round(OK_vec[1,2],6)) + ", Gibraltar: "
+ str(round(OK_vec[2,2],6)))

# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/04_ablation_'+prediction_algorithm+'.h5'):
        os.remove('research_and_results/04_ablation_'+prediction_algorithm+'.h5')
    file = h5py.File('research_and_results/04_ablation_'+prediction_algorithm+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()