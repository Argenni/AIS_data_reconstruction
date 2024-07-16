"""
Checks the statistical differences of algorithms used in different stages of 
AIS data reconstruction (damages 2 bits of 5% or 10% AIS messages and performs Wilcoxon pairwise test). \n
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifier from each AIS message, len=num_messages. \n
Creates 01c_performance_.h5 file, with OK_vec and measurements, OK_vec2 and measurements2 with (for each dataset):
- if clustering: differences between DBSCAN and kmeans silhouette qnd CC,
- if anomaly detection: differences between RF and XGBoost F1 score for fields and messages,
- if prediction: differences between VAR and XGBoost SMAE (no OK_vec2 and measurements2).
"""
print("\n----------- AIS data reconstruction performance - statistical tests --------- ")

# Important imports
import numpy as np
import h5py
from sklearn.metrics import silhouette_score, f1_score
from scipy.stats import wilcoxon
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering, calculate_CC
from utils.anomaly_detection import AnomalyDetection, calculate_ad_metrics
from utils.prediction import Prediction, calculate_SMAE
from utils.miscellaneous import count_number, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
stage = 'clustering' # 'clustering', 'ad' or 'prediction'
num_metrics = 2
num_experiments = 10
percentages = [5, 10]
significance = 0.05
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
    file = h5py.File(name='research_and_results/01c_performance_'+stage+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    measurements = np.array(file.get('measurements'))
    if stage!='prediction':
        OK_vec2 = np.array(file.get('OK_vec2'))
        measurements2 = np.array(file.get('measurements2'))
    file.close()

else:  # or run the computations
    filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    measurements = np.zeros((len(filename), len(percentages), 2, num_experiments))
    if stage!='prediction': measurements2 = np.zeros((len(filename), len(percentages), 2, num_experiments))
    for file_num in range(len(filename)):
        print(" Analysing " + str(file_num+1) + ". dataset...")
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data = Data(file)
        file.close()
        if stage!='clustering':
            data.split(train_percentage=50, val_percentage=25)
            data.X_train, _, _ = data.standardize(data.Xraw_train)
            data.X_val, _, _ = data.standardize(data.Xraw_val)
        data.X, _, _ = data.standardize(data.Xraw) 

        # Damage selected messages 
        for percentage_num in range(len(percentages)):
            np.random.seed(1)
            for i in range(num_experiments):  # For each of the randomly chosen AIS messages
                Xraw_corr = copy.deepcopy(data.Xraw)
                MMSI_corr = copy.deepcopy(data.MMSI)
                message_decoded_corr = copy.deepcopy(data.message_decoded)
                corruption = Corruption(data.X)
                ad = AnomalyDetection(ad_algorithm='xgboost')
                fields = []
                messages = []
                num_messages = int(len(data.MMSI)*percentages[percentage_num]/100)
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
                clustering = Clustering()
                _, MMSI_vec = count_number(data.MMSI)
                K_kmeans, _ = count_number(MMSI_corr)
                idx_corr, K_DBSCAN = clustering.run_DBSCAN(X=X_corr, distance=distance) # default clustering - DBSCAN

                if stage=='clustering':
                    measurements[file_num, percentage_num, 1, i] = silhouette_score(X_corr, idx_corr)
                    measurements2[file_num, percentage_num, 1, i] = calculate_CC(idx_corr, data.MMSI, MMSI_vec)
                    idx_corr, _ = clustering.run_kmeans(X=X_corr, K=K_kmeans) # perform also k-means clustering
                    measurements[file_num, percentage_num, 0, i] = silhouette_score(X_corr, idx_corr)
                    measurements2[file_num, percentage_num, 0, i] = calculate_CC(idx_corr, data.MMSI, MMSI_vec)
                    
                else:
                    # Perform anomaly detection
                    ad.detect_in_1element_clusters( # default ad - XGBoost
                        idx=idx_corr,
                        idx_vec=range(-1, np.max(idx_corr)+1),
                        X=X_corr,
                        message_decoded=message_decoded_corr)
                    ad.detect_in_multielement_clusters(
                        idx=idx_corr,
                        message_decoded=message_decoded_corr,
                        timestamp=data.timestamp)
                    if stage=='ad':
                        # Compute results of anomaly detection - XGBoost
                        f1 = []
                        for n in range(num_messages):
                            ad_metrics = calculate_ad_metrics(fields[n], ad.outliers[messages[n]][2])
                            f1.append(ad_metrics["f1"])
                        measurements[file_num, percentage_num, 1, i] = np.mean(f1)
                        pred = np.array([ad.outliers[n][0] for n in range(len(ad.outliers))], dtype=int)
                        true = np.array(corruption.indices_corrupted, dtype=int)
                        measurements2[file_num, percentage_num, 1, i] = f1_score(true, pred) 
                        # Compute results of anomaly detection - RF
                        ad2 = AnomalyDetection(ad_algorithm='rf')
                        ad2.detect_in_1element_clusters(
                            idx=idx_corr,
                            idx_vec=range(-1, np.max(idx_corr)+1),
                            X=X_corr,
                            message_decoded=message_decoded_corr)
                        ad2.detect_in_multielement_clusters(
                            idx=idx_corr,
                            message_decoded=message_decoded_corr,
                            timestamp=data.timestamp)
                        f1 = []
                        for n in range(num_messages):
                            ad_metrics = calculate_ad_metrics(fields[n], ad2.outliers[messages[n]][2])
                            f1.append(ad_metrics["f1"])
                        measurements[file_num, percentage_num, 0, i] = np.mean(f1)
                        pred = np.array([ad2.outliers[n][0] for n in range(len(ad2.outliers))], dtype=int)
                        true = np.array(corruption.indices_corrupted, dtype=int)
                        measurements2[file_num, percentage_num, 0, i] = f1_score(true, pred)
                    elif stage == 'prediction':
                        prediction_algorithm = ['xgboost', 'ar']
                        for alg_num in range(len(prediction_algorithm)):
                            prediction = Prediction(prediction_algorithm=prediction_algorithm[alg_num])
                            message_decoded_new, idx_new = prediction.find_and_reconstruct_data(
                                message_decoded=message_decoded_corr, 
                                message_bits=[],
                                idx=idx_corr,
                                timestamp=data.timestamp,
                                outliers=ad.outliers,
                                if_bits=False)
                            mae_new = []
                            for n in range(num_messages):
                                mae_new.append(calculate_SMAE(
                                    prediction=message_decoded_new[messages[n],fields[n][0]],
                                    real = data.message_decoded[messages[n],fields[n][0]],
                                    field=fields[n][0]))
                                mae_new.append(calculate_SMAE(
                                    prediction=message_decoded_new[messages[n],fields[n][1]],
                                    real = data.message_decoded[messages[n],fields[n][1]],
                                    field=fields[n][1]))
                            measurements[file_num, percentage_num, alg_num, i] = np.mean(mae_new)

    # Perform true test
    OK_vec = np.zeros((len(filename), len(percentages), 2))
    for file_num in range(len(filename)):
        for percentage_num in range(len(percentages)):
            test = wilcoxon(
                x=measurements[file_num, percentage_num, 1, :],
                y=measurements[file_num, percentage_num, 0, :],
                alternative='greater'
            )
            OK_vec[file_num, percentage_num, 0] = test.statistic
            OK_vec[file_num, percentage_num, 1] = test.pvalue
    if stage!='prediction':
        OK_vec2 = np.zeros((len(filename), len(percentages), 2))
        for file_num in range(len(filename)):
            for percentage_num in range(len(percentages)):
                test = wilcoxon(
                    x=measurements2[file_num, percentage_num, 1, :],
                    y=measurements2[file_num, percentage_num, 0, :],
                    alternative='greater'
                )
                OK_vec2[file_num, percentage_num, 0] = test.statistic
                OK_vec2[file_num, percentage_num, 1] = test.pvalue


# Visualisation
print(" Complete.")
print(" With 5% messages damaged:")
print(" - Gdansk - statistic: " + str(round(OK_vec[0,0,0],4)) + ", pvalue: " + str(round(OK_vec[0,0,1],4)) + ", null hypothesis " 
      + ("proven" if OK_vec[0,0,1]>significance else "rejected"))
print(" - Baltic - statistic: " + str(round(OK_vec[1,0,0],4)) + ", pvalue: " + str(round(OK_vec[1,0,1],4)) + ", null hypothesis " 
      + ("proven" if OK_vec[1,0,1]>significance else "rejected"))
print(" - Gibral - statistic: " + str(round(OK_vec[2,0,0],4)) + ", pvalue: " + str(round(OK_vec[2,0,1],4)) + ", null hypothesis " 
      + ("proven" if OK_vec[2,0,1]>significance else "rejected"))
print(" With 10% messages damaged:")
print(" - Gdansk - statistic: " + str(round(OK_vec[0,1,0],4)) + ", pvalue: " + str(round(OK_vec[0,1,1],4)) + ", null hypothesis " 
      + ("proven" if OK_vec[0,1,1]>significance else "rejected"))
print(" - Baltic - statistic: " + str(round(OK_vec[1,1,0],4)) + ", pvalue: " + str(round(OK_vec[1,1,1],4)) + ", null hypothesis " 
      + ("proven" if OK_vec[1,1,1]>significance else "rejected"))
print(" - Gibral - statistic: " + str(round(OK_vec[2,1,0],4)) + ", pvalue: " + str(round(OK_vec[2,1,1],4)) + ", null hypothesis " 
      + ("proven" if OK_vec[2,1,1]>significance else "rejected"))
if stage!='prediction':
    print("Second measure:")
    print(" With 5% messages damaged:")
    print(" - Gdansk - statistic: " + str(round(OK_vec2[0,0,0],4)) + ", pvalue: " + str(round(OK_vec2[0,0,1],4)) + ", null hypothesis " 
        + ("proven" if OK_vec2[0,0,1]>significance else "rejected"))
    print(" - Baltic - statistic: " + str(round(OK_vec2[1,0,0],4)) + ", pvalue: " + str(round(OK_vec2[1,0,1],4)) + ", null hypothesis " 
        + ("proven" if OK_vec2[1,0,1]>significance else "rejected"))
    print(" - Gibral - statistic: " + str(round(OK_vec2[2,0,0],4)) + ", pvalue: " + str(round(OK_vec2[2,0,1],4)) + ", null hypothesis " 
        + ("proven" if OK_vec2[2,0,1]>significance else "rejected"))
    print(" With 10% messages damaged:")
    print(" - Gdansk - statistic: " + str(round(OK_vec2[0,1,0],4)) + ", pvalue: " + str(round(OK_vec2[0,1,1],4)) + ", null hypothesis " 
        + ("proven" if OK_vec2[0,1,1]>significance else "rejected"))
    print(" - Baltic - statistic: " + str(round(OK_vec2[1,1,0],4)) + ", pvalue: " + str(round(OK_vec2[1,1,1],4)) + ", null hypothesis " 
        + ("proven" if OK_vec2[1,1,1]>significance else "rejected"))
    print(" - Gibral - statistic: " + str(round(OK_vec2[2,1,0],4)) + ", pvalue: " + str(round(OK_vec2[2,1,1],4)) + ", null hypothesis " 
        + ("proven" if OK_vec2[2,1,1]>significance else "rejected"))


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if os.path.exists('research_and_results/01c_performance_'+stage+'.h5'):
        os.remove('research_and_results/01c_performance_'+stage+'.h5')
    file = h5py.File('research_and_results/01c_performance_'+stage+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.create_dataset('measurements', data=measurements)
    if stage!='prediction':
        file.create_dataset('OK_vec2', data=OK_vec2)
        file.create_dataset('measurements2', data=measurements2)
    file.close()