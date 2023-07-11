# ------------------ Examine the impact of observation time on AIS message reconstruction --------------------
"""
Analyse the datasets using different time windows and check the performace
of all stages of AIS message reconstruction.
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_messages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_messages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (805)
Creates 00_timecomp_.h5 file, with OK_vec with average:
 - if clustering: silhouette and CC,
 - if anomaly detection: F1 score of detecting messages and fields
"""
print("\n----------- AIS message reconstruction - observation time comparison  --------- ")

# ----------- Part 0 - Initialization ----------
# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, silhouette_score
plt.rcParams.update({'font.size': 16})
import datetime
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering, calculate_CC
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption, TimeWindow

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
stage = 'ad' # 'clustering', 'ad' or 'prediction'
filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
percentages_clust = [0, 5, 10, 20]
percentages_ad = [5, 10, 20]
if stage == 'clustering': percentages = percentages_clust
elif stage == 'ad': percentages = percentages_ad
windows = [5, 10, 15, 20, 30, 60, 120, 180, 360]

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
    if stage == 'clustering':
        file = h5py.File(name='research_and_results/00_timecomp_' + clustering_algorithm, mode='r')
    elif stage == 'ad':
        file = h5py.File(name='research_and_results/00_timecomp_' + ad_algorithm, mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations
    bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    measure1 = []
    measure2 = []
    slides=np.zeros((len(windows),len(filename)))
    for file_num in range(len(filename)):
        measure1.append([]) # First index - for a filename
        measure2.append([])
        for percentage_num in range(len(percentages)):
            measure1[file_num].append([]) # Second index - for percentage of damaged messages
            measure2[file_num].append([])
        # Load the data from the right file
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data_original = Data(file)
        data_original.split(train_percentage=50, val_percentage=25)
        file.close()
        overall_time = max(data_original.timestamp)-min(data_original.timestamp)
        overall_time = overall_time.seconds/60
        for window_num in range(len(windows)):   
            for percentage_num in range(len(percentages)):        
                measure1[file_num][percentage_num].append([]) # Third index - for window
                measure2[file_num][percentage_num].append([])
            if windows[window_num] > overall_time: 
                measure1[file_num][percentage_num][window_num].append(0)
                measure2[file_num][percentage_num][window_num].append(0)
            else: 
                start = 0
                stop = start + windows[window_num]
                while stop <= overall_time:
                    # Select only messages from the given time window
                    data = copy.deepcopy(data_original)
                    print(start)
                    print(stop)
                    time_window = TimeWindow(start, stop)
                    time_window.use_time_window(data)
                    data.X, _, _ = data.normalize(data.Xraw)
                    if (data.Xraw).shape[0] == 0: break
                    else: 
                        for percentage_num in range(len(percentages)):
                            # Corrupt data
                            Xraw_corr = copy.deepcopy(data.Xraw)
                            MMSI_corr = copy.deepcopy(data.MMSI)
                            message_decoded_corr = copy.deepcopy(data.message_decoded)
                            corruption = Corruption(data.Xraw,1)
                            messages = []
                            fields = []
                            num_messages = int(len(data.MMSI)*percentages[percentage_num]/100)
                            for n in range(num_messages):
                                bits_corr = np.random.choice(bits, size=2, replace=False)
                                field = [sum(field_bits <= bit) for bit in np.sort(bits_corr)]
                                fields.append(field)
                                message_bits_corr, message_idx = corruption.corrupt_bits(message_bits=data.message_bits, bit_idx=bits_corr[0])
                                message_bits_corr, message_idx = corruption.corrupt_bits(message_bits_corr, message_idx=message_idx, bit_idx=bits_corr[1])
                                messages.append(message_idx)
                                # put it back to the dataset
                                X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
                                Xraw_corr[message_idx,:] = X_0
                                MMSI_corr[message_idx] = MMSI_0
                                message_decoded_corr[message_idx,:] = message_decoded_0
                            # Preprocess data
                            _, MMSI_vec = count_number(data.MMSI)                
                            K, _ = count_number(MMSI_corr)  # Count number of groups/ships
                            Xcorr, _, _ = data.normalize(Xraw_corr) 
                            clustering = Clustering() # perform clustering
                            if clustering_algorithm == 'kmeans':
                                print(" Running k-means clustering...")
                                idx_corr, centroids = clustering.run_kmeans(X=Xcorr,K=K)
                            elif clustering_algorithm == 'DBSCAN':
                                print(" Running DBSCAN clustering...")
                                idx_corr, K = clustering.run_DBSCAN(X=Xcorr,distance=distance)
                            # Run anomaly detection if needed 
                            if stage == 'ad' or stage == 'prediction':
                                outliers = AnomalyDetection(data=data, ad_algorithm=ad_algorithm)
                                outliers.detect_standalone_clusters(
                                    idx=idx_corr,
                                    idx_vec=range(-1, np.max(idx_corr)+1),
                                    X=Xcorr,
                                    message_decoded=message_decoded_corr)
                                outliers.detect_inside(
                                    idx=idx_corr,
                                    message_decoded=message_decoded_corr,
                                    timestamp=data.timestamp)
                            # Compute quality measures
                            if stage == 'clustering':
                                measure1[file_num][percentage_num][window_num].append(silhouette_score(idx_corr))
                                measure1[file_num][percentage_num][window_num].append(calculate_CC(idx_corr, data.MMSI, MMSI_vec))
                            elif stage == 'ad':
                                pred = np.array([outliers.outliers[n][0] for n in range(len(outliers.outliers))], dtype=int)
                                true = np.array(corruption.indices_corrupted, dtype=int)
                                measure1[file_num][percentage_num][window_num].append(f1_score(true, pred))
                                f1 =[]
                                for n in range(num_messages):
                                    accuracy = calculate_ad_accuracy(fields[n], outliers.outliers[messages[n]][2])
                                    f1.append(accuracy["f1"])
                                measure2[file_num][percentage_num][window_num].append(np.mean(f1))
                            if percentage_num==0: slides[window_num,file_num] = slides[window_num,file_num]+1
                    # Slide the time window
                    if windows[window_num] == 5: 
                        start = start + windows[window_num]
                        stop = stop + windows[window_num]
                    else:
                        start = start + np.ceil(windows[window_num]/2)
                        stop = stop + np.ceil(windows[window_num]/2)

    # Compute final results                              
    OK_vec = np.zeros((len(percentages), len(windows), 2)) # For computed quality measures for each time window length
    cum_slides = np.sum(slides, axis=1).reshape(len(windows),-1)
    cum_measures =  np.zeros((2,len(percentages),len(windows),len(filename)))
    for percentage_num in range(len(percentages)):
        for window_num in range(len(windows)):
            for file_num in range(len(filename)):
                cum_measures[0,percentage_num,window_num,file_num] = np.sum(measure1[file_num][percentage_num][window_num])
                cum_measures[1,percentage_num,window_num,file_num] = np.sum(measure2[file_num][percentage_num][window_num])
            OK_vec[percentage_num,window_num,0] = np.sum(cum_measures[0,percentage_num,window_num,:])/cum_slides[window_num] 
            OK_vec[percentage_num,window_num,1] = np.sum(cum_measures[1,percentage_num,window_num,:])/cum_slides[window_num]     

# Visualize
fig, ax = plt.subplots(nrows=2)
for i in range(2):
    legend = []
    for percentage_num in range(len(percentages)):
        ax[i].plot(windows,OK_vec[percentage_num,:,i])
        legend.append(str(percentages[percentage_num])+"% messages damaged")
    ax[i].set_xlabel("Time frame length [min]")
    ax[i].legend(legend)
if stage == 'clustering': 
    ax[0].set_ylabel("Silhouette")
    ax[1].set_ylabel("Correctness coefficient")
elif stage == 'ad': 
    ax[0].set_ylabel("F1 - messages")
    ax[1].set_ylabel("F1 - fields")
fig.show()

# Save the results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    # Save file
    input("Press Enter to save and exit...")
    if stage == 'clustering':
        if os.path.exists('research_and_results/00_timecomp_'+clustering_algorithm):
            os.remove('research_and_results/00_timecomp_'+clustering_algorithm)
            file = h5py.File('research_and_results/00_timecomp_'+clustering_algorithm, mode='a')
    elif stage == 'ad':
        if os.path.exists('research_and_results/00_timecomp_'+ad_algorithm):
            os.remove('research_and_results/00_timecomp_'+ad_algorithm)
            file = h5py.File('research_and_results/00_timecomp_'+ad_algorithm, mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()