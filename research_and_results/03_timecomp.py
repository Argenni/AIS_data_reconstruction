"""
Analyses the datasets using different time windows and check the performace
of all stages of AIS message reconstruction. \n
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifiers from each AIS message, len=num_messages. \n
Creates 03_timecomp_.h5 file, with OK_vec with average: 
 - if clustering: silhouette and CC,
 - if anomaly detection: F1 score of detecting messages and fields,
 - if prediction: SMAE of pure prediction and after anomaly detection.
"""
print("\n----------- The impact of observation time on AIS message reconstruction  --------- ")

# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, silhouette_score
params = {'axes.labelsize': 16,'axes.titlesize':16, 'font.size': 16, 'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(params)
import timeit
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering, calculate_CC
from utils.anomaly_detection import AnomalyDetection, calculate_ad_metrics
from utils.prediction import Prediction, calculate_SMAE
from utils.miscellaneous import count_number, Corruption, TimeWindow

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
language = 'pl' # 'pl' or 'eng' - for graphics only
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf' or 'xgboost'
prediction_algorithm = 'xgboost' # 'ar' or  'xgboost'
stage = 'all' # 'clustering', 'ad', 'prediction' or 'all' 
if stage == 'clustering': percentages =  [0, 5, 10, 20]
else: percentages = [5, 10, 20]
windows = [5, 10, 15, 20, 30, 60, 120]
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
    if stage == 'clustering':
        file = h5py.File(name='research_and_results/03_timecomp_' + clustering_algorithm+'.h5', mode='r')
    elif stage == 'ad':
        file = h5py.File(name='research_and_results/03_timecomp_' + ad_algorithm+'.h5', mode='r')
    elif stage == 'prediction':
        file = h5py.File(name='research_and_results/03_timecomp_prediction_' + prediction_algorithm+'.h5', mode='r')
    elif stage == 'all':
        file = h5py.File(name='research_and_results/03_timecomp_time_' + prediction_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations
    filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
    if stage=='clustering':
        bits = list(range(145))  
        bits.append(148)
    else:
        bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    measure1 = []
    measure2 = []
    if stage == 'all': measure3 = []
    slides=np.zeros((len(windows),len(filename)))
    for file_num in range(len(filename)):
        measure1.append([]) # First index - for a filename
        measure2.append([])
        if stage == 'all': measure3.append([])
        for percentage_num in range(len(percentages)):
            measure1[file_num].append([]) # Second index - for percentage of damaged messages
            measure2[file_num].append([])
            if stage == 'all': measure3[file_num].append([])
        # Load the data from the right file
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data_original = Data(file)
        file.close()
        if stage!='clustering':
            data_original.split(train_percentage=50, val_percentage=25)
            data_original.X_train, _, _ = data_original.standardize(data_original.Xraw_train)
            data_original.X_val, _, _ = data_original.standardize(data_original.Xraw_val)
        overall_time = max(data_original.timestamp)-min(data_original.timestamp)
        overall_time = overall_time.seconds/60
        for window_num in range(len(windows)):
            print(" Analysing: dataset " + str(file_num+1)+"., time window " + str(windows[window_num]) +" min...") 
            for percentage_num in range(len(percentages)):        
                measure1[file_num][percentage_num].append([]) # Third index - for window
                measure2[file_num][percentage_num].append([])
                if stage == 'all': measure3[file_num][percentage_num].append([])
                if windows[window_num] > overall_time: 
                    measure1[file_num][percentage_num][window_num].append(0)
                    measure2[file_num][percentage_num][window_num].append(0)
                    if stage == 'all': measure3[file_num][percentage_num][window_num].append(0)
            if windows[window_num] <= overall_time: 
                start = 0
                stop = start + windows[window_num]
                while stop <= overall_time:
                    # Select only messages from the given time window
                    data = copy.deepcopy(data_original)
                    time_window = TimeWindow(start, stop)
                    data = time_window.use_time_window(data, crop_train=False, crop_val=False, verbose=False)
                    data.X, _, _ = data.standardize(data.Xraw)
                    if (data.Xraw).shape[0]>1:
                        for percentage_num in range(len(percentages)):
                            # Damage data
                            Xraw_corr = copy.deepcopy(data.Xraw)
                            MMSI_corr = copy.deepcopy(data.MMSI)
                            message_decoded_corr = copy.deepcopy(data.message_decoded)
                            corruption = Corruption(data.Xraw)
                            messages = []
                            fields = []
                            num_messages = int(np.ceil(len(data.MMSI)*percentages[percentage_num]/100))
                            if num_messages>len(data.MMSI): num_messages = num_messages -1 
                            for n in range(num_messages):
                                bits_corr = np.random.choice(bits, size=2, replace=False)
                                field = list(set([sum(field_bits <= bit) for bit in np.sort(bits_corr)]))
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
                            if stage == 'all': tic = timeit.default_timer()
                            _, MMSI_vec = count_number(data.MMSI)                
                            K, MMSI_corr_vec = count_number(MMSI_corr)  # Count number of groups/ships
                            Xcorr, _, _ = data.standardize(Xraw_corr) 
                            clustering = Clustering() # perform clustering
                            if clustering_algorithm == 'kmeans':
                                idx_corr, _ = clustering.run_kmeans(X=Xcorr,K=K)
                            elif clustering_algorithm == 'DBSCAN':
                                idx_corr, K = clustering.run_DBSCAN(X=Xcorr,distance=distance)
                            if stage == 'all': toc1 = timeit.default_timer() - tic
                            # Run anomaly detection if needed 
                            if stage != 'clustering':
                                ad = AnomalyDetection(ad_algorithm=ad_algorithm)
                                ad.detect_in_1element_clusters(
                                    idx=idx_corr,
                                    idx_vec=range(-1, np.max(idx_corr)+1),
                                    X=Xcorr,
                                    message_decoded=message_decoded_corr)
                                ad.detect_in_multielement_clusters(
                                    idx=idx_corr,
                                    message_decoded=message_decoded_corr,
                                    timestamp=data.timestamp)
                                if stage == 'all': toc2 = timeit.default_timer() - tic
                            if stage == 'prediction' or stage == 'all': # Perform prediction if needed
                                prediction = Prediction(prediction_algorithm=prediction_algorithm)
                                message_decoded_new, idx_new = prediction.find_and_reconstruct_data(
                                    message_decoded=message_decoded_corr, 
                                    message_bits=[],
                                    idx=idx_corr,
                                    timestamp=data.timestamp,
                                    outliers=ad.outliers,
                                    if_bits=False)
                                if stage == 'all': toc3 = timeit.default_timer() - tic
                            # Compute quality measures
                            if stage == 'clustering':
                                if count_number(idx_corr)[0]<Xcorr.shape[0]:
                                    measure1[file_num][percentage_num][window_num].append(silhouette_score(Xcorr, idx_corr))
                                else: measure1[file_num][percentage_num][window_num].append(1)
                                measure2[file_num][percentage_num][window_num].append(calculate_CC(idx_corr, data.MMSI, MMSI_vec))
                            elif stage == 'ad':
                                pred = np.array([ad.outliers[n][0] for n in range(len(ad.outliers))], dtype=int)
                                true = np.array(corruption.indices_corrupted, dtype=int)
                                measure1[file_num][percentage_num][window_num].append(f1_score(true, pred))
                                f1 =[]
                                for n in range(num_messages):
                                    ad_metrics = calculate_ad_metrics(fields[n], ad.outliers[messages[n]][2])
                                    f1.append(ad_metrics["f1"])
                                measure2[file_num][percentage_num][window_num].append(np.mean(f1))
                            elif stage == 'prediction':
                                mae_pred = []
                                mae_ad = []
                                for n in range(num_messages):
                                    for field in fields[n]:
                                        pred = prediction.reconstruct_data(
                                            message_decoded=data.message_decoded, 
                                            timestamp=data.timestamp,
                                            idx=data.MMSI,
                                            message_idx=messages[n],
                                            field=field)
                                        mae_pred.append(calculate_SMAE(
                                            prediction=pred if pred is not None else message_decoded_corr[messages[n],field],
                                            real=data.message_decoded[messages[n], field],
                                            field=field))
                                        mae_ad.append(calculate_SMAE(
                                            prediction=message_decoded_new[messages[n], field],
                                            real=data.message_decoded[messages[n], field],
                                            field=field))
                                measure1[file_num][percentage_num][window_num].append(np.mean(mae_pred))
                                measure2[file_num][percentage_num][window_num].append(np.mean(mae_ad))
                            elif stage == 'all':
                                measure1[file_num][percentage_num][window_num].append(toc1)
                                measure2[file_num][percentage_num][window_num].append(toc2)
                                measure3[file_num][percentage_num][window_num].append(toc3)
                        slides[window_num,file_num] = slides[window_num,file_num]+1
                    # Slide the time window
                    if windows[window_num] == 5: 
                        start = start + windows[window_num]
                        stop = stop + windows[window_num]
                    else:
                        start = start + np.ceil(windows[window_num]/2)
                        stop = stop + np.ceil(windows[window_num]/2)

    # Compute final results                              
    OK_vec = np.zeros((len(percentages), len(windows), 3 if stage=='all' else 2)) # For computed quality measures for each time window length
    cum_slides = np.sum(slides, axis=1).reshape(len(windows),-1)
    cum_measures =  np.zeros((3 if stage=='all' else 2, len(percentages), len(windows), len(filename)))
    for percentage_num in range(len(percentages)):
        for window_num in range(len(windows)):
            for file_num in range(len(filename)):
                cum_measures[0,percentage_num,window_num,file_num] = np.sum(measure1[file_num][percentage_num][window_num])
                cum_measures[1,percentage_num,window_num,file_num] = np.sum(measure2[file_num][percentage_num][window_num])
                if stage=='all': cum_measures[2,percentage_num,window_num,file_num] = np.sum(measure3[file_num][percentage_num][window_num])
            if cum_slides[window_num]:
                OK_vec[percentage_num,window_num,0] = np.sum(cum_measures[0,percentage_num,window_num,:])/cum_slides[window_num] 
                OK_vec[percentage_num,window_num,1] = np.sum(cum_measures[1,percentage_num,window_num,:])/cum_slides[window_num] 
                if stage=='all':  OK_vec[percentage_num,window_num,2] = np.sum(cum_measures[2,percentage_num,window_num,:])/cum_slides[window_num]


# Visualize
print(" Complete.")
colors = {0:'b', 1:'r', 2:'g', 3:'k'}
if stage != 'all':
    fig, ax = plt.subplots(nrows=2)
    for i in range(2):
        legend = []
        for percentage_num in range(len(percentages)):
            ax[i].plot(windows,OK_vec[percentage_num,:,i], color=colors[percentage_num])
            ax[i].scatter(windows,OK_vec[percentage_num,:,i], color=colors[percentage_num], s=6)
            if language=='eng': legend.append(str(percentages[percentage_num])+"% messages damaged")
            elif language=='pl': legend.append(str(percentages[percentage_num])+"% uszkodzonych wiadomości")
        if language=='eng': ax[i].set_xlabel("Time frame length [min]")
        elif language=='pl': ax[i].set_xlabel("Czas obserwacji [min]")
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    if stage == 'clustering': 
        ax[0].set_ylabel("Silhouette")
        ax[1].set_ylabel("CC")
    elif stage == 'ad': 
        if language=='eng': ax[0].set_ylabel("F1 - messages")
        elif language=='pl': ax[0].set_ylabel("F1 dla wiadomości")
        if language=='eng': ax[1].set_ylabel("F1 - fields")
        elif language=='pl': ax[1].set_ylabel("F1 dla pól")
    elif stage  == 'prediction':
        if language=='eng': ax[0].set_ylabel("SMAE - pure prediction")
        elif language=='pl': ax[0].set_ylabel("SMAE - tylko etap predykcji")
        if language=='eng': ax[1].set_ylabel("SMAE - with anomaly detection")
        elif language=='pl':ax[1].set_ylabel("SMAE - cały proces rekonstrukcji")
    fig.legend(legend, loc=9)
    fig.show()
else:
    fig, ax = plt.subplots()
    legend = []
    if language=='eng': stages = {0:"clustering", 1:"anomaly detection", 2:"prediction"}
    elif language=='pl': stages = {0:"grupowanie", 1:"wykrywanie wyjątków", 2:"predykcja"}
    linestyles = {0:"-", 1:"--", 2:":"}
    for i in range(3):
        for percentage_num in range(len(percentages)):
            ax.plot(windows,OK_vec[percentage_num,:,i], linestyle=linestyles[i], color=colors[percentage_num])
            ax.scatter(windows,OK_vec[percentage_num,:,i], color=colors[percentage_num], s=6)
            if language=='eng': legend.append(str(percentages[percentage_num])+"% messages damaged, "+stages[i])
            elif language=='pl': legend.append(str(percentages[percentage_num])+"% uszkodzonych wiadomości, "+stages[i])
    if language=='eng': ax.set_xlabel("Time frame length [min]")
    elif language=='pl': ax.set_xlabel("Czas obserwacji [min]")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if language=='eng': ax.set_ylabel("Average elapsed time [s]")
    elif language=='pl': ax.set_ylabel("Średni czas wykonania [s]")
    fig.legend(legend, loc=9)
    fig.show()

# Save the results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if stage == 'clustering':
        if os.path.exists('research_and_results/03_timecomp_'+clustering_algorithm+'.h5'):
            os.remove('research_and_results/03_timecomp_'+clustering_algorithm+'.h5')
        file = h5py.File('research_and_results/03_timecomp_'+clustering_algorithm+'.h5', mode='a')
    elif stage == 'ad':
        if os.path.exists('research_and_results/03_timecomp_'+ad_algorithm+'.h5'):
            os.remove('research_and_results/03_timecomp_'+ad_algorithm+'.h5')
        file = h5py.File('research_and_results/03_timecomp_'+ad_algorithm+'.h5', mode='a')
    elif stage == 'prediction':
        if os.path.exists('research_and_results/03_timecomp_prediction_'+prediction_algorithm+'.h5'):
            os.remove('research_and_results/03_timecomp_prediction_'+prediction_algorithm+'.h5')
        file = h5py.File('research_and_results/03_timecomp_prediction_'+prediction_algorithm+'.h5', mode='a')
    elif stage == 'all':
        if os.path.exists('research_and_results/03_timecomp_time_'+prediction_algorithm+'.h5'):
            os.remove('research_and_results/03_timecomp_time_'+prediction_algorithm+'.h5')
        file = h5py.File('research_and_results/03_timecomp_time_'+prediction_algorithm+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()