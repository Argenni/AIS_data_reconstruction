"""
Checks the performace of AIS data reconstruction (damages 2 bits of 5% or 10% AIS messages). \n
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifier from each AIS message, len=num_messages. \n
Creates 01b_performance_.h5 file, with OK_vec with:
- if clustering: for both DBSCAN and kmeans, percentage of correctly clustered messages,
- if anomaly detection: for each amount of dataset/corrupted bits:
    0. message classification recall,
    1. message classification precision,
    2. message classification recall,
    3. message classification precision.
"""
print("\n----------- AIS data reconstruction performance - given percentage of damaged messages at a time --------- ")

# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
plt.rcParams.update({'font.size': 16})
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering, calculate_CC
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'xgboost' # 'rf', 'xgboost' or 'LOF'
stage = 'ad' # 'clustering' or 'ad'
num_metrics = {'clustering':5, 'ad':4}
num_experiments = {'clustering':1, 'ad':10}
if stage == 'clustering': percentages = [0]
else: percentages = [5, 10]
# --------------------------------------------------------------------------------
class AnomalyDetection_LOF(AnomalyDetection):
    """
    Class that inherits from Anomaly Detection for checking the performance of LOF in this stage.
    """
    
    def __init__(self, wavelet='morlet'):
        """
        Class initialization (class object creation). Argument:
         wavelet (optional) - string, which wavelet to use while computing cwt in 1-element clusters analysis:
            'morlet' or 'ricker' (as available in SciPy), default='morlet'.
        """
        self._wavelet = wavelet
        self.k = 5
        self.outliers = []

    def _train_1element_classifier(self): pass
    def _create_1element_classifier_dataset(self): pass
    def _optimize_1element_classifier(self): pass
    def _optimize_knn(self): pass
    def _create_multielement_classifier_dataset(self): pass
    def _train_multielement_classifier(self): pass
    def _optimize_multielement_classifier(self): pass
    def _find_damaged_fields(self): pass

    def detect_in_1element_clusters(self, idx, idx_vec, X, message_decoded):
        """
        Runs the entire anomaly detection based on searching for 1-element clusters. Arguments:
        - idx - list of number of cluster assigned to each AIS message, len=num_messages,
        - idx_vec - list of uniqe cluster numbers in a dataset,
        - X - numpy array, AIS feature vectors, shape=(num_messages, num_features (115)),
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)).
        """
        # Initialize
        if len(self.outliers)==0 or len(self.outliers)!=X.shape[0]:
            self.outliers = np.zeros((X.shape[0],3), dtype=int).tolist()
        # Find 1-element clusters
        indices = self._find_1element_clusters(idx, idx_vec)
        for i in indices:
            idx_new = copy.deepcopy(idx)
            # Mark those points as outliers
            self.outliers[i][0] = 1
            # Find the correct clusters for that points
            self.outliers[i][1] = self._find_correct_cluster(X, idx_new, i, indices)
            idx_new[i] = self.outliers[i][1]
            # Find the damaged fields to correct
            messages_idx = (np.where(np.array(idx_new)==idx_new[i])[0]).tolist()
            message_idx_new = messages_idx.index(i)
            for field in self.fields:
                samples = []
                for message in messages_idx:
                    samples.append(np.array(self.compute_fields_diff(message_decoded, idx_new, message, field)))
                clf = LocalOutlierFactor()
                pred = clf.fit_predict(samples)
                if pred[message_idx_new] == -1:
                    if self.outliers[i][2] == 0: self.outliers[i][2] = [field]
                    else: 
                        if field not in self.outliers[i][2]: 
                            self.outliers[i][2] = self.outliers[i][2] + [field]
            # If around half of fields are classified abnormal, that message is not an outlier
            if self.outliers[i][2] != 0:
                if len(self.outliers[i][2])>=np.floor(len(self.fields)/2): self.outliers[i][0] = 0

    def detect_in_multielement_clusters(self, idx, message_decoded, timestamp):
        """
        Runs the anomaly detection for messages inside multi-element clusters. Arguments:
        - idx - list of number of cluster assigned to each AIS message, len=num_messages,
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
        - timestamp - list of strings with timestamp of each message, len=num_messages.
        """
        # Initialize
        if len(self.outliers)==0 or len(self.outliers)!=message_decoded.shape[0]:
            self.outliers = np.zeros((message_decoded.shape[0],3), dtype=int).tolist()
        # Evaluate identifier fields [2,3,12]
        for field in self.fields_static:
            _, idx_vec = count_number(idx)
            for i in idx_vec:
                messages_idx = (np.where(np.array(idx)==i)[0]).tolist()
                if len(messages_idx)>2:
                    waveform = message_decoded[messages_idx,field]
                    if np.std(waveform): #If there is at least one outstanding value
                        waveform = waveform.reshape(-1, 1)    
                        clf = LocalOutlierFactor()
                        pred2 = clf.fit_predict(waveform)
                        outliers = (np.where(pred2==-1)[0]).tolist()
                        for outlier in outliers:
                            message_idx = messages_idx[outlier]
                            self.outliers[message_idx][0] = 1
                            self.outliers[message_idx][1] = idx[message_idx]
                            if self.outliers[message_idx][2]==0: self.outliers[message_idx][2] = [field]
                            else: 
                                if field not in self.outliers[message_idx][2]: 
                                    self.outliers[message_idx][2] = self.outliers[message_idx][2] + [field]
        # Evaluate regular fields [5,7,8,9]
        pred = []
        for field in range(len(self.fields_dynamic)):
            samples = []
            for message_idx in range(message_decoded.shape[0]):
                samples.append(self.compute_multielement_sample(message_decoded, idx, message_idx, timestamp, self.fields_dynamic[field]))
            clf = LocalOutlierFactor()
            pred.append(np.round(clf.fit_predict(samples)))
        for message_idx in range(message_decoded.shape[0]):
            if len(np.where(np.array(idx)==idx[message_idx])[0])>2:
                fields = []
                for i in range(len(self.fields_dynamic)):
                    if pred[i][message_idx]==-1: fields.append(self.fields_dynamic[i])
                if len(fields):
                    self.outliers[message_idx][0] = 1
                    self.outliers[message_idx][1] = idx[message_idx]
                    if self.outliers[message_idx][2]==0: self.outliers[message_idx][2] = fields
                    else: self.outliers[message_idx][2] = self.outliers[message_idx][2] + fields
# --------------------------------------------------------------------------------------------------------------------------

# Decide what to do
precomputed = 'start'
while precomputed != '1' and precomputed != '2':
    precomputed = input("Choose: \n1 - Run computations from scratch \n2 - Load precomputed values \n")
    if precomputed != '1' and precomputed != '2':
        print("Unrecognizable answer.")

# Load data
print(" Importing files... ")
if precomputed == '2':  # Load file with precomputed values
    if stage == 'clustering': file = h5py.File(name='research_and_results/01b_performance_'+clustering_algorithm+'.h5', mode='r')
    elif stage == 'ad': file = h5py.File(name='research_and_results/01b_performance_'+ad_algorithm+'.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations
    filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
    if stage != 'clustering':
        bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    OK_vec = np.zeros((len(filename), len(percentages), num_metrics[stage], num_experiments[stage]))
    for file_num in range(len(filename)):
        print(" Analysing " + str(file_num+1) + ". dataset...")
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data = Data(file)
        file.close()
        if stage != 'clustering':
            data.split(train_percentage=50, val_percentage=25)
            data.X_train, _, _ = data.standarize(data.Xraw_train)
            data.X_val, _, _ = data.standarize(data.Xraw_val)
        data.X, _, _ = data.standarize(data.Xraw)
        # Perform (first) clustering
        clustering = Clustering()
        K, MMSI_vec = count_number(data.MMSI)
        if clustering_algorithm == 'kmeans': idx, _ = clustering.run_kmeans(X=data.X, K=K)
        elif clustering_algorithm == 'DBSCAN': idx, K = clustering.run_DBSCAN(X=data.X, distance=distance)

        if stage == 'clustering':
            # Compute results of clustering
            OK_vec[file_num, 0, 0, 0] = K
            OK_vec[file_num, 0, 1, 0] = silhouette_score(data.X, idx)
            CC, CHC, VHC = calculate_CC(
                idx=idx, 
                MMSI=data.MMSI,
                MMSI_vec=MMSI_vec, 
                if_all=True)
            OK_vec[file_num, 0, 2, 0] = CHC
            OK_vec[file_num, 0, 3, 0] = VHC
            OK_vec[file_num, 0, 4, 0] = CC
        
        else: 
            # Damage selected messages 
            for percentage_num in range(len(percentages)):
                np.random.seed(1)
                for i in range(num_experiments[stage]):  # For each of the randomly chosen AIS messages
                    Xraw_corr = copy.deepcopy(data.Xraw)
                    MMSI_corr = copy.deepcopy(data.MMSI)
                    message_decoded_corr = copy.deepcopy(data.message_decoded)
                    corruption = Corruption(data.X)
                    if ad_algorithm=='LOF': ad = AnomalyDetection_LOF()
                    else: ad = AnomalyDetection(ad_algorithm=ad_algorithm)
                    fields = []
                    messages = []
                    num_messages = int(len(data.MMSI)*percentages[percentage_num]/100)
                    for n in range(num_messages):
                        # Choose 0.05 or 0.1 of all messages and damage 2 their random bits
                        field = np.random.choice(ad.fields_dynamic, size=2, replace=False)
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
                    X_corr, _, _ = data.standarize(Xraw_corr)
                    # cluster again to find new cluster assignment
                    if clustering_algorithm == 'kmeans': 
                        K_corr, _ = count_number(MMSI_corr)
                        idx_corr, _ = clustering.run_kmeans(X=X_corr, K=K_corr)
                    elif clustering_algorithm == 'DBSCAN':
                        idx_corr, K_corr = clustering.run_DBSCAN(X=X_corr, distance=distance)
                    # Perform anomaly detection
                    ad.detect_in_1element_clusters(
                        idx=idx_corr,
                        idx_vec=range(-1, np.max(idx_corr)+1),
                        X=X_corr,
                        message_decoded=message_decoded_corr)
                    ad.detect_in_multielement_clusters(
                        idx=idx_corr,
                        message_decoded=message_decoded_corr,
                        timestamp=data.timestamp)

                    if stage == 'ad':
                        # Compute results of anomaly detection
                        pred = np.array([ad.outliers[n][0] for n in range(len(ad.outliers))], dtype=int)
                        true = np.array(corruption.indices_corrupted, dtype=int)
                        OK_vec[file_num, percentage_num, 0, i] = recall_score(true, pred) # message recall
                        OK_vec[file_num, percentage_num, 1, i] = precision_score(true, pred) # message precision
                        recall = []
                        precision =[]
                        for n in range(num_messages):
                            accuracy = calculate_ad_accuracy(fields[n], ad.outliers[messages[n]][2])
                            recall.append(accuracy["recall"])
                            precision.append(accuracy["precision"])
                        OK_vec[file_num, percentage_num, 2, i] = np.mean(recall) # field recall
                        OK_vec[file_num, percentage_num, 3, i] = np.mean(precision) # field precision
    OK_vec = np.mean(OK_vec, axis=3)*100


# Visualisation
print(" Complete.")
if stage=='clustering': 
    print("For "+ clustering_algorithm +":")
    print("- No. clusters - Gdansk: " + str(int(OK_vec[0,0,0]/100)) + ", Baltic: " + str(int(OK_vec[1,0,0]/100)) + ", Gibraltar: "
    + str(int(OK_vec[2,0,0]/100)))
    print("- Silhouette - Gdansk: " + str(round(OK_vec[0,0,1],2)) + "%, Baltic: " + str(round(OK_vec[1,0,1],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,1],2)) + "%")
    print("- CHC - Gdansk: " + str(round(OK_vec[0,0,2],2)) + "%, Baltic: " + str(round(OK_vec[1,0,2],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,2],2)) + "%")
    print("- VHC - Gdansk: " + str(round(OK_vec[0,0,3],2)) + "%, Baltic: " + str(round(OK_vec[1,0,3],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,3],2)) + "%")
    print("- CC - Gdansk: " + str(round(OK_vec[0,0,4],2)) + "%, Baltic: " + str(round(OK_vec[1,0,4],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,4],2)) + "%")
elif stage=='ad':
    print("For "+ ad_algorithm + ", with 5% messages damaged:")
    print("- Recall (message) - Gdansk: " + str(round(OK_vec[0,0,0],2)) + "%, Baltic: " + str(round(OK_vec[1,0,0],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,0],2)) + "%")
    print("- Precision (message) - Gdansk: " + str(round(OK_vec[0,0,1],2)) + "%, Baltic: " + str(round(OK_vec[1,0,1],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,1],2)) + "%")
    print("- Recall (field) - Gdansk: " + str(round(OK_vec[0,0,2],2)) + "%, Baltic: " + str(round(OK_vec[1,0,2],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,2],2)) + "%")
    print("- Precision (field) - Gdansk: " + str(round(OK_vec[0,0,3],2)) + "%, Baltic: " + str(round(OK_vec[1,0,3],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,3],2)) + "%")
    print("For "+ ad_algorithm + ", with 10% messages damaged:")
    print("- Recall (message) - Gdansk: " + str(round(OK_vec[0,1,0],2)) + "%, Baltic: " + str(round(OK_vec[1,1,0],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,0],2)) + "%")
    print("- Precision (message) - Gdansk: " + str(round(OK_vec[0,1,1],2)) + "%, Baltic: " + str(round(OK_vec[1,1,1],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,1],2)) + "%")
    print("- Recall (field) - Gdansk: " + str(round(OK_vec[0,1,2],2)) + "%, Baltic: " + str(round(OK_vec[1,1,2],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,2],2)) + "%")
    print("- Precision (field) - Gdansk: " + str(round(OK_vec[0,1,3],2)) + "%, Baltic: " + str(round(OK_vec[1,1,3],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,3],2)) + "%")


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if stage == 'clustering':
        if os.path.exists('research_and_results/01b_performance_'+clustering_algorithm+'.h5'):
            os.remove('research_and_results/01b_performance_'+clustering_algorithm+'.h5')
        file = h5py.File('research_and_results/01b_performance_'+clustering_algorithm+'.h5', mode='a')
    elif stage == 'ad':
        if os.path.exists('research_and_results/01b_performance_'+ad_algorithm+'.h5'):
            os.remove('research_and_results/01b_performance_'+ad_algorithm+'.h5')
        file = h5py.File('research_and_results/01b_performance_'+ad_algorithm+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()