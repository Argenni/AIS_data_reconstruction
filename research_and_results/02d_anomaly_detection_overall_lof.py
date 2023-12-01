"""
Artificially damages random bits of randomly chosen AIS messages and checks the performace
of anomaly detection phase using LOF instead of RF/XGBoost.
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len=num_messages.
Creates 02c_anomaly_detection_standalone_clusters_Gdansk_.h5 file, with OK_vec with:
1. message indication recall,
2. message indication precision,
3. message indication accuracy,
4. fields to correct classification recall,
5. fields to correct classification precision.
"""
print("\n----------- AIS Anomaly detection - LOF accuracy --------- ")

# ----------- Initialization ----------
# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
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
filename = 'Gibraltar.h5' # 'Gdansk', 'Baltic', 'Gibraltar'
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
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
        for field in self.inside_fields2:
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
        for field in range(len(self.inside_fields)):
            samples = []
            for message_idx in range(message_decoded.shape[0]):
                samples.append(self.compute_multielement_sample(message_decoded, idx, message_idx, timestamp, self.inside_fields[field]))
            clf = LocalOutlierFactor()
            pred.append(np.round(clf.fit_predict(samples)))
        for message_idx in range(message_decoded.shape[0]):
            if len(np.where(np.array(idx)==idx[message_idx])[0])>2:
                fields = []
                for i in range(len(self.inside_fields)):
                    if pred[i][message_idx]==-1: fields.append(self.inside_fields[i])
                if len(fields):
                    self.outliers[message_idx][0] = 1
                    self.outliers[message_idx][1] = idx[message_idx]
                    if self.outliers[message_idx][2]==0: self.outliers[message_idx][2] = fields
                    else: self.outliers[message_idx][2] = self.outliers[message_idx][2] + fields


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
        name='research_and_results/02d_anomaly_detection_overall_percentage_lof_' + filename,
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
    data.X_train, _, _ = data.standarize(data.Xraw_train)
    data.X_val, _, _ = data.standarize(data.Xraw_val)
    data.X, _, _ = data.standarize(data.Xraw)  

    # First clustering
    clustering = Clustering()
    if clustering_algorithm == 'kmeans':
        idx, centroids = clustering.run_kmeans(X=data.X,K=K)
    elif clustering_algorithm == 'DBSCAN':
        idx, K = clustering.run_DBSCAN(X=data.X,distance=distance)


    # ----------- Part 1 - Computing accuracy ----------
    print(" Damaging messages...") 
    # Artificially damage the dataset
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
            corruption = Corruption(data.X)
            ad = AnomalyDetection_LOF()
            fields = []
            messages = []
            num_messages = int(len(data.MMSI)*percentage/100)
            for n in range(num_messages):
                # Choose 0.05 or 0.1 of all messages and damage 2 their random bits
                field = np.random.choice(ad.inside_fields, size=2, replace=False)
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
            ad.detect_in_1element_clusters(
                idx=idx_corr,
                idx_vec=range(-1, np.max(idx_corr)+1),
                X=X_corr,
                message_decoded=message_decoded_corr)
            ad.detect_in_multielement_clusters(
                idx=idx_corr,
                message_decoded=message_decoded_corr,
                timestamp=data.timestamp)
            
            # Compute accuracy
            pred = np.array([ad.outliers[n][0] for n in range(len(ad.outliers))], dtype=int)
            true = np.array(corruption.indices_corrupted, dtype=int)
            OK_vec2[i,0] = recall_score(true, pred)
            OK_vec2[i,1] = precision_score(true, pred)
            OK_vec2[i,2] = np.mean(true == pred)
            recall = []
            precision =[]
            for n in range(num_messages):
                accuracy = calculate_ad_accuracy(fields[n], ad.outliers[messages[n]][2])
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
print(" With 0.05 messages damaged:")
print(" - Message indication recall: " + str(round(OK_vec_1[0],2)) + "%")
print(" - Message indication precision: " + str(round(OK_vec_1[1],2)) + "%")
print(" - Message indication accuracy: " + str(round(OK_vec_1[2],2)) + "%")
print(" - Feature indication recall: " + str(round(OK_vec_1[3],2)) + "%")
print(" - Feature indication precision: " + str(round(OK_vec_1[4],2)) + "%")
print(" With 0.1 messages damaged:")
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
    if os.path.exists('research_and_results/02d_anomaly_detection_overall_percentage_lof_'+filename):
        os.remove('research_and_results/02d_anomaly_detection_overall_percentage_lof_'+filename)
    File = h5py.File(
        'research_and_results/02d_anomaly_detection_overall_percentage_lof_'+filename, 
        mode='a'
        )
    File.create_dataset('OK_vec_1', data=OK_vec_1)
    File.create_dataset('OK_vec_2', data=OK_vec_2)
    File.close()
