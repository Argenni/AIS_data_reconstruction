# ----------- Library of functions used in research regarding of AIS message reconstruction ----------
# --------------------- not necessarily in the main pipeline -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import copy
import sys
sys.path.append(".")
from utils.miscellaneous import count_number
from utils.anomaly_detection import AnomalyDetection
from sklearn.neighbors import LocalOutlierFactor



# ----------------------- For clustering phase ------------------------------
def check_cluster_assignment(idx, idx_corr, message_idx):
    """
    Checks if the corrupted message is assigned together with other messages from its vessel
    Arguments:
    - idx - list of indices of clusters assigned to each message, len = num_messages,
    - idx_corr - list of indices of clusters assigned to each message in a corrupted dataset, 
        len = num_messages
    - message_idx - integer scalar, index of a message that was corrupted
    """
    idx_before = idx[message_idx]
    idx_now = idx_corr[message_idx]
    # Find all messages originally clustered with the corrupted message
    indices_original = np.where(idx == idx_before)
    # Find a cluster that contains most of those messages after the corruption
    percentage = []
    _, idx_corr_vec = count_number(idx_corr)
    for i in idx_corr_vec:  # for each cluster in corrupted data
        indices_cluster = np.where(idx_corr == i)  # find messages from that cluster
        intersection = set(indices_original[0]).intersection(indices_cluster[0])  # find messages both in original cluster and examined cluster
        percentage.append(len(intersection)/len(indices_original[0]))  # calculate how many messages from the original cluster are in examined cluster
    idx_preferable = idx_corr_vec[percentage.index(max(percentage))]  # the cluster with the biggest percentage is probably the right one
    # Check if that cluster is the same as before
    result = idx_now == idx_preferable
    return result



# ------------------- For anomaly detection phase ---------------------------
def visualize_corrupted_bits(OK_vec_all, titles):
    """
    Plots the results of damaging certain bits in AIS message
    Arguments: 
    - OK_vec_all - numpy array containing percentages of correctness regarding each bit
        shape = (num_subplots, num_bits)
    - titles - dictionary with titles for each subplot,
        keys are the number of a subplot {'0':"title_for_first_subplot", ...}
    """
    bits = list(range(145))  # create a list of meaningful bits to visualize
    bits.append(148)
    fig, ax = plt.subplots(OK_vec_all.shape[0], sharex=True, sharey=True)
    for i in range(OK_vec_all.shape[0]):
        OK_vec = OK_vec_all[i, :]
        ax[i].set_title(titles[str(i)])  # get titles from the dictionary
        # Plot each meesage fields with a different color - other bits are 0s
        ax[i].bar(bits, np.concatenate((OK_vec[0:6], np.zeros((140))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((6)), OK_vec[6:8], np.zeros((138))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((8)), OK_vec[8:38], np.zeros((108))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((38)), OK_vec[38:42], np.zeros((104))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((42)), OK_vec[42:50], np.zeros((96))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((50)), OK_vec[50:60], np.zeros((86))), axis=0))
        temp = np.zeros(146)
        temp[60] = OK_vec[60]
        ax[i].bar(bits, temp)
        ax[i].bar(bits, np.concatenate((np.zeros((61)), OK_vec[61:89], np.zeros((57))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((89)), OK_vec[89:116], np.zeros((30))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((116)), OK_vec[116:128], np.zeros((18))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((128)), OK_vec[128:137], np.zeros((9))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((137)), OK_vec[137:143], np.zeros((3))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((143)), OK_vec[143:145], np.zeros((1))), axis=0))
        temp = np.zeros(146)
        temp[145] = OK_vec[145]
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

class AnomalyDetection_LOF(AnomalyDetection):
    """Class that inherits from Anomaly Detection, for checking the performance of LOF in this stage"""
    
    def __init__(self, data, wavelet = 'morlet', set='test'):
        self._wavelet = wavelet
        if set == 'train': self.outliers = np.zeros((data.X_train.shape[0],3), dtype=int).tolist()
        elif set == 'val': self.outliers = np.zeros((data.X_val.shape[0],3), dtype=int).tolist()
        else: self.outliers = np.zeros((data.X.shape[0],3), dtype=int).tolist()
        self.k = 5

    def _train_field_classifier(self): pass
    def _create_field_classifier_dataset(self): pass
    def _optimize_standalone_cluster_classifier(self): pass
    def _optimize_knn(self): pass
    def _create_inside_field_classifier_dataset(self): pass
    def _train_inside_field_classifier(self): pass
    def _optimize_inside_field_classifier(self): pass
    def _find_damaged_fields(self): pass

    def detect_standalone_clusters(self, idx, idx_vec, X, message_decoded):
        """
        Run the entire anomaly detection based on searching for standalone clusters
        Arguments:
        - idx - list of number of cluster assigned to each AIS message, len = num_messages
        - idx_vec - list of uniqe cluster numbers in a dataset
        - X - numpy array, AIS feature vectors, shape = (num_messages, num_features (115))
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        """
        # Find standalone clusters
        indices = self._find_standalone_clusters(idx, idx_vec)
        for i in indices:
            idx_new = copy.deepcopy(idx)
            # Mark those points as outliers
            self.outliers[i][0] = 1
            # Find the correct clusters for that points
            self.outliers[i][1] = self._find_correct_cluster(X, idx_new, i, indices)
            idx_new[i] = self.outliers[i][1]
            # Find the damaged fields to correct
            messages_idx = (np.where(np.array(idx)==idx_new[i])[0]).tolist()
            message_idx_new = messages_idx.index(i)
            samples = []
            for field in self.fields:
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
            if len(self.outliers[i][2])>=np.floor(len(self.fields)/2): self.outliers[i][0] = 0

    def detect_inside(self, idx, message_decoded, timestamp):
        """
        Run the anomaly detection for messages inside proper clusters
        Arguments:
        - idx - list of number of cluster assigned to each AIS message, len = num_messages
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        - timestamp - list of strings with timestamp of each message, len = num_messages
        """
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
                samples.append(self.compute_inside_sample(message_decoded, idx, message_idx, timestamp, self.inside_fields[field]))
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


