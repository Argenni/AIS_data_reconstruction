# ----------- Library of functions used in anomaly detection phase of AIS message reconstruction ----------
import numpy as np
import torch
torch.manual_seed(0)
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy import signal
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import h5py
import copy
import pickle
import os
import sys
sys.path.append(".")
from utils.initialization import decode, Data
from utils.clustering import Clustering, check_cluster_assignment
from utils.miscellaneous import Corruption
from utils.anomaly_detection_cnn import ConvNet



class AnomalyDetection:
    """
    Class that introduces anomaly detection in AIS data
    """
    outliers = []  # list with anomaly detection information, shape = (num_messages, 3)
    #  (1. column - if a message is outlier, 2. column - proposed correct cluster, 3. column - possibly damaged fields)
    fields = [2,3,4,5,7,8,9,10,12]
    inside_fields = [4,5,7,8,9,10]
    _field_classifier = []
    _num_estimators = 15
    _max_depth = 5
    _k = 5
    _conv_net = []
    _sample_length = 20

    def __init__(self, data, if_visualize=False, optimize=None):
        """
        Class initializer
        Arguments: 
        data - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        difference (optional) - string deciding how to indicate the anomaly,
            'wavelet' (default - by wavelet transform) or 'std' (by standard deviation)
        if_visualize (optional) - boolean deciding whether to show training performance or not,
            default = False
        optimize (optional) - string deciding whether to optimize classifier hyperparametres or not,
	        'max_depth' or 'n_estimators', default = None
        """
        # Initialize models and necessary variables
        self.outliers = np.zeros((data.X.shape[0],3), dtype=int).tolist()
        if os.path.exists('utils/anomaly_detection_files/field_classifier.h5'):
            # If there is a file with the trained classifier saved, load it
            self._field_classifier = pickle.load(open('utils/anomaly_detection_files/field_classifier.h5', 'rb'))
        else:
            # otherwise train a classifier from scratch
            self._train_field_classifier(data)
        if os.path.exists('utils/anomaly_detection_files/convnet.h5'):
            # If there is a file with the trained convolutional network saved, load it
            self._conv_net = pickle.load(open('utils/anomaly_detection_files/convnet.h5', 'rb'))
        else:
            # otherwise train a classifier from scratch
            self._train_convnet()
        # Show some classifier parametres if allowed
        if if_visualize:
            # Calculate the accuracy of the classifiers on the training data
            variables = pickle.load(open('utils/anomaly_detection_files/standalone_clusters_inputs.h5', 'rb'))
            accuracy = []
            for i in range(len(variables[1])):
                pred = self._field_classifier[i].predict(variables[0][i])
                accuracy.append(np.mean(pred == variables[1][i]))
            print(" Average accuracy of field classifier on trainset: " + str(round(np.mean(accuracy),4)))
            variables = pickle.load(open('utils/anomaly_detection_files/convnet_inputs.h5', 'rb'))
            print(" Average accuracy of a convnet:")
            y = np.array(variables[1]).reshape(-1,1)
            pred = self._conv_net(variables[0])
            pred = np.round(pred.detach().numpy())
            print("  trainset " + str(round(np.mean(pred == y),4)) )
            y = np.array(variables[3]).reshape(-1,1)
            pred = self._conv_net(variables[2])
            pred = np.round(pred.detach().numpy())
            print("  valset: " + str(round(np.mean(pred== y),4)))
        # Optimize hyperparametres if allowed
        if optimize == 'max_depth': self._optimize_rf(data, parameter='max_depth')
        elif optimize == 'n_estimators': self._optimize_rf(data, parameter='n_estimators')
        elif optimize == 'k': self._optimize_knn(data)
    

    ### ---------------------------- Standalone clusters part ---------------------------------
    def _train_field_classifier(self, data_original):
        """ 
        Train a random forest to classify which fields of AIS message to correct
        and save it as pickle in utils/anomaly_detection_files/field_classifier.h5
        Argument: data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        # Check if the file with the field classifier inputs exist
        if not os.path.exists('utils/anomaly_detection_files/standalone_clusters_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training a classifier...")
            self._create_field_classifier_dataset(data_original=data_original)
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/standalone_clusters_inputs.h5', 'rb'))
        differences = variables[0]
        y = variables[1]
        # Train one classifier for each class
        print("  Training an anomaly detector...")
        self._field_classifier = []
        for i in range(len(y)):
            self._field_classifier.append(RandomForestClassifier(
                random_state=0,
                criterion='entropy',
                n_estimators=self._num_estimators, 
                max_depth=self._max_depth
                ).fit(differences[i],y[i]))
        print("  Complete.")
        # Save
        pickle.dump(self._field_classifier, open('utils/anomaly_detection_files/field_classifier.h5', 'ab'))
        
    def _create_field_classifier_dataset(self, data_original):
        """
        Corrupt random messages, collect the corrupted fields and their differences to create a dataset 
        that a field classifier can learn on and save it as pickle in 
        utils/anomaly_detection_files/standalone_clusters_inputs.h5
        Argument: data - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI 
        """
        data = copy.deepcopy(data_original)
        # Compose a dataset from train and val sets, keep test untouched
        message_bits = np.concatenate((data.message_bits_train, data.message_bits_val), axis=0)
        message_decoded = np.concatenate((data.message_decoded_train, data.message_decoded_val), axis=0)
        MMSI = np.concatenate((data.MMSI_train, data.MMSI_val), axis=0)
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 148])  # range of fields
        differences =  []
        y = []
        corruption = Corruption(message_bits,1)
        for field in self.fields:  # Corrupt the specified field
            differences_field = []
            y_field = []
            for message_idx in range(message_bits.shape[0]): 
                # If there is at least one message from the past
                if sum(MMSI == MMSI[message_idx])>2:
                    # Choose a bit to corrupt (based on a range of the field)
                    bit = np.random.randint(field_bits[field-1], field_bits[field]-1)
                    # Corrupt that bit in a randomly chosen message
                    message_bits_corr, message_idx = corruption.corrupt_bits(
                        message_bits=message_bits,
                        bit_idx=bit,
                        message_idx=message_idx)
                    message_decoded_corr = copy.deepcopy(message_decoded) 
                    _, _, message_decoded_0 = decode(message_bits_corr[message_idx,:])  # decode from binary             
                    message_decoded_corr[message_idx] = message_decoded_0
                    # Create a vector for anomaly detection classifier
                    X = self.compute_fields_diff(message_decoded_corr, MMSI, message_idx, field)
                    if X[0] and X[1]: 
                        differences_field.append(X)
                        # Create a ground truth vector y
                        y_field.append(int(1)) 
                    # Add negative sample - no corruption
                    X = self.compute_fields_diff(message_decoded, MMSI, message_idx, field)
                    differences_field.append(X)
                    # Create a ground truth vector y
                    y_field.append(int(0))
            # Combine all into one big dataset
            differences.append(np.array(differences_field))
            y.append(y_field)
        # Divide everything into train and val sets
        differences_train = []
        differences_val = []
        y_train = []
        y_val = []
        for field in range(len(y)):
            differences_train_0, differences_val_0, y_train_0, y_val_0 = train_test_split(
                differences[field],
                y[field],
                test_size=0.25,
                shuffle=True
            )
            differences_train.append(differences_train_0)
            differences_val.append(differences_val_0)
            y_train.append(y_train_0)
            y_val.append(y_val_0)
        # Save file with the inputs for the classifier
        variables = [differences_train, y_train, differences_val, y_val]
        pickle.dump(variables, open('utils/anomaly_detection_files/standalone_clusters_inputs.h5', 'ab'))

    def compute_fields_diff(self, message_decoded, idx, message_idx, field):
        """
        Computes the input data for field anomaly detection classifier
        (relative differencee between fields' standard deviation or wavelet transform with and without an outlier,
        the higher the difference, the more likely that field is corrupted)
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        - idx - list of indices of clusters assigned to each message, len = num_messages
        - message_idx - integer scalar, index of a potential outlier to correct
        - field - integer scalar, a field to examine
        Returns: X - list of computed differences
        """
        X = []
        with_ = message_decoded[idx == idx[message_idx],field] # All messages from the determined cluster
        without = np.delete(message_decoded,message_idx,axis=0)  # Without the corrupted message
        without = without[np.delete(idx,message_idx,axis=0) == idx[message_idx],field]
        # ------------- For all fields ---------------------
        # Compute wavelet transform difference
        if len(with_)>2: with2 = abs(with_[1:len(with_)]-with_[0:len(with_)-1]) # compute the derivative
        else: with2 = [0,0]
        scale = max(with2) # normalize
        if scale and sum(with2): with2 = with2/scale 
        with_cwt = abs(signal.cwt(with2, signal.morlet2, np.array([1,3])))
        if len(without)>1: without2 = abs(without[1:len(without)]-without[0:len(without)-1])
        else: without2 = [0]
        if scale and sum(with2): without2 = without2/scale
        without_cwt = abs(signal.cwt(without2, signal.morlet2, np.array([1,3])))
        X.append(np.abs(max(with_cwt[0,:])-max(without_cwt[0,:]))/(max(with_cwt[0,:])+1e-6)) # relative difference
        # Compute standard deviation difference
        X.append((np.abs(np.std(with_) - np.std(without)))/(np.std(with_)+1e-6))
        return X
    
    def _optimize_rf(self, data_original, parameter):
        """ 
        Choose optimal value of max_depth or n_estimators for a random forest classification
        Arguments: 
        - data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        - parameter - string indicating which parameter to optimize: 'max_depth', 'n_estimators'
        """
        # Check if the file with the field classifier inputs exist
        if not os.path.exists('utils/anomaly_detection_files/standalone_clusters_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training a classifier...")
            self._create_field_classifier_dataset(data_original=data_original)
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/standalone_clusters_inputs.h5', 'rb'))
        differences_train = variables[0]
        y_train = variables[1]
        differences_val = variables[2]
        y_val = variables[3]
        # Iterate over params to find optimal one
        params = [2, 5, 8, 10, 13, 15, 20, 30, 50, 100]
        accuracy_train = []
        accuracy_val = []
        print(" Search for optimal " + parameter + "...")
        for param in params:
            field_classifier = []
            if parameter == 'max_depth':
                for i in range(len(y_train)):
                    field_classifier.append(RandomForestClassifier(
                        random_state=0, 
                        n_estimators=self._num_estimators, 
                        max_depth=param,
                        ).fit(differences_train[i],y_train[i]))
            elif parameter == 'n_estimators':
                for i in range(len(y_train)):
                    field_classifier.append(RandomForestClassifier(
                        random_state=0, 
                        n_estimators=param, 
                        max_depth=self._max_depth,
                        ).fit(differences_train[i],y_train[i]))
            # Calculate the accuracy of the classifier on the training and validation data
            accuracies_field_train = []
            accuracies_field_val = []
            for i in range(len(y_train)):
                pred = field_classifier[i].predict(np.array(differences_train[i]))
                accuracies_field_train.append(np.mean(pred == y_train[i]))
                pred = field_classifier[i].predict(np.array(differences_val[i]))
                accuracies_field_val.append(np.mean(pred == y_val[i]))
            accuracy_train.append(np.mean(accuracies_field_train))
            accuracy_val.append(np.mean(accuracies_field_val))
        print(" Complete.")
        # Plot
        fig, ax = plt.subplots()
        ax.plot(params, accuracy_train, color='k')
        ax.plot(params, accuracy_val, color='b')
        ax.set_title("Average accuracy vs " + parameter)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Average accuracy")
        ax.legend(["Training set", "Validation set"])
        fig.show()
        # Retrain the model
        if parameter == 'max_depth':
            self._max_depth = int(input("Choose the optimal max_depth: "))
        elif parameter == 'n_estimators':
            self._num_estimators = int(input("Choose the optimal n_estimators: "))
        if os.path.exists('utils/anomaly_detection_files/standalone_clusters_field_classifier.h5'):
            os.remove('utils/anomaly_detection_files/standalone_clusters_field_classifier.h5')
        self._train_field_classifier(data_original)

    def _optimize_knn(self, data_original):
        """ 
        Choose optimal value of k for k-NN classificator
        Argument: data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        data = copy.deepcopy(data_original)
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 148])  # range of fields
        # Iterate over params to find optimal one
        params = [1,3,5,7,9]            
        accuracy = []
        print(" Search for optimal k...")
        for param in params:
            accuracy_k = []
            # Try several times
            message_bits = np.concatenate((data.message_bits_train, data.message_bits_val), axis=0)
            corruption = Corruption(message_bits,1)
            for i in range(int(message_bits.shape[0]/10)):
            # Create a corrupted dataset
                # Compose a dataset from train and val sets, keep test untouched
                message_bits = np.concatenate((data.message_bits_train, data.message_bits_val), axis=0)
                Xraw = np.concatenate((data.Xraw_train, data.Xraw_val), axis=0)
                MMSI = np.concatenate((data.MMSI_train, data.MMSI_val), axis=0)
                # Choose the field to corrupt
                field = np.random.permutation(np.array(self.fields))[0]
                # Choose a bit to corrupt (based on a range of the field)
                bit = np.random.randint(field_bits[field-1], field_bits[field]-1)
                # Corrupt that bit in a randomly chosen message
                message_bits_corr, message_idx = corruption.corrupt_bits(
                    message_bits=message_bits,
                    bit_idx=bit) 
                X_0, _, _ = decode(message_bits_corr[message_idx,:])  # decode from binary
                X_corr = copy.deepcopy(Xraw)
                X_corr[message_idx] = X_0
                X_corr, _, _ = data.normalize(X_corr)
                X_0 = X_corr[message_idx]
                X_corr = np.delete(X_corr, message_idx, axis=0)
                MMSI_corr = np.delete(MMSI,message_idx,axis=0)
            # Train k-NN classifier with different k on the given dataset
                knn_classifier = KNeighborsClassifier(
                    n_neighbors=param).fit(X_corr,MMSI_corr)
            # Calculate the accuracy of the classifier
                pred = knn_classifier.predict(X_0.reshape(1,-1))
                accuracy_k.append(pred == MMSI[message_idx])
            accuracy.append(np.mean(accuracy_k))
        print(" Complete.")
        # Plot
        fig, ax = plt.subplots()
        ax.plot(params, accuracy, color='k')
        ax.set_title("Average accuracy vs k")
        ax.set_xlabel("k")
        ax.set_ylabel("Average accuracy")
        fig.show()
        # Save the optimal kvalue
        self._k = int(input("Choose the optimal k: "))

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
            self.outliers[i][2] = self._find_damaged_fields(message_decoded, idx_new, i)
            # If around half of fields are classified abnormal, that message is not an outlier
            if len(self.outliers[i][2])>=np.floor(len(self.fields)/2): self.outliers[i][0] = 0

    def _find_standalone_clusters(self, idx, idx_vec):
        """
        Finds clusters that consist only of 1 message
        Arguments: 
        - idx - list of number of cluster assigned to each AIS message, len = num_messages
        - idx_vec - list of uniqe cluster numbers in a dataset
        Returns: indices - list with the indices of messages being the standalone clusters
        """
        idx = copy.deepcopy(idx)
        idx = np.array(idx)
        temp = []
        for i in idx_vec:
            # Count how many messages from each cluster there are
            temp.append(idx[idx==i].shape[0])
        # List clusters consisting of only 1 message 
        idx_out = np.array(idx_vec)[np.array(temp)==1]
        indices = []
        for i in idx_out:
            # Mark messages within standalone clusters
            index = np.where(idx == i)[0]
            indices.append(index[0])
        return indices
    
    def _find_correct_cluster(self, X, idx, message_idx, indices):
        """
        Finds possibly correct clusters for standalone messages using knn
        Arguments: 
        - X - numpy array, AIS feature vectors, shape = (num_messages, num_features (115))
        - idx - list of indices of clusters assigned to each message, len = num_messages
        - message_idx - integer scalar, index of a potential outlier to correct
        - indices - list of indices of potential outliers in a dataset
        Returns: integer scalar, index of a potentially correct cluster that point should belong to
        """
        # Define the right clusters for outlying point
        outliers = np.zeros((X.shape[0]),dtype=int)
        for i in indices: outliers[i] = 1
        knn = KNeighborsClassifier(n_neighbors=5)  # Find the closest 5 points to the outliers (except other outliers)
        knn.fit(X[outliers!=1,:], idx[outliers!= 1])  # the cluster that those points belong
        pred = knn.predict(X[message_idx,:].reshape(1,-1))
        return pred[0] # is potentially the right cluster for the outlier

    def _find_damaged_fields(self, message_decoded, idx, message_idx):
        """
        Define the damaged fields of a AIS message
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        - idx - list of indices of clusters assigned to each message, len = num_messages
        - message_idx - integer scalar, index of a potential outlier to correct
        Returns: list of possibly damaged fields
        """
        fields = []
        for i, field in enumerate(self.fields):
            pred = self._field_classifier[i].predict(
                np.array(self.compute_fields_diff(message_decoded, idx, message_idx, field)
                ).reshape(1,-1))[0]
            if pred: fields.append(field)
        return fields


    ### -------------------------- Inside anomalies part ------------------------------------
    def compute_cwt(self, waveform):
        """
        Compute wavelet transform for a given normalized waveform
        Argument: waveform - 1-dim numpy array representing the waveform 
        """
        cwt = np.zeros((self._sample_length))
        start = self._sample_length-(waveform.shape[0]-1)
        if waveform.shape[0]>2:
            waveform = waveform[1:waveform.shape[0]]-waveform[0:waveform.shape[0]-1] # compute the derivative
            waveform = waveform/(np.max(waveform)+1e-6) # normalize
            cwt[start:self._sample_length] = abs(signal.cwt(waveform, signal.morlet2, np.array([1,3])))[0]       
        return cwt


    def _create_convnet_dataset(self):
        """
        Create a dataset that a convolutional network can learn on by corrupting randomly chosen messages
        and save it as pickle in utils/anomaly_detection_files/convnet_inputs.h5 
        Argument: data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        file = h5py.File(name='data/Gdansk.h5', mode='r')
        data = Data(file=file)
        data.split(train_percentage=50, val_percentage=25) # split into train, val and test set
        file.close()
        # Compose a dataset from train and val sets, keep test untouched
        message_bits = []
        message_bits.append(data.message_bits_train)
        message_bits.append(data.message_bits_val)
        message_decoded = []
        message_decoded.append(data.message_decoded_train)
        message_decoded.append(data.message_decoded_val)
        Xraw = []
        Xraw.append(data.Xraw_train)
        Xraw.append(data.Xraw_val)
        MMSI = []
        MMSI.append(data.MMSI_train)
        MMSI.append(data.MMSI_val)
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 148])  # range of fields
        x = []
        x.append([])
        x.append([])
        y = []
        y.append([])
        y.append([])
        corruption = []
        corruption.append(Corruption(message_bits[0],1))
        corruption.append(Corruption(message_bits[1],1))
        clustering = Clustering() # First clustering
        idx = []
        idx.append(clustering.run_DBSCAN(X=data.normalize(Xraw[0])[0])[0])
        idx.append(clustering.run_DBSCAN(X=data.normalize(Xraw[1])[0])[0])
        for field in self.inside_fields:  # Corrupt the specified field
            for i in range(1000): # If i even add to train, if i odd to val
                stop = False
                while not stop:
                    message_idx = np.random.randint(message_bits[i%2].shape[0])
                    # If there is at least one message from the past
                    if len(np.where(np.array(MMSI[i%2]) == MMSI[i%2][message_idx])[0])>self._sample_length:
                        # Choose a bit to corrupt (based on a range of the field)
                        bit = np.random.randint(field_bits[field-1], field_bits[field]-1)
                        # Corrupt that bit in a randomly chosen message
                        message_bits_corr, _ = corruption[i%2].corrupt_bits(
                            message_bits=message_bits[i%2],
                            bit_idx=bit,
                            message_idx=message_idx)
                        message_decoded_corr = copy.deepcopy(message_decoded[i%2])
                        X_corr = copy.deepcopy(Xraw[i%2])
                        X_0, _, message_decoded_0 = decode(message_bits_corr[message_idx,:])  # decode from binary             
                        message_decoded_corr[message_idx] = message_decoded_0
                        X_corr[message_idx] = X_0
                        # cluster again to find new cluster assignment
                        idx_corr, _ = clustering.run_DBSCAN(X=data.normalize(X_corr)[0])
                        # Check if the cluster is inside a proper cluster: if so, stop searching
                        stop = check_cluster_assignment(idx[i%2], idx_corr, message_idx)
                # Create a sample - take _sample_length consecutive examples
                messages_idx = np.where(np.array(MMSI[i%2])==MMSI[i%2][message_idx])[0]
                new_message_idx = np.where(messages_idx==message_idx)[0]
                new_message_decoded = message_decoded_corr[messages_idx,:]
                new_message_decoded = new_message_decoded[:,self.inside_fields]
                start_idx = int(np.floor(new_message_idx/self._sample_length)*self._sample_length)
                stop_idx = start_idx + self._sample_length
                if stop_idx > messages_idx.shape[0]:
                    stop_idx = messages_idx.shape[0]
                    start_idx = stop_idx - self._sample_length
                X = new_message_decoded[range(start_idx, stop_idx),:]
                cwt = [self.compute_cwt(X[:,f]) for f in range(len(self.inside_fields))]
                x[i%2].append(cwt)
                # Create a ground truth vectors y
                y[i%2].append(1)
                # Create a negative sample
                new_message_decoded = message_decoded[i%2][messages_idx,:]
                new_message_decoded = new_message_decoded[:,self.inside_fields]
                X = new_message_decoded[range(start_idx, stop_idx),:]
                cwt = [self.compute_cwt(X[:,f]) for f in range(len(self.inside_fields))]
                x[i%2].append(cwt)
                y[i%2].append(0)
                #sample_messages_idx = messages_idx[range(start_idx, stop_idx)]
                #sample_message_idx = np.where(sample_messages_idx==message_idx)[0]
                #vec_field = np.zeros((len(self.inside_fields)))
                #vec_field[self.inside_fields.index(field)]=1
                #y_field[i%2].append(vec_field)
        # Save file with the inputs for the classifier
        x_train, y_train = shuffle(x[0], y[0])
        x_val, y_val = shuffle(x[1], y[1])
        variables = [x_train, y_train, x_val, y_val]
        pickle.dump(variables, open('utils/anomaly_detection_files/convnet_inputs.h5', 'ab'))

    def _train_convnet(self):
        """
        Train convolutional network for detecting anomalies in AIS data (which message is damaged)
        and save it as pickle in utils/anomaly_detection_files/convnet.h5
        Argument: data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        self._conv_net = ConvNet()
        self._conv_net = self._conv_net.float()
        # Check if the file with the training data exist
        if not os.path.exists('utils/anomaly_detection_files/convnet_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training a convolutional network...")
            self._create_convnet_dataset()
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/convnet_inputs.h5', 'rb'))
        x_train = variables[0]
        y_train = variables[1]
        x_val = variables[2]
        y_val = variables[3]
        # Set criterion and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self._conv_net.parameters(), lr=0.005)
        # Train CNN
        loss_train = []
        loss_val = []
        for epoch in range(2000):
            # Eval
            self._conv_net.eval()
            with torch.no_grad():
                pred = self._conv_net(x_val)
                loss = criterion(pred, torch.tensor(y_val, dtype=torch.float).reshape((len(y_val),-1)))
                loss_val.append(loss.detach().numpy())
            # Train
            self._conv_net.train()
            optimizer.zero_grad()
            pred = self._conv_net(x_train)
            loss = criterion(pred, torch.tensor(y_train, dtype=torch.float).reshape((len(y_train),-1)))
            loss.backward()
            optimizer.step()
            print("Epoch " + str(epoch) + ": loss " + str(loss.detach().numpy()))
            loss_train.append(loss.detach().numpy())
        print("  Complete.")
        fig, ax = plt.subplots()
        ax.plot(loss_train, color='k')
        ax.plot(loss_val, color='r')
        ax.set_title("Losses in each epoch - CNN")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Training loss", "Validation loss"])
        fig.show()
        # Save the model
        pickle.dump(self._conv_net, open('utils/anomaly_detection_files/convnet.h5', 'ab'))

    def detect_inside(self, idx, idx_vec, message_decoded):
        """
        Run the anomaly detection for messages inside proper clusters
        Arguments:
        - idx - list of number of cluster assigned to each AIS message, len = num_messages
        - idx_vec - list of uniqe cluster numbers in a dataset
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        """
        idx = copy.deepcopy(idx)
        idx = np.array(idx)
        for i in idx_vec:
            # Extract messages from each cluster
            messages_idx = np.where(idx==i)[0]
            if messages_idx.shape[0] > 2:
                # Extract the examined field values
                waveform = message_decoded[idx==i,:]
                waveform = waveform[:,self.inside_fields]
                # Analyse the waveform segment-wise
                num_segments = int(np.ceil((waveform.shape[0]-self._sample_length)/(self._sample_length/2)))
                for segment in range(num_segments+1):
                    if segment == num_segments:
                        new_range = range(segment*int(self._sample_length/2), waveform.shape[0])
                    else:
                        new_range = range(segment*int(self._sample_length/2), (segment+2)*int(self._sample_length/2))
                    new_messages_idx = messages_idx[new_range]
                    batch = waveform[new_range,:]
                    cwt = [self.compute_cwt(batch[:,field]) for field in self.inside_fields]
                    # Pass it to convolutional network and get prediction
                    pred_field, pred_mess = self._conv_net(cwt)
                    pred_field = np.round(pred_field.detach().numpy())
                    pred_mess = np.round(pred_mess.detach().numpy())
                    if sum(pred_mess) and sum(pred_field):
                        outlier = np.where(pred_mess==1)[0]
                        outlier = outlier - (self._sample_length-batch.shape[0]+1)
                        self.outliers[new_messages_idx[outlier]][0]=1
                        self.outliers[new_messages_idx[outlier]][1]=i
                        if type(self.outliers[new_messages_idx[outlier]][2])==list:
                            field = self.inside_fields[np.where(pred_field==1)[0]]
                            if field not in self.outliers[new_messages_idx[outlier]][2]:
                                self.outliers[new_messages_idx[outlier]][2].append(field)
                        else:
                            self.outliers[new_messages_idx[outlier]][2] = [field]
       

def calculate_ad_accuracy(real, predictions):
    """
    Computes the accuracy of anomaly detecion phase
    Arguments: 
    - real - list of true labels (integers)
    - predictions - array of predicted labels (integers)
    Returns: accuracies - dictionary containing the computed metrics:
    "jaccard", "hamming", "recall", "precision", "f1"
    """
    if type(predictions)!=list: predictions = [predictions] 
    # Calculate Jaccard metric 
    jaccard = len(set(real).intersection(set(predictions)))/len(set(real).union(set(predictions)))
    # Calculate Hamming metric
    real_vec = np.zeros((14), dtype=bool)
    for field in real: real_vec[field] = True
    pred_vec = np.zeros((14), dtype=bool)
    for pred in predictions: pred_vec[pred] = True
    hamming = np.mean(np.bitwise_not(np.bitwise_xor(real_vec, pred_vec)))
    # Calculate recall - percentage of how many corrupted fields were predicted
    recall = []
    for real_0 in real:
        recall.append(real_0 in predictions)
    recall = np.mean(recall)
    # Calculate precision - percentage of how many predicted fields were actually corrupted
    precision = []
    for pred_0 in predictions:
        precision.append(pred_0 in real)
    if len(precision): precision = np.mean(precision)
    else: precision = 0
    # Calculate F1 score
    if (precision+recall): f1 = 2*precision*recall/(precision+recall)
    else: f1 = 0
    # Gather all
    accuracies = {
        "jaccard": jaccard,
        "hamming": hamming,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }
    return accuracies