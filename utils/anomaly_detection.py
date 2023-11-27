# ----------- Library of functions used in anomaly detection stage of AIS message reconstruction ----------
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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
from utils.miscellaneous import Corruption, count_number


class AnomalyDetection:
    """
    Class that introduces anomaly detection in AIS data
    """
    outliers = []  # list with anomaly detection information, shape = (num_messages, 3)
    #  (1. column - if a message is outlier, 2. column - proposed correct cluster, 3. column - possibly damaged fields)
    fields = [2,3,5,7,8,9,12]
    inside_fields = [5,7,8,9]
    inside_fields2 = [2,3,12]
    _field_classifier = []
    _num_estimators_rf = 15
    _max_depth_rf = 5
    _num_estimators_xgboost = 12
    _max_depth_xgboost = 2
    _num_estimators2_rf = 15
    _max_depth2_rf = 15
    _num_estimators2_xgboost = 20
    _max_depth2_xgboost = 7
    _k = 5
    _inside_field_classifier = []
    _ad_algorithm = [] # 'rf' or 'xgboost'
    _wavelet = [] # 'morlet' or 'ricker'

    def __init__(self, data, if_visualize=False, optimize=None, ad_algorithm='xgboost', wavelet='morlet', set='test'):
        """
        Class initializer. Arguments: \n
        data - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        if_visualize (optional) - boolean deciding whether to show training performance or not,
            default = False
        optimize (optional) - string deciding whether to optimize classifier hyperparameters or not,
	        'max_depth' or 'n_estimators', default = None
        ad_algorithm (optional) - string deciding which anomaly detection classifier to use:
            'rf' or 'xgboost', default = 'xgboost'
        wavelet (optional) - string deciding which wavelet to use while computing cwt in standalone clusters analysis:
            'morlet' or 'ricker' (as available in SciPy), default = 'morlet'
        set (optional) - string indicating which part of the dataset in self.data to analyse:
            'train', 'val' or 'test', default = 'test'
        """
        # Initialize models and necessary variables
        self._ad_algorithm = ad_algorithm
        self._wavelet = wavelet
        if set == 'train': self.outliers = np.zeros((data.X_train.shape[0],3), dtype=int).tolist()
        elif set == 'val': self.outliers = np.zeros((data.X_val.shape[0],3), dtype=int).tolist()
        else: self.outliers = np.zeros((data.X.shape[0],3), dtype=int).tolist()
        if os.path.exists('utils/anomaly_detection_files/standalone_'+wavelet+'_field_classifier_'+ad_algorithm+'.h5'):
            # If there is a file with the trained standalone clusters field classifier saved, load it
            self._field_classifier = pickle.load(open('utils/anomaly_detection_files/standalone_'+wavelet+'_field_classifier_'+ad_algorithm+'.h5', 'rb'))
        else:
            # otherwise train a classifier from scratch
            self._train_field_classifier(data)
        if os.path.exists('utils/anomaly_detection_files/inside_field_classifier_'+ad_algorithm+'.h5'):
            # If there is a file with the trained inside clusters field classifier saved, load it
            self._inside_field_classifier = pickle.load(open('utils/anomaly_detection_files/inside_field_classifier_'+ad_algorithm+'.h5', 'rb'))
        else:
            # otherwise train a classifier from scratch
            self._train_inside_field_classifier()
        # Show some classifier metrics if allowed
        if if_visualize:
            # Calculate the accuracy of the classifiers on the training data
            variables = pickle.load(open('utils/anomaly_detection_files/standalone_'+wavelet+'_inputs.h5', 'rb'))
            print(" Average accuracy of standalone clusters field classifier:")
            accuracy = []
            for i in range(len(variables[1])):
                pred = self._field_classifier[i].predict(variables[0][i])
                accuracy.append(np.mean(pred == variables[1][i]))          
            print("  trainset: " + str(round(np.mean(accuracy),4)))
            accuracy = []
            for i in range(len(variables[3])):
                pred = self._field_classifier[i].predict(variables[2][i])
                accuracy.append(np.mean(pred == variables[3][i]))
            print("  valset: " + str(round(np.mean(accuracy),4)))
            variables = pickle.load(open('utils/anomaly_detection_files/inside_field_classifier_inputs.h5', 'rb'))
            print(" Average accuracy of inside clusters field classifier:")
            y = np.array(variables[1])
            accuracy = []
            for i in range(len(variables[1])):
                pred = self._inside_field_classifier[i].predict(variables[0][i])
                accuracy.append(np.mean(pred == y))
            print("  trainset " + str(round(np.mean(accuracy),4)) )
            y = np.array(variables[3])
            accuracy = []
            for i in range(len(variables[3])):
                pred = self._inside_field_classifier[i].predict(variables[2][i])
                accuracy.append(np.mean(pred == variables[3][i]))
            print("  valset " + str(round(np.mean(accuracy),4)) )
        # Optimize hyperparametres if allowed
        if optimize == 'max_depth': self._optimize_standalone_cluster_classifier(data, hyperparameter='max_depth')
        elif optimize == 'n_estimators': self._optimize_standalone_cluster_classifier(data, hyperparameter='n_estimators')
        elif optimize == 'k': self._optimize_knn(data)
        elif optimize == 'max_depth2': self._optimize_inside_field_classifier(hyperparameter='max_depth')
        elif optimize == 'n_estimators2': self._optimize_inside_field_classifier(hyperparameter='n_estimators')
    

    ### ---------------------------- Standalone clusters part ---------------------------------
    def _train_field_classifier(self, data_original):
        """ 
        Train a random forest or xgboost to classify which fields of AIS message to correct
        and save it as pickle in utils/anomaly_detection_files/field_classifier.h5
        Argument: data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        # Check if the file with the field classifier inputs exist
        if not os.path.exists('utils/anomaly_detection_files/standalone_'+self._wavelet+'_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training a classifier...")
            self._create_field_classifier_dataset(data_original=data_original)
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/standalone_'+self._wavelet+'_inputs.h5', 'rb'))
        differences = variables[0]
        y = variables[1]
        # Train one classifier for each class
        print("  Training an anomaly detector...")
        self._field_classifier = []
        for i in range(len(y)):
            if self._ad_algorithm == 'rf':
                self._field_classifier.append(RandomForestClassifier(
                    random_state=0,
                    criterion='entropy',
                    n_estimators=self._num_estimators_rf, 
                    max_depth=self._max_depth_rf
                    ).fit(differences[i],y[i]))
            else:  
                self._field_classifier.append(XGBClassifier(
                    random_state=0,
                    n_estimators=self._num_estimators_xgboost, 
                    max_depth=int(np.floor(self._max_depth_xgboost))
                    ).fit(differences[i],y[i]))
        print("  Complete.")
        # Save
        pickle.dump(self._field_classifier, open('utils/anomaly_detection_files/standalone_'+self._wavelet+'_field_classifier_'+self._ad_algorithm+'.h5', 'ab'))
        
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
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
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
        pickle.dump(variables, open('utils/anomaly_detection_files/standalone_'+self._wavelet+'_inputs.h5', 'ab'))

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
        if self._wavelet == 'morlet': without_cwt = abs(signal.cwt(without2, signal.morlet2, np.array([1,3])))
        elif self._wavelet == 'ricker': without_cwt = abs(signal.cwt(without2, signal.ricker, np.array([1,3])))
        X.append(np.abs(max(with_cwt[0,:])-max(without_cwt[0,:]))/(max(with_cwt[0,:])+1e-6)) # relative difference
        # Compute standard deviation difference
        X.append((np.abs(np.std(with_) - np.std(without)))/(np.std(with_)+1e-6))
        return X
    
    def _optimize_standalone_cluster_classifier(self, data_original, hyperparameter):
        """ 
        Choose optimal value of max_depth or n_estimators for a Random Forest/XGBoost classifier
        for standalone clusters
        Arguments: 
        - data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        - hyperparameter - string indicating which hyperparameter to optimize: 'max_depth' or 'n_estimators'
        """
        # Check if the file with the field classifier inputs exist
        if not os.path.exists('utils/anomaly_detection_files/standalone_'+self._wavelet+'_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training a classifier...")
            self._create_field_classifier_dataset(data_original=data_original)
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/standalone_'+self._wavelet+'_inputs.h5', 'rb'))
        differences_train = variables[0]
        y_train = variables[1]
        differences_val = variables[2]
        y_val = variables[3]
        # Iterate over params to find optimal one
        params = [2, 5, 8, 10, 13, 15, 20, 30, 50, 100]
        accuracy_train = []
        accuracy_val = []
        print(" Search for optimal " + hyperparameter + "...")
        for param in params:
            field_classifier = []
            if hyperparameter=='max_depth' and self._ad_algorithm=='rf':
                for i in range(len(y_train)):
                    field_classifier.append(RandomForestClassifier(
                        random_state=0, 
                        n_estimators=self._num_estimators_rf, 
                        max_depth=param,
                        ).fit(differences_train[i],y_train[i]))
            elif hyperparameter == 'n_estimators' and self._ad_algorithm=='rf':
                for i in range(len(y_train)):
                    field_classifier.append(RandomForestClassifier(
                        random_state=0, 
                        n_estimators=param, 
                        max_depth=self._max_depth_rf,
                        ).fit(differences_train[i],y_train[i]))
            elif hyperparameter=='max_depth' and self._ad_algorithm=='xgboost':
                for i in range(len(y_train)):
                    field_classifier.append(XGBClassifier(
                        random_state=0, 
                        n_estimators=self._num_estimators_xgboost, 
                        max_depth=param,
                        ).fit(differences_train[i],y_train[i]))
            elif hyperparameter == 'n_estimators' and self._ad_algorithm=='xgboost':
                for i in range(len(y_train)):
                    field_classifier.append(XGBClassifier(
                        random_state=0, 
                        n_estimators=param, 
                        max_depth=self._max_depth_xgboost,
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
        ax.set_title("Average accuracy vs " + hyperparameter)
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel("Average accuracy")
        ax.legend(["Training set", "Validation set"])
        fig.show()
        # Retrain the model
        if hyperparameter == 'max_depth':
            self._max_depth = int(input("Choose the optimal max_depth: "))
        elif hyperparameter == 'n_estimators':
            self._num_estimators = int(input("Choose the optimal n_estimators: "))
        if os.path.exists('utils/anomaly_detection_files/standalone_'+self._wavelet+'_field_classifier_'+self._ad_algorithm+'.h5'):
            os.remove('utils/anomaly_detection_files/standalone_'+self._wavelet+'_field_classifier_'+self._ad_algorithm+'.h5')
        self._train_field_classifier(data_original)

    def _optimize_knn(self, data_original):
        """ 
        Choose optimal value of k for k-NN classifier for standalone clusters
        Argument: data_original - object of a Data class, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        data = copy.deepcopy(data_original)
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
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
    def compute_inside_sample(self, message_decoded, MMSI, message_idx, timestamp, field):
        """
        Computes the input data for field anomaly detection classifier inside proper clusters
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        - MMSI - list of MMSI identifier from each AIS message, len = num_messages
        - message_idx - integer scalar, index of a potential outlier to correct
        - timestamp - list of strings with timestamp of each message, len = num_messages
        - field - integer scalar, a field to examine
        Returns: X - list of computed differences
        Argument: sample - a vectoer descibing a message for anomaly detection, np.zeros((12 or 9))
        """
        sample = np.zeros((12))
        # Select 3 consecutive samples
        messages_idx = np.where(np.array(MMSI)==MMSI[message_idx])[0]
        new_message_idx = np.where(messages_idx==message_idx)[0][0]
        if new_message_idx>0: previous_idx = messages_idx[new_message_idx-1]
        else: previous_idx=-1
        if new_message_idx<messages_idx.shape[0]-1: next_idx = messages_idx[new_message_idx+1]
        else: next_idx=-1          
        if field == 9: 
            sample = np.zeros((9))
            delta_lon_deg1 = message_decoded[next_idx, 7]-message_decoded[message_idx, 7]
            delta_lon_deg2 = message_decoded[message_idx, 7]-message_decoded[previous_idx, 7]
            delta_lat_deg1 = message_decoded[next_idx, 8]-message_decoded[message_idx, 8]
            delta_lat_deg2 = message_decoded[message_idx, 8]-message_decoded[previous_idx, 8]
            if delta_lon_deg1 !=0:
                sample[0] = np.arctan(delta_lat_deg1/delta_lon_deg1)/np.pi*180
                if delta_lon_deg1<0: cart = np.sign(delta_lat_deg1)*180-np.arctan(delta_lat_deg1/abs(delta_lon_deg1))/np.pi*180
                elif delta_lon_deg1>0: cart = np.arctan(delta_lat_deg1/delta_lon_deg1)/np.pi*180
            else: cart = 90 # delta_lon_deg = 0
            course = np.mod(90-cart,360)
            sample[2] = abs(course - message_decoded[message_idx, 9])
            if delta_lon_deg2 !=0:
                sample[1] = np.arctan(delta_lat_deg2/delta_lon_deg2)/np.pi*180
                if delta_lon_deg2<0: cart = np.sign(delta_lat_deg2)*180-np.arctan(delta_lat_deg2/abs(delta_lon_deg2))/np.pi*180
                elif delta_lon_deg2>0: cart = np.arctan(delta_lat_deg2/delta_lon_deg2)/np.pi*180
            else: cart = 90
            course = np.mod(90-cart,360)
            sample[3] = abs(course - message_decoded[message_idx, 9])
            if previous_idx!=-1:
                sample[4] = ((timestamp[message_idx]-timestamp[previous_idx]).seconds)/60
                sample[5] = message_decoded[previous_idx,9]/360
            sample[6] = message_decoded[message_idx,9]/360
            if next_idx!=-1:
                sample[7] = ((timestamp[next_idx]-timestamp[message_idx]).seconds)/60
                sample[8] = message_decoded[next_idx,9]/360
        else: 
            sample = np.zeros((12))
            if previous_idx!=-1:
                sample[0] = ((timestamp[message_idx]-timestamp[previous_idx]).seconds)/60 # timestamp difference
                sample[1] = (message_decoded[message_idx,7]-message_decoded[previous_idx,7])/180 # longitude difference
                sample[2] = (message_decoded[message_idx,8]-message_decoded[previous_idx,8])/90 # latitude difference
                sample[3] = message_decoded[previous_idx,5]/102.2 # speed value
                sample[4] = message_decoded[previous_idx,9]/360 # course value
            sample[5] = message_decoded[message_idx,5]/102.2
            sample[6] = message_decoded[message_idx,9]/360
            if next_idx!=-1:
                sample[7] = ((timestamp[next_idx]-timestamp[message_idx]).seconds)/60
                sample[8] = (message_decoded[next_idx,7]-message_decoded[message_idx,7])/180
                sample[9] = (message_decoded[next_idx,8]-message_decoded[message_idx,8])/90
                sample[10] = message_decoded[next_idx,5]/102.2
                sample[11] = message_decoded[next_idx,9]/360
        return sample


    def _create_inside_field_classifier_dataset(self):
        """
        Create a dataset that a inside field classifier can learn on by corrupting randomly chosen messages
        and save it as pickle in utils/anomaly_detection_files/inside_field_classifier_inputs.h5 
        Requires Baltic.h5 and Gibraltar.h5 files, containing all 3 datasets (train, val, test) with:
          X, Xraw, message_bits, message_decoded, MMSI
        """
        file = h5py.File(name='data/Baltic.h5', mode='r')
        data1 = Data(file=file)
        data1.split(train_percentage=50, val_percentage=25) # split into train, val and test set
        file.close()
        file = h5py.File(name='data/Gibraltar.h5', mode='r')
        data2 = Data(file=file)
        data2.split(train_percentage=50, val_percentage=25) # split into train, val and test set
        file.close()
        # Compose a dataset from train and val sets, keep test untouched
        message_bits = []
        message_bits.append(np.concatenate((data1.message_bits_train, data2.message_bits_train), axis=0))
        message_bits.append(np.concatenate((data1.message_bits_val, data2.message_bits_val), axis=0))
        message_decoded = []
        message_decoded.append(np.concatenate((data1.message_decoded_train, data2.message_decoded_train), axis=0))
        message_decoded.append(np.concatenate((data1.message_decoded_val, data2.message_decoded_val), axis=0))
        MMSI = []
        MMSI.append(np.concatenate((data1.MMSI_train, data2.MMSI_train), axis=0))
        MMSI.append(np.concatenate((data1.MMSI_val, data2.MMSI_val), axis=0))
        timestamp = []
        timestamp.append(np.concatenate((data1.timestamp_train, data2.timestamp_train), axis=0))
        timestamp.append(np.concatenate((data1.timestamp_val, data2.timestamp_val), axis=0))
        field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
        x = []
        x.append([]) # x[0] - train
        x.append([]) # x[1] - val
        y = []
        y.append([])
        y.append([])
        corruption = []
        corruption.append(Corruption(message_bits[0],1))
        corruption.append(Corruption(message_bits[1],1))
        for i in [0,1]: # If i==0 add to train, if i==1 to val
            for field in self.inside_fields:
                samples_field = []
                y_field = []
                for message_idx in range(message_bits[i].shape[0]):
                    # If there is at least one message from the past
                    if len(np.where(np.array(MMSI[i]) == MMSI[i][message_idx])[0])>2:
                        # Choose a bit to corrupt (based on a range of the field)
                        bit_idx = np.random.randint(field_bits[field-1], field_bits[field]-1)
                        # Corrupt that bit (or two bits if message_idx is odd)
                        message_bits_corr, _ = corruption[i].corrupt_bits(
                            message_bits=message_bits[i],
                            bit_idx=bit_idx,
                            message_idx=message_idx)
                        if message_idx%2: # For odd idx, corrupt another bit
                            new_fields = copy.deepcopy(self.inside_fields)
                            new_fields.remove(field)
                            new_field = np.random.choice(new_fields)
                            new_bit_idx = np.random.randint(field_bits[new_field-1], field_bits[new_field]-1)
                            message_bits_corr, _ = corruption[i].corrupt_bits(
                                message_bits=message_bits_corr,
                                bit_idx=new_bit_idx,
                                message_idx=message_idx)
                        message_decoded_corr = copy.deepcopy(message_decoded[i])
                        _, _, message_decoded_0 = decode(message_bits_corr[message_idx,:])  # decode from binary             
                        message_decoded_corr[message_idx] = message_decoded_0
                        # Create a sample - take 3 consecutive examples
                        sample = self.compute_inside_sample(
                            message_decoded=message_decoded_corr,
                            MMSI=MMSI[i],
                            message_idx=message_idx,
                            timestamp=timestamp[i],
                            field=field)
                        samples_field.append(sample)
                        y_field.append(1) # create a ground truth vector y
                        # Create some negative (no corruption) samples
                        if message_idx%6:
                            sample = self.compute_inside_sample(
                                message_decoded=message_decoded[i],
                                MMSI=MMSI[i],
                                message_idx=message_idx,
                                timestamp=timestamp[i],
                                field=field)
                            samples_field.append(sample)
                            y_field.append(0)
                x[i].append(samples_field)
                y[i].append(y_field)
        # Save file with the inputs for the classifier
        variables = [x[0], y[0], x[1], y[1]]
        pickle.dump(variables, open('utils/anomaly_detection_files/inside_field_classifier_inputs.h5', 'ab'))

    def _train_inside_field_classifier(self):
        """
        Train inside field classifier for detecting anomalies in AIS data (which message is damaged)
        and save it as pickle in utils/anomaly_detection_files/inside_field_classifier.h5
        """
        # Check if the file with the training data exist
        if not os.path.exists('utils/anomaly_detection_files/inside_field_classifier_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training inside field classifier...")
            self._create_inside_field_classifier_dataset()
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/inside_field_classifier_inputs.h5', 'rb'))
        for i in range(len(self.inside_fields)):
            if self._ad_algorithm=='rf':
                if self.inside_fields[i] != 9:
                    self._inside_field_classifier.append(RandomForestClassifier(
                        random_state=0,
                        criterion='entropy',
                        n_estimators=self._num_estimators2_rf, 
                        max_depth=self._max_depth2_rf
                        ).fit(variables[0][i],variables[1][i]))
                else:
                    self._inside_field_classifier.append(RandomForestClassifier(
                        random_state=0,
                        criterion='entropy',
                        n_estimators=self._num_estimators2_rf, 
                        max_depth=int(np.floor(0.8*self._max_depth2_rf))
                        ).fit(variables[0][i],variables[1][i]))
            else:
                if self.inside_fields[i] != 9:
                    self._inside_field_classifier.append(XGBClassifier(
                        random_state=0,
                        n_estimators=self._num_estimators2_xgboost, 
                        max_depth=self._max_depth2_xgboost
                        ).fit(variables[0][i],variables[1][i]))
                else:
                    self._inside_field_classifier.append(XGBClassifier(
                        random_state=0,
                        n_estimators=self._num_estimators2_xgboost, 
                        max_depth=int(np.floor(0.8*self._max_depth2_xgboost))
                        ).fit(variables[0][i],variables[1][i]))
        # Save the model
        pickle.dump(self._inside_field_classifier, open('utils/anomaly_detection_files/inside_field_classifier_'+self._ad_algorithm+'.h5', 'ab'))

    def _optimize_inside_field_classifier(self, hyperparameter):
        """ 
        Choose optimal value of max_depth or n_estimators for a Random Forest/XGBoost classifier
        for detecting damaged fields inside proper clusters
        Argument: hyperparameter - string indicating which hyperparameter to optimize: 'max_depth', 'n_estimators'
        """
        # Check if the file with the field classifier inputs exist
        if not os.path.exists('utils/anomaly_detection_files/inside_field_classifier_inputs.h5'):
            # if not, create a corrupted dataset
            print("  Preparing for training a classifier...")
            self._create_inside_field_classifier_dataset()
            print("  Complete.")
        variables = pickle.load(open('utils/anomaly_detection_files/inside_field_classifier_inputs.h5', 'rb'))
        x_train = variables[0]
        y_train = variables[1]
        x_val = variables[2]
        y_val = variables[3]
        # Iterate over params to find optimal one
        params = [2, 5, 8, 10, 13, 15, 20, 30, 50, 100]
        accuracy_train = []
        accuracy_train_course = []
        accuracy_val = []
        accuracy_val_course = []
        print(" Search for optimal " + hyperparameter + "...")
        for param in params:
            field_classifier = []
            if hyperparameter=='max_depth' and self._ad_algorithm=='rf':
                for i in range(len(y_train)-1):
                    field_classifier.append(RandomForestClassifier(
                        random_state=0, 
                        n_estimators=self._num_estimators2_rf, 
                        max_depth=param,
                        ).fit(x_train[i],y_train[i]))
                field_classifier.append(RandomForestClassifier(
                    random_state=0, 
                    n_estimators=self._num_estimators2_rf, 
                    max_depth=param,
                    ).fit(x_train[i+1],y_train[i+1]))
            elif hyperparameter=='n_estimators' and self._ad_algorithm=='rf':
                for i in range(len(y_train)-1):
                    field_classifier.append(RandomForestClassifier(
                        random_state=0, 
                        n_estimators=param, 
                        max_depth=self._max_depth2_rf,
                        ).fit(x_train[i],y_train[i]))
                field_classifier.append(RandomForestClassifier(
                    random_state=0, 
                    n_estimators=param, 
                    max_depth=int(np.floor(0.8*self._max_depth2_rf)),
                    ).fit(x_train[i+1],y_train[i+1]))
            elif hyperparameter=='max_depth' and self._ad_algorithm=='xgboost':
                for i in range(len(y_train)-1):
                    field_classifier.append(XGBClassifier(
                        random_state=0, 
                        n_estimators=self._num_estimators2_xgboost, 
                        max_depth=param,
                        ).fit(x_train[i],y_train[i]))
                field_classifier.append(XGBClassifier(
                    random_state=0, 
                    n_estimators=self._num_estimators2_xgboost, 
                    max_depth=param,
                    ).fit(x_train[i+1],y_train[i+1]))
            elif hyperparameter=='n_estimators' and self._ad_algorithm=='xgboost':
                for i in range(len(y_train)-1):
                    field_classifier.append(XGBClassifier(
                        random_state=0, 
                        n_estimators=param, 
                        max_depth=self._max_depth2_xgboost,
                        ).fit(x_train[i],y_train[i]))
                field_classifier.append(XGBClassifier(
                    random_state=0, 
                    n_estimators=param, 
                    max_depth=int(np.floor(0.8*self._max_depth2_xgboost)),
                    ).fit(x_train[i+1],y_train[i+1]))
            # Calculate the accuracy of the classifier on the training and validation data
            accuracies_field_train = []
            accuracies_field_val = []
            for i in range(len(y_train)):
                pred = field_classifier[i].predict(np.array(x_train[i]))
                accuracies_field_train.append(f1_score(y_train[i], pred))
                pred = field_classifier[i].predict(np.array(x_val[i]))
                accuracies_field_val.append(f1_score(y_val[i], pred))
            accuracy_train.append(np.mean(accuracies_field_train[:-1]))
            accuracy_train_course.append(accuracies_field_train[-1])
            accuracy_val.append(np.mean(accuracies_field_val[:-1]))
            accuracy_val_course.append(accuracies_field_val[-1])
        print(" Complete.")
        # Plot
        fig, ax = plt.subplots()
        ax.plot(params, accuracy_train, color='k')
        ax.plot(params, accuracy_val, color='b')
        ax.plot(params, accuracy_train_course, color='r')
        ax.plot(params, accuracy_val_course, color='g')
        ax.set_title("Average f1 vs " + hyperparameter)
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel("Average f1")
        ax.legend(["Training set - fields 5,7,8", "Validation set - fields 5,7,8",
                   "Training set - field 9", "Validation set - field 9" ])
        fig.show()
        # Retrain the model
        if hyperparameter == 'max_depth':
            if self._ad_algorithm=='rf': self._max_depth2_rf = int(input("Choose the optimal max_depth: "))
            elif self._ad_algorithm=='xgboost': self._max_depth2_xgboost = int(input("Choose the optimal max_depth: "))
        elif hyperparameter == 'n_estimators':
            if self._ad_algorithm=='rf': self._num_estimators2_rf = int(input("Choose the optimal n_estimators: "))
            elif self._ad_algorithm=='xgboost': self._num_estimators2_xgboost = int(input("Choose the optimal n_estimators: "))
        if os.path.exists('utils/anomaly_detection_files/inside_field_classifier_'+self._ad_algorithm+'.h5'):
            os.remove('utils/anomaly_detection_files/inside_field_classifier_'+self._ad_algorithm+'.h5')
        self._train_inside_field_classifier()

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
                        clf = IsolationForest(random_state=0).fit(waveform)
                        pred2 = clf.predict(waveform)
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
            pred.append(np.round(self._inside_field_classifier[field].predict(samples)))
        for message_idx in range(message_decoded.shape[0]):
            if len(np.where(np.array(idx)==idx[message_idx])[0])>2:
                fields = []
                for i in range(len(self.inside_fields)):
                    if pred[i][message_idx]: fields.append(self.inside_fields[i])
                if len(fields):
                    self.outliers[message_idx][0] = 1
                    self.outliers[message_idx][1] = idx[message_idx]
                    if self.outliers[message_idx][2]==0: self.outliers[message_idx][2] = fields
                    else: self.outliers[message_idx][2] = self.outliers[message_idx][2] + fields

       
def calculate_ad_accuracy(real, predictions):
    """
    Computes the accuracy of anomaly detecion phase
    Arguments: 
    - real - list of true labels (integers)
    - predictions - list of predicted labels (integers)
    Returns: accuracies - dictionary containing the computed metrics:
    "jaccard", "hamming", "recall", "precision", "f1"
    """
    if type(predictions)==0: predictions = [] 
    if type(predictions)==list:
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
            "f1": f1 }
        return accuracies
    elif predictions==0: return {"jaccard": 0, "hamming": 0, "recall": 0, "precision": 0, "f1": 0}