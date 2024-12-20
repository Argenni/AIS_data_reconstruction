"""
Functions and classes used in prediction stage of AIS message reconstruction
"""

import numpy as np
from xgboost import XGBRegressor
import statsmodels.api as sm
import matplotlib.pyplot as plt
params = {'axes.labelsize': 16,'axes.titlesize':16, 'font.size': 16, 'legend.fontsize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
plt.rcParams.update(params)
import copy
import h5py
import pickle
import os
import sys
sys.path.append(".")
from utils.initialization import Data, encode
from utils.miscellaneous import count_number

class Prediction:
    """
    Class that introduces prediction phase in reconstruction of AIS data.
    """
    predictions = []
    fields = [2,3,5,7,8,9,12]
    fields_dynamic = [5,7,8,9]
    fields_static = [2,3,12]
    _decimals = [1,4,4,1]
    _prediction_algorithm = 'xgboost'
    _regressor = []
    _max_depth = 15
    _num_estimators = 20
    _lags = 1
    _language = [] # 'pl' or 'eng' - for graphics only
    _verbose = []
    
    def __init__(self, verbose=False, optimize=None, prediction_algorithm='xgboost', language='eng'):
        """
        Class initialization (class object creation). Arguments:
        - verbose (optional) - Boolean, whether to print running logs or not, default=False,
        - optimize (optional) - string, name of regressor hyperparameter to optimize, 
            'max_depth', 'n_estimators' (for xgboost), 'lags' (for autoregression), default=None (no optimization),
        - prediction_algorithm (optional) - string deciding which model to use, 'xgboost' or 'ar', default='xgboost'.
        - language - string, 'pl' for Polish or 'eng' for English (only for graphics text translation).
        """
        # Initialize models and necessary variables
        self._prediction_algorithm = prediction_algorithm
        self._verbose = verbose
        self._language = language
        if self._prediction_algorithm == 'xgboost':
            if os.path.exists('utils/prediction_files/regressor_'+prediction_algorithm+'.h5'):
                # If there is a file with the trained regressor saved, load it
                self._regressor = pickle.load(open('utils/prediction_files/regressor_'+prediction_algorithm+'.h5', 'rb'))
            else:
                # otherwise train a regressor from scratch
                self._train_regressor()
        # Optimize hyperparametres if allowed
        if optimize == 'max_depth': self._optimize_regression(hyperparameter='max_depth')
        elif optimize == 'n_estimators': self._optimize_regression(hyperparameter='n_estimators')
        elif optimize == 'lags': self._optimize_regression(hyperparameter='lags')
        # Show some regressor metrics if allowed
        if self._verbose:
            # Calculate the performance of the regressor on training and validation data
            if not os.path.exists('utils/prediction_files/dataset_'+prediction_algorithm+'.h5'):
                self._create_regression_dataset()
            variables = pickle.load(open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'rb'))
            print(" Average SMAE of regressor:")
            mae = []
            for field_num in range(len(variables[1])):
                if self._prediction_algorithm == 'xgboost': 
                    pred = self._regressor[field_num].predict(variables[0][field_num])
                    mae.append(calculate_SMAE(pred, variables[1][field_num], self.fields[field_num]))
                elif self._prediction_algorithm == 'ar':
                    y_true = []
                    y_pred = []
                    for trajectory in range(len(variables[1][field_num])):
                        pred = self._validate_prediction(
                            pred=self._predict_ar(variables[0][field_num][trajectory], self._lags, self.fields[field_num]),
                            field=self.fields[field_num])
                        y_true.append(variables[1][field_num][trajectory])
                        y_pred.append(pred)
                    mae.append(calculate_SMAE(y_pred, y_true, self.fields[field_num]))
            print("  trainset: " + str(round(np.mean(mae),4)))
            if self._prediction_algorithm == 'xgboost': 
                mae = []
                for field_num in range(len(variables[3])):
                    pred = self._regressor[field_num].predict(variables[2][field_num])
                    mae.append(calculate_SMAE(pred, variables[3][field_num], self.fields[field_num]))
                print("  valset: " + str(round(np.mean(mae),4)))

    def _train_regressor(self):
        """
        Trains a xgboost regressor that will be used in prediction phase of AIS data reconstruction
        and saves it as pickle in utils/prediction_files/regressor_.h5 and in self._regressor.
        """
        if self._prediction_algorithm == 'xgboost':
            # Check if the file with the regressor inputs exist
            if not os.path.exists('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5'):
                # if not, create a corrupted dataset
                self._create_regression_dataset()
            variables = pickle.load(open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'rb'))
            # Train one classifier for each class
            print(" Training a regressor...")
            self._regressor = []
            for field_num in range(len(variables[0])):
                self._regressor.append(XGBRegressor(
                    random_state=0,
                    n_estimators=self._num_estimators, 
                    max_depth=self._max_depth if field_num in [2,3,4,5] else 3)
                    .fit(variables[0][field_num],variables[1][field_num]))
            print(" Complete.")
            # Save
            pickle.dump(self._regressor, open('utils/prediction_files/regressor_'+self._prediction_algorithm+'.h5', 'ab'))

    def _create_regression_dataset(self):
        """
        Creates a dataset that a regressor for prediciton phase of AIS data reconstruction will be trained on
        and saves it as pickle in utils/prediction_files/dataset_.h5.
        """
        print(" Preparing for training a regressor...")
        # Import files
        file = h5py.File(name='data/Baltic.h5', mode='r')
        data1 = Data(file=file)
        data1.split(train_percentage=50, val_percentage=25) # split into train, val and test set
        file.close()
        file = h5py.File(name='data/Gibraltar.h5', mode='r')
        data2 = Data(file=file)
        data2.split(train_percentage=50, val_percentage=25) # split into train, val and test set
        file.close()
        # Compose a dataset from train and val sets, keep test untouched
        message_decoded = []
        message_decoded.append(np.concatenate((data1.message_decoded_train, data2.message_decoded_train), axis=0))
        message_decoded.append(np.concatenate((data1.message_decoded_val, data2.message_decoded_val), axis=0))
        MMSI = []
        MMSI.append(np.concatenate((data1.MMSI_train, data2.MMSI_train), axis=0))
        MMSI.append(np.concatenate((data1.MMSI_val, data2.MMSI_val), axis=0))
        if self._prediction_algorithm == 'xgboost':
            timestamp = []
            timestamp.append(np.concatenate((data1.timestamp_train, data2.timestamp_train), axis=0))
            timestamp.append(np.concatenate((data1.timestamp_val, data2.timestamp_val), axis=0))    
        variables = [[],[]] if self._prediction_algorithm == 'ar' else [[],[],[],[]]
        for i in range(len(variables)):
            for field_num in range(len(self.fields)): 
                variables[i].append([])        
        if self._prediction_algorithm == 'xgboost':
            for i in [0,1]:
                for message_idx in range(len(MMSI[i])):
                    messages_idx = np.where(np.array(MMSI[i]) == MMSI[i][message_idx])[0]
                    if messages_idx.shape[0]>2:
                        for field in self.fields:
                            sample = self._create_regressor_sample( 
                                message_decoded=message_decoded[i],
                                timestamp=timestamp[i],
                                idx=MMSI[i],
                                message_idx=message_idx,
                                field=field)
                            if sample is not None: 
                                variables[i+i][self.fields.index(field)].append(sample)
                                if field in [2,3,7,8,12]: 
                                    new_message_idx = np.where(messages_idx==message_idx)[0][0]
                                    previous_idx = messages_idx[new_message_idx-1]
                                    variables[i+i+1][self.fields.index(field)].append( # append the difference
                                    message_decoded[i][message_idx,field] - message_decoded[i][previous_idx,field])
                                else: variables[i+i+1][self.fields.index(field)].append(message_decoded[i][message_idx,field])
        if self._prediction_algorithm == 'ar':
            MMSI = np.concatenate((MMSI[0], MMSI[1]), axis=0)
            message_decoded = np.concatenate((message_decoded[0], message_decoded[1]), axis=0)
            MMSI_vec = count_number(MMSI)[1]
            for MMSI_0 in MMSI_vec:
                message_idx = np.where(MMSI==MMSI_0)[0][-1]
                for field_num in range(len(self.fields)):
                    batch = self._create_ar_sample(
                        message_decoded=message_decoded,
                        idx=MMSI,
                        message_idx=message_idx,
                        field=self.fields[field_num])
                    if batch.shape[0]>20:
                        variables[0][field_num].append(batch)
                        if field_num in [0,1,3,4,6]:  # for lon, lat and static
                            previous_idx = np.where(MMSI==MMSI_0)[0][-2]
                            variables[1][field_num].append( # append the difference
                                message_decoded[message_idx,self.fields[field_num]] - message_decoded[previous_idx,self.fields[field_num]])
                        else: 
                            variables[1][field_num].append(message_decoded[message_idx,self.fields[field_num]])
        pickle.dump(variables, open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'ab'))
        print(" Complete.")

    def _optimize_regression(self, hyperparameter):
        """ 
        Chooses optimal value of regressor's hyperparameters for prediction stage. \n
        Argument: hyperparameter - string indicating which hyperparameter to optimize: 
        'max_depth' or 'n_estimators' (for XGBoost) or 'lags' (for VAR).
        """
        # Check if the file with the classifier dataset exist
        if not os.path.exists('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5'):
            # if not, create a damaged dataset
            self._create_regression_dataset()
        variables = pickle.load(open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'rb'))
        mae_train = []
        mae_val = []
        print(" Searching for optimal " + hyperparameter + "...")
        if self._prediction_algorithm == 'xgboost':
            params = [2, 5, 8, 10, 13, 15, 20, 30, 50, 100]
            for param in params:
                regressor = []
                mae_train_field = []
                mae_val_field = []
                for field in self.fields_dynamic:
                    field_num = self.fields.index(field)
                    field_num_d = self.fields_dynamic.index(field)
                    if hyperparameter == 'max_depth':
                        regressor.append(XGBRegressor(
                            random_state=0, 
                            n_estimators=self._num_estimators, 
                            max_depth=param,
                            ).fit(variables[0][field_num], variables[1][field_num]))
                    elif hyperparameter == 'n_estimators':
                        regressor.append(XGBRegressor(
                            random_state=0, 
                            n_estimators=param, 
                            max_depth=self._max_depth,
                            ).fit(variables[0][field_num], variables[1][field_num]))
                    pred = regressor[field_num_d].predict(np.array(variables[0][field_num]))
                    mae_train_field.append(calculate_SMAE(pred, variables[1][field_num], self.fields[field_num]))
                    pred = regressor[field_num_d].predict(np.array(variables[2][field_num]))
                    mae_val_field.append(calculate_SMAE(pred, variables[3][field_num], self.fields[field_num]))
                mae_train.append(np.mean(mae_train_field))
                mae_val.append(np.mean(mae_val_field))
        elif self._prediction_algorithm == 'ar' and hyperparameter=='lags':
            params = [1,2,3,5,7,10,20]
            for param in params:
                mae_train_field = []
                for field_num in range(len(variables[1])):
                    y_pred = []
                    y_true = []
                    for trajectory in range(len(variables[1][field_num])):
                        pred = self._validate_prediction(
                            pred=self._predict_ar(variables[0][field_num][trajectory], param, self.fields[field_num]),
                            field=self.fields[field_num])
                        y_pred.append(pred)
                        y_true.append(variables[1][field_num][trajectory])
                    mae_train_field.append(calculate_SMAE(y_pred, y_true, self.fields[field_num]))
                mae_train.append(np.mean(mae_train_field))
        print(" Complete. ")
        fig, ax = plt.subplots()
        ax.plot(params, mae_train, color='k')
        ax.scatter(params, mae_train, color='k', s=6)
        if self._prediction_algorithm == 'xgboost': 
            ax.plot(params, mae_val, color='b')
            ax.scatter(params, mae_val, color='b', s=6)
            if self._language == 'eng': ax.legend(["Training set", "Validation set"])
            elif self._language == 'pl': ax.legend(["Zbiór treningowy", "Zbiór walidacyjny"])
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel("SMAE") 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.show()
        # Save results
        if hyperparameter == 'max_depth': self._max_depth = int(input("Choose the optimal max_depth: "))
        elif hyperparameter == 'n_estimators': self._num_estimators = int(input("Choose the optimal n_estimators: "))
        elif hyperparameter == 'lags': self._lags = int(input("Choose the optimal maxlags: "))
        # Retrain the model
        if hyperparameter == 'max_depth' or hyperparameter == 'n_estimators':
            if os.path.exists('utils/prediction_files/regressor_'+self._prediction_algorithm+'.h5'):
                os.remove('utils/prediction_files/regressor_'+self._prediction_algorithm+'.h5')
            self._train_regressor()

    def _create_ar_sample(self, message_decoded, idx, message_idx, field):
        """
        Computes the sample for autoregression for prediction stage. \n
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - message_idx - scalar, int, index of a message to correct,
        - field - scalar, int, a field to examine. \n
        Returns: sample - a numpy-array with time series based on a trajectory, shape=(num_previous_messages, 4 or 3).
        """
        # Take datapoints only from the given cluster AND from the past
        cluster_idx = idx[message_idx]
        idx_cropped = idx[0:message_idx]
        message_decoded_cropped = message_decoded[0:message_idx,:]
        message_decoded_cropped = message_decoded_cropped[np.array(idx_cropped)==cluster_idx,:]
        # Take only the fields of interest
        if message_decoded_cropped.shape[0]>2:
            if field in [5,7,8]:
                num_messages = message_decoded_cropped.shape[0]
                sample = np.zeros((num_messages-1, 3))                           
                sample[:,0] = message_decoded_cropped[1:num_messages, 5]
                sample[:,1] = message_decoded_cropped[1:num_messages, 7] - message_decoded_cropped[0:num_messages-1, 7]
                sample[:,2] = message_decoded_cropped[1:num_messages, 8] - message_decoded_cropped[0:num_messages-1, 8]
            elif field == 9:
                num_messages = message_decoded_cropped.shape[0]
                sample = np.zeros((num_messages-1, 4))
                sample[:,0] = message_decoded_cropped[1:num_messages, 7]
                sample[:,1] = message_decoded_cropped[1:num_messages, 8]
                sample[:,2] = message_decoded_cropped[1:num_messages, 9]
                delta_lon_deg = message_decoded_cropped[1:num_messages, 7] - message_decoded_cropped[0:num_messages-1, 7]
                delta_lat_deg = message_decoded_cropped[1:num_messages, 8] - message_decoded_cropped[0:num_messages-1, 8]
                for i in range(num_messages-1):
                    if delta_lon_deg[i] != 0: sample[i,3] = np.arctan(delta_lat_deg[i]/delta_lon_deg[i])/np.pi*180
                    else: sample[i,3] = 90
            elif field in self.fields_static:
                sample = message_decoded_cropped[1:, [7,8,field]]
        else:
            sample = np.array([]) 
        return sample
    
    def _create_regressor_sample(self, message_decoded, timestamp, idx, message_idx, field):
        """
        Computes the sample for  XGBoost regressor for prediction stage. \n
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
        - timestamp - list of strings with timestamp of each message, len=num_messages,
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - message_idx - scalar, int, index of a message to correct,
        - field - scalar, int, a field to examine. \n
        Returns: sample - a numpy-array vector descibing a message for correct, shape=(?,).
        """
        messages_idx = np.where(np.array(idx)==idx[message_idx])[0]
        new_message_idx = np.where(messages_idx==message_idx)[0][0]
        if 0 < new_message_idx < (messages_idx.shape[0]-1): 
            previous_idx = messages_idx[new_message_idx-1]
            next_idx = messages_idx[new_message_idx+1]
            if field in [7,8]: 
                sample = np.zeros((7))
                if field == 7: 
                    sample[0] = message_decoded[message_idx,8]-message_decoded[previous_idx,8]
                    sample[1] = message_decoded[next_idx,8]-message_decoded[message_idx,8]
                    sample[2] = (message_decoded[next_idx,7]-message_decoded[previous_idx,7])/2
                    sample[3] = 1 if message_decoded[message_idx,9]<180 else -1
                else: 
                    sample[0] = message_decoded[message_idx,7]-message_decoded[previous_idx,7]
                    sample[1] = message_decoded[next_idx,7]-message_decoded[message_idx,7]
                    sample[2] = (message_decoded[next_idx,8]-message_decoded[previous_idx,8])/2
                    sample[3] = 1 if message_decoded[message_idx,9]<90 or message_decoded[message_idx,9]>270 else -1
                sample[4] = message_decoded[message_idx,5]
                sample[5] = (timestamp[message_idx]-timestamp[previous_idx]).seconds # timestamp difference
                sample[6] = (timestamp[next_idx]-timestamp[message_idx]).seconds
            elif field == 5:
                sample = np.zeros((8))
                sample[0] = np.abs(message_decoded[message_idx,7]-message_decoded[previous_idx,7]) # longitude difference
                sample[1] = np.abs(message_decoded[message_idx,8]-message_decoded[previous_idx,8]) # latitude difference
                sample[2] = np.abs(message_decoded[next_idx,7]-message_decoded[message_idx,7]) # longitude difference
                sample[3] = np.abs(message_decoded[next_idx,8]-message_decoded[message_idx,8])
                sample[4] = message_decoded[previous_idx,5]
                sample[5] = message_decoded[next_idx,5]
                sample[6] = (timestamp[message_idx]-timestamp[previous_idx]).seconds
                sample[7] = (timestamp[next_idx]-timestamp[message_idx]).seconds
            elif field == 9:
                sample = np.zeros((6))
                sample[0] = message_decoded[message_idx,7]-message_decoded[previous_idx,7] 
                sample[1] = message_decoded[message_idx,8]-message_decoded[previous_idx,8]
                sample[2] = message_decoded[next_idx,7]-message_decoded[message_idx,7] 
                sample[3] = message_decoded[next_idx,8]-message_decoded[message_idx,8] 
                sample[4] = message_decoded[previous_idx,9]
                sample[5] = message_decoded[next_idx,9]
            else:
                sample = np.zeros((2))
                sample[0] = (message_decoded[next_idx,field]-message_decoded[previous_idx,field])/2
                sample[1] = np.std(np.delete(message_decoded[messages_idx,field], new_message_idx, axis=0))
        else: sample = None
        return sample

    def _predict_ar(self, sample, lags, field):
        """
        Predicts the value of a given field of AIS message using autoregression. \n
        Arguments:
        - sample - a numpy-array with time series based on a trajectory, shape=(num_previous_messages, 4 or 3),
        - field - scalar, int, a field to examine. \n
        Returns: pred - a scalar, float, correct field value.
        """
        model = sm.tsa.VAR(sample)
        res = model.fit(maxlags=lags, trend='n')
        pred = res.forecast(sample, steps=1)
        if field in [5,7,8]: pred = pred[0, [5,7,8].index(field)]
        elif field in [2,3,9,12]: pred = pred[0,2]
        return pred

    def find_and_reconstruct_data(self, message_decoded, message_bits, idx, timestamp, outliers, if_bits=True):
        """
        Takes every field (and message) marked in previous stage as an outlier, predicts its correct form, saves the results 
        in self.predictions and modifies the original dataset accordingly. \n
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
        - message_bits - numpy array, original AIS messages in binary form (1 column = 1 bit), shape=(num_mesages, num_bits (168)),
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - timestamp - list of strings with timestamp of each message, len=num_messages,
        - outliers - numpy array with anomaly detection information, shape=(num_messages, 3) (anomaly_detection.txt)
          (1. column - if a message is outlier, 2. column - proposed correct cluster, 3. column - possibly damaged field),
        - if_bits - (optional) boolean, decide if to update also dataset in binary form (default=True). \n
        Returns: message_bits_reconstructed, message_decoded_reconstructed - numpy array, corrected dataset in binary/decimal form.
        """
        if self._verbose: print("Reconstructing data...")
        if if_bits: message_bits_reconstructed = copy.deepcopy(message_bits)
        message_decoded_reconstructed = copy.deepcopy(message_decoded)
        idx_new = copy.deepcopy(idx)
        indices = []
        for i in range(len(idx)):
            if outliers[i][0]==1: 
                indices.append(i)
                idx_new[i] = outliers[i][1]
        for message_idx in indices:
            message_decoded_0 = message_decoded_reconstructed[message_idx,:]
            dict = {}
            dict.update({'message_idx':message_idx})
            fields = outliers[message_idx][2]
            include = True
            for field in fields:
                pred = self.reconstruct_data(
                    message_decoded=copy.deepcopy(message_decoded_reconstructed),
                    timestamp=timestamp,
                    idx=idx_new,
                    message_idx=message_idx,
                    field=field)
                if pred is None: include = False
                else: 
                    dict.update({field: pred})
                    message_decoded_0[field] = pred
            if include: 
                self.predictions.append(dict)
                message_decoded_reconstructed[message_idx,:] = message_decoded_0
                if if_bits: message_bits_reconstructed[message_idx,:] = encode(message_decoded_0)
        if self._verbose: print("Complete.")
        if if_bits: return message_bits_reconstructed, message_decoded_reconstructed, idx_new
        else: return message_decoded_reconstructed, idx_new

    def reconstruct_data(self, message_decoded, timestamp, idx, message_idx, field):
        """
        Predicts the correct form of a value from a given field of a given AIS message. \n
        Arguments:
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
        - timestamp - list of strings with timestamp of each message, len=num_messages,
        - idx - list of indices of clusters assigned to each message, len=num_messages,
        - message_idx - scalar, int, index of a message to correct,
        - field - scalar, int, a field to examine. \n
        Returns: pred - a scalar, float, correct field value.
        """
        if self._prediction_algorithm == 'ar':
            sample = self._create_ar_sample(message_decoded, idx, message_idx, field)
            if sample.shape[0]==0 or sample.shape[0]==1: pred = None # too few examples to make prediction
            else:
                messages_idx = np.where(np.array(idx)==idx[message_idx])[0]
                new_message_idx = np.where(messages_idx==message_idx)[0][0]
                previous_idx = messages_idx[new_message_idx-1]
                if sample.shape[0]>self._lags: # normal prediction - enough examples for VAR 
                    pred = self._predict_ar(sample, self._lags, field)
                    if field == 7 or field == 8: pred = pred + message_decoded[previous_idx, field]
                else: # too few examples for VAR, but enough for estimation (between 2 and lags)
                    if field in self.fields_dynamic: 
                        delta_lon_deg = message_decoded[message_idx, 7]-message_decoded[previous_idx, 7]
                        delta_lat_deg = message_decoded[message_idx, 8]-message_decoded[previous_idx, 8]
                        if field == 9: # cog
                            if delta_lon_deg<0: cart = np.sign(delta_lat_deg)*180-np.arctan(delta_lat_deg/abs(delta_lon_deg))/np.pi*180
                            elif delta_lon_deg>0: cart = np.arctan(delta_lat_deg/delta_lon_deg)/np.pi*180
                            else: cart = 90 # delta_lon_deg = 0
                            pred = np.mod(90-cart,360)
                        else:
                            coeff_lon = 111320 * np.cos(message_decoded[message_idx, 8]*np.pi/180)
                            delta_lon = delta_lon_deg * coeff_lon
                            coeff_lat = 111320
                            delta_lat = delta_lat_deg * coeff_lat
                            time = timestamp[message_idx]-timestamp[previous_idx]
                            time = time.seconds
                            if field == 5: # sog
                                dist = np.sqrt(delta_lat*delta_lat + delta_lon*delta_lon)
                                pred = dist/(time * 0.5144) # in knots
                            elif field == 7: # lon
                                dist = message_decoded[message_idx, 5] * 0.5144 * time.seconds # in m
                                if 0 < message_decoded[message_idx, 9] < 180: # heading east
                                    pred = message_decoded[message_idx, 7] + np.sqrt(abs(dist*dist - delta_lat*delta_lat))/coeff_lon
                                else: # heading west
                                    pred = message_decoded[message_idx, 7] - np.sqrt(abs(dist*dist - delta_lat*delta_lat))/coeff_lon
                            elif field == 8: # lat
                                dist = message_decoded[message_idx, 5] * 0.5144 * time.seconds # in m
                                if 90 < message_decoded[message_idx, 9] < 270: # heading south
                                    pred = message_decoded[message_idx, 8] - np.sqrt(abs(dist*dist - delta_lon*delta_lon))/coeff_lat
                                else: # heading north
                                    pred = message_decoded[message_idx, 8] + np.sqrt(abs(dist*dist - delta_lon*delta_lon))/coeff_lat
                        pred = np.round(pred, decimals=self._decimals[self.fields_dynamic.index(field)])
                    elif  field in self.fields_static: pred = np.round(np.mean(sample[:,2]))
        elif self._prediction_algorithm == 'xgboost':
            sample = self._create_regressor_sample(message_decoded, timestamp, idx, message_idx, field)
            if sample is None: pred = None
            else: 
                pred = self._regressor[self.fields.index(field)].predict(sample.reshape(1,-1))[0]
                if field in [2,3,7,8,12]: 
                    messages_idx = np.where(np.array(idx)==idx[message_idx])[0]
                    new_message_idx = np.where(messages_idx==message_idx)[0][0]
                    previous_idx = messages_idx[new_message_idx-1]
                    pred = pred + message_decoded[previous_idx, field]
        if pred is not None: pred = self._validate_prediction(pred, field)
        return pred
    
    def _validate_prediction(self, pred, field):
        """
        Makes sure the predicted values matches the official specification (in terms of range, decimals). \n
        Argumenets:
        - pred - a scalar, float, predicted field value.
        - field - scalar, int, a field that was examined. \n
        Returns: pred - a scalar, float, correct field value.
        """
        if field in self.fields_static:
            pred = np.round(pred)
            if field == 2: 
                if pred < 0 or pred > 999999999: pred = 200000000
            elif field == 3:
                if pred < 0 or pred > 15: pred = 15
            elif field == 12:
                if pred < 0 or pred > 2: pred = 0
        elif field in self.fields_dynamic:
            pred = np.round(pred, decimals=self._decimals[self.fields_dynamic.index(field)])
            if field == 5: 
                if pred < 0: pred = np.abs(pred)
                if pred > 102.2: pred = 102.3
            elif field == 7:
                if pred == -180: pred = 180
                elif pred < -180 or pred > 180: pred = np.mod(pred, (-180)*np.sign(pred))
            elif field == 8:
                if pred < -90 or pred > 90: pred = 90*np.sign(pred) - np.mod(pred, 90*np.sign(pred))
            elif field == 9:
                if pred < 0 or pred > 359.9: pred = np.mod(pred, 360)
        return pred
    

def calculate_SMAE(prediction, real, field):
    """
    Computes SMAE (or its modified version for COG) of the prediction. \n
    Arguments:
    - prediction - float scalar/list/numpy array with predictions,
    - real - float scalar/list/numpy array with ground truth,
    - field - int, scalar, field to examine. \n
    Returns: calculated SMAE, float, scalar
    """   
    scale = {2:999999999, 3:15, 5:102.3, 7:180, 8:180, 9:180, 12:2}
    if field == 7 or field == 9: # for COG and longitude
        if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
            smae = []
            iterations = len(prediction) if isinstance(prediction, list) else prediction.shape[0]
            for i in range(iterations):
                smae.append(np.min((np.abs(real[i]-prediction[i]), 360-np.abs(real[i]-prediction[i])))/scale[field])
        else: smae = np.min((np.abs(real-prediction), 360-np.abs(real-prediction)))/scale[field]
    else: smae = np.abs(np.array(real)-np.array(prediction))/scale[field]
    return np.mean(smae)