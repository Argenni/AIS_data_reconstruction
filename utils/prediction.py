"""
Functions and classes used in prediction stage of AIS message reconstruction
"""

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import h5py
import pickle
import os
import sys
sys.path.append(".")
from utils.initialization import Data

class Prediction:
    """
    Class that introduces prediction phase in reconstruction of AIS data.
    """
    reconstructed = []
    _prediction_algorithm = 'xgboost'
    _regressor = []
    _max_depth = 7
    _num_estimators = 20
    
    def __init__(self, if_visualize=False, optimize=None, prediction_algorithm='xgboost'):
        """
        Class initialization (class object creation). Arguments:
        - if_visualize (optional) - boolean deciding whether to show training performance or not,
            default=False,
        - optimize (optional) - string, name of regressor hyperparameter to optimize, 
            'max_depth' or 'n_estimators', default=None (no optimization),
        - prediction_algorithm (optional) - string deciding which model to use, default = 'xgboost'.
        """
        # Initialize models and necessary variables
        self._prediction_algorithm = prediction_algorithm
        if os.path.exists('utils/prediction_files/regressor'+prediction_algorithm+'.h5'):
            # If there is a file with the trained standalone clusters field classifier saved, load it
            self._regressor = pickle.load(open('utils/prediction_files/regressor_'+prediction_algorithm+'.h5', 'rb'))
        else:
            # otherwise train a classifier from scratch
            self._train_regressor()
        # Show some regressor metrics if allowed
        if if_visualize:
            # Calculate the MSE of the regressor on the training and validation data
            variables = pickle.load(open('utils/prediciton_files/dataset_'+self._prediction_algorithm+'.h5', 'rb'))
            print(" Average MSE of regressor:")
            mse = []
            for i in range(len(variables[1])):
                pred = self._regressor[i].predict(variables[0][i])
                mse.append(mean_squared_error(variables[1][i],pred))
            print("  trainset: " + str(round(np.mean(mse),4)))
            mse = []
            for i in range(len(variables[3])):
                pred = self._regressor[i].predict(variables[2][i])
                mse.append(mean_squared_error(variables[3][i],pred))
            print("  valset: " + str(round(np.mean(mse),4)))
        # Optimize hyperparametres if allowed
        if optimize == 'max_depth': self._optimize_regressor(hyperparameter='max_depth')
        elif optimize == 'n_estimators': self._optimize_regressor(hyperparameter='n_estimators')

    def _train_regressor(self):
        """
        Trains a regressor that will be used in prediction phase of AIS data reconstruction
        and saves it as pickle in utils/prediction_files/regressor_.h5 and in self._regressor.
        """
        # Check if the file with the regressor inputs exist
        if not os.path.exists('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5'):
            # if not, create a corrupted dataset
            print(" Preparing for training a regressor...")
            self._create_regressor_dataset()
            print(" Complete.")
        variables = pickle.load(open('utils/prediction_files/dataset'+self._prediction_algorithm+'.h5', 'rb'))
        # Train one classifier for each class
        print(" Training a regressor...")
        self._regressor = []
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        print(" Complete.")
        # Save
        pickle.dump(self._regressor, open('utils/prediction_files/regressor_'+self._prediction_algorithm+'.h5', 'ab'))

    def _create_regressor_dataset(self):
        """
        Creates a dataset that a regressor for prediciton phase of AIS data reconstruction will be trained on
        and saves it as pickle in utils/prediction_files/dataset_.h5.
        """
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
        # !!!!!!!!!!!!!!!!!!!!
        variables = []
        pickle.dump(variables, open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'ab'))

    def _optimize_regressor(self, hyperparameter):
        pass

    def reconstruct_data(self):
        pass