"""
Functions and classes used in prediction stage of AIS message reconstruction
"""

import numpy as np
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import h5py
import pickle
import os
import sys
sys.path.append(".")
from utils.initialization import Data
from utils.miscellaneous import count_number

class Prediction:
    """
    Class that introduces prediction phase in reconstruction of AIS data.
    """
    reconstructed = []
    fields = [2,3,5,7,8,9,12]
    _prediction_algorithm = 'xgboost'
    _regressor = []
    _max_depth = 7
    _num_estimators = 20
    _p = 3
    _q = 3
    _verbose = []
    
    def __init__(self, verbose=False, optimize=None, prediction_algorithm='xgboost'):
        """
        Class initialization (class object creation). Arguments:
        - verbose (optional) - Boolean, whether to print running logs or not, default=False,
        - optimize (optional) - string, name of regressor hyperparameter to optimize, 
            'max_depth', 'n_estimators' (for xgboost), 'p' or 'q' (for autoregression), default=None (no optimization),
        - prediction_algorithm (optional) - string deciding which model to use, 'xgboost' or 'ar', default='xgboost'.
        """
        # Initialize models and necessary variables
        self._prediction_algorithm = prediction_algorithm
        self._verbose = verbose
        if self._prediction_algorithm == 'xgboost':
            if os.path.exists('utils/prediction_files/regressor'+prediction_algorithm+'.h5'):
                # If there is a file with the trained regressor saved, load it
                self._regressor = pickle.load(open('utils/prediction_files/regressor_'+prediction_algorithm+'.h5', 'rb'))
            else:
                # otherwise train a regressor from scratch
                self._train_regressor()
        # Optimize hyperparametres if allowed
        if optimize == 'max_depth': self._optimize_regression(hyperparameter='max_depth')
        elif optimize == 'n_estimators': self._optimize_regression(hyperparameter='n_estimators')
        elif optimize == 'p': self._optimize_regression(hyperparameter='p')
        elif optimize == 'q': self._optimize_regression(hyperparameter='q')
        # Show some regressor metrics if allowed
        if self._verbose:
            # Calculate the MSE of the regressor on the training and validation data
            variables = pickle.load(open('utils/prediciton_files/dataset_'+self._prediction_algorithm+'.h5', 'rb'))
            print(" Average MSE of regressor:")
            mse = []
            if self._prediction_algorithm == 'xgboost': 
                for i in range(len(variables[1])):
                    pred = self._regressor[i].predict(variables[0][i])
                    mse.append(mean_squared_error(variables[1][i],pred))
            elif self._prediction_algorithm == 'ar':
                for i in range(len(variables[1])):
                    mse_ar = []
                    for j in range(len(variables[1][i])):
                        pred = self._predict_ar(variables[0][j], self.fields[i])
                        mse_ar.append(mean_squared_error(variables[1][i][j],pred))
                    mse.append(np.mean(mse_ar))
            print("  trainset: " + str(round(np.mean(mse),4)))
            mse = []
            if self._prediction_algorithm == 'xgboost': 
                for i in range(len(variables[3])):
                    pred = self._regressor[i].predict(variables[2][i])
                    mse.append(mean_squared_error(variables[3][i],pred))
            elif self._prediction_algorithm == 'ar':
                for i in range(len(variables[3])):
                    mse_ar = []
                    for j in range(len(variables[3][i])):
                        pred = self._predict_ar(variables[2][j], self.fields[i])
                        mse_ar.append(mean_squared_error(variables[3][i][j],pred))
                    mse.append(np.mean(mse_ar))
            print("  valset: " + str(round(np.mean(mse),4)))

    def _train_regressor(self):
        """
        Trains a regressor that will be used in prediction phase of AIS data reconstruction
        and saves it as pickle in utils/prediction_files/regressor_.h5 and in self._regressor.
        """
        if self._prediction_algorithm == 'xgboost':
            # Check if the file with the regressor inputs exist
            if not os.path.exists('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5'):
                # if not, create a corrupted dataset
                self._create_regression_dataset()
            variables = pickle.load(open('utils/prediction_files/dataset'+self._prediction_algorithm+'.h5', 'rb'))
            # Train one classifier for each class
            print(" Training a regressor...")
            self._regressor = []
            for i in range(len(variables[0])):
                self._regressor.append(XGBRegressor(
                    random_state=0,
                    n_estimators=self._num_estimators, 
                    max_depth=self._max_depth)
                    .fit(variables[0][i],variables[1][i]))
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
        variables = [[],[],[],[]]
        for i in range(self.fields): 
            if self._prediction_algorithm == 'xgboost':
                variables[0].append([]) # training set          
                variables[2].append([]) # validation set
            variables[1].append([]) # training labels          
            variables[3].append([]) # validation labels
        for i in [0,1]:
            if self._prediction_algorithm == 'xgboost':
                for message_idx in range(len(MMSI[i])):
                    if len(np.where(np.array(MMSI[i]) == MMSI[i][message_idx])[0])>2:
                        for field in self.fields:
                            variables[i+i][self.fields.index(field)].append(self._create_regressor_sample(self, 
                                message_decoded=message_decoded[i],
                                idx=MMSI[i],
                                message_idx=message_idx,
                                field=field))
                            variables[i+i+1][self.fields.index(field)].append(message_decoded[i][message_idx,field])
            if self._prediction_algorithm == 'ar':
                MMSI_vec = count_number(MMSI[i])[1]
                for MMSI_0 in MMSI_vec:
                    indices = MMSI[i] == MMSI_0
                    batch = message_decoded[indices,:]
                    if batch.shape[0]>self._p and batch.shape[0]>self._q:
                        variables[i+i].append(batch[:-1,:])
                        for field in self.fields:
                            variables[i+i+1][self.fields.index(field)].append(batch[-1,field])
        pickle.dump(variables, open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'ab'))
        print(" Complete.")

    def _optimize_regression(self, hyperparameter):
        """ 
        Chooses optimal value of XGBoost hyperparameters for prediction stage. \n
        Argument: hyperparameter - string indicating which hyperparameter to optimize: 
        'max_depth' or 'n_estimators'.
        """
        # Check if the file with the classifier dataset exist
        if not os.path.exists('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5'):
            # if not, create a damaged dataset
            self._create_regression_dataset()
        variables = pickle.load(open('utils/prediction_files/dataset_'+self._prediction_algorithm+'.h5', 'rb'))
        mse_train = []
        mse_val = []
        print(" Searching for optimal " + hyperparameter + "...")
        if self._prediction_algorithm == 'xgboost':
            params = [2, 5, 8, 10, 13, 15, 20, 30, 50, 100]
            for param in params:
                regressor = []
                mse_train_field = []
                mse_val_field = []
                for i in range(len(variables[1])):
                    if hyperparameter == 'max_depth':
                        regressor.append(XGBRegressor(
                            random_state=0, 
                            n_estimators=self._num_estimators, 
                            max_depth=param,
                            ).fit(variables[0][i], variables[1][i]))
                    elif hyperparameter == 'n_estimators':
                        regressor.append(XGBRegressor(
                            random_state=0, 
                            n_estimators=param, 
                            max_depth=self._max_depth,
                            ).fit(variables[0][i], variables[1][i]))
                    pred = regressor[i].predict(np.array(variables[0][i]))
                    mse_train_field.append(mean_squared_error(pred,variables[1][i]))
                    pred = regressor[i].predict(np.array(variables[2][i]))
                    mse_val_field.append(mean_squared_error(pred,variables[3][i]))
                mse_train.append(np.mean(mse_train_field))
                mse_val.append(np.mean(mse_val_field))
        elif self._prediction_algorithm == 'ar':
            params = [1,2,3,5,7,10,20]
            for param in params:
                mse_train_field = []
                mse_val_field = []
                for i in range(len(variables[1])):
                    for j in range(len(variables[1][i])):
                        if hyperparameter=='p':
                            model = sm.tsa.VARMAX(endog=variables[0][j], order=(param, self._q))
                        elif hyperparameter=='q':
                            model = sm.tsa.VARMAX(endog=variables[0][j], order=(self._p, param))
                        res = model.fit()
                        pred = res.forecast(step=1)
                        mse_train_field.append(mean_squared_error(pred,variables[1][i][j]))
                mse_train.append(np.mean(mse_train_field))
                mse_val.append(np.mean(mse_val_field))
        print(" Complete. ")
        fig, ax = plt.subplots()
        ax.plot(params, mse_train, color='k')
        ax.plot(params, mse_val, color='b')
        ax.set_title("MSE vs " + hyperparameter)
        ax.set_xlabel(hyperparameter)
        ax.set_ylabel("MSE")
        ax.legend(["Training set", "Validation set"])
        fig.show()
        # Save results
        if hyperparameter == 'max_depth': self._max_depth = int(input("Choose the optimal max_depth: "))
        elif hyperparameter == 'n_estimators': self._num_estimators = int(input("Choose the optimal n_estimators: "))
        elif hyperparameter == 'p': self._p = int(input("Choose the optimal p: "))
        elif hyperparameter == 'q': self._q = int(input("Choose the optimal q: "))
        # Retrain the model
        if hyperparameter == 'max_depth' or hyperparameter == 'n_estimators':
            if os.path.exists('utils/prediction_files/regressor_'+self._prediction_algorithm+'.h5'):
                os.remove('utils/prediction_files/regressor_'+self._prediction_algorithm+'.h5')
            self._train_regressor()

    def _create_ar_sample(self, message_decoded, idx, message_idx):
        # Take datapoints only from the given cluster AND from the past
        cluster_idx = idx[message_idx]
        idx_cropped = idx[0:message_idx]
        message_decoded_cropped = message_decoded[idx_cropped==cluster_idx,:]
        # Take only the fields of interest
        sample = message_decoded_cropped[:, self.fields]
        return sample
    
    def _create_regressor_sample(self, message_decoded, idx, message_idx, field):
        pass

    def _predict_ar(self, sample, field):
        model = sm.tsa.VARMAX(endog=sample, order=(self._p, self._q))
        res = model.fit()
        pred = res.forecast(step=1)
        return pred[self.fields.index(field)]

    def find_data_to_reconstruct(self, message_decoded, idx, outliers):
        predictions = np.zeros((len(idx),len(self.fields)))
        indices = np.where(outliers[:,0]==1)[0]
        for message_idx in indices:
            fields = outliers[message_idx,2]
            for field in fields:
                predictions[message_idx,self.fields.index(field)] = self.reconstruct_data(
                    message_decoded=message_decoded,
                    idx=idx,
                    message_idx=message_idx,
                    field=field)
        return predictions

    def reconstruct_data(self, message_decoded, idx, message_idx, field):
        if self._prediction_algorithm == 'ar':
            sample = self._create_ar_sample(message_decoded, idx, message_idx)
            if sample.shape[0]>=self._p and sample.shape[0]>=self._q:
                pred = self._predict_ar(sample, field)
            else: pred = np.mean(sample[:,self.fields.index(field)])
        elif self._prediction_algorithm == 'xgboost':
            sample = self._create_regressor_sample(message_decoded, idx, message_idx, field)
            pred = self._regressor[self.fields.index(field)].predict(sample)
        return pred