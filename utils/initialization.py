"""
Functions and classes used in initialization of AIS message reconstruction
"""

from sklearn import preprocessing
import numpy as np
import datetime
from utils.miscellaneous import count_number
import copy 


def decode(message_bits):
    """
    Reads single AIS message in binary from and decodes it into decimal. \n
    Argument: message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_bits (168),).
    Returns: 
    - message_decoded - list of AIS message fields decoded from binary to decimal, len=(num_fields (14))
    - X - numpy array, AIS feature vector (w/o normalization), shape=(num_features (115),)
    - MMSI - scalar, int, MMSI identifier from decoded AIS message.
    """
    X = np.zeros(115)
    message_decoded = []
    # 0 Decode the message type 
    message_decoded.append(int('0b'+str(message_bits[0:6]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    # 1 Decode repeat indicator
    message_decoded.append(int('0b'+str(message_bits[6:8]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    # 2 Decode MMSI
    message_decoded.append(int('0b'+str(message_bits[8:38]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    MMSI = message_decoded[2]
    ship = np.remainder(MMSI,1000000)
    temp = ship
    ship_id = np.zeros(6)
    ship_vec = np.zeros(60)
    for i in range(6)[::-1]:
        ship_id[i] = np.remainder(temp,10)
        temp = np.floor((temp-ship_id[i])/10)      
    for i in range(6):
        ship_vec[int(ship_id[i]+i*10)]=1
    X[25:85] = ship_vec #get ship's id
    country = (MMSI-ship)/1000000
    temp=country
    country_id = np.zeros(3)
    country_vec = np.zeros(30)
    for i in range(3)[::-1]:
        country_id[i] = np.remainder(temp,10)
        temp = np.floor((temp-country_id[i])/10)
    for i in range(3):
        country_vec[int(country_id[i]+i*10)]=1
    X[85:115] = country_vec #get country's id
    # 3 Decode navigational status
    message_decoded.append(int('0b'+str(message_bits[38:42]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    temp = np.zeros(16)
    temp[message_decoded[3]]=1
    X[2:18]=temp
    # 4 Decode  rate of turns
    message_decoded.append(int('0b'+str(message_bits[43:50]).replace('[', '').replace(']', '').replace(' ', ''), base=2) - np.power(2,7) * message_bits[42])
    # 5 Decode speed over ground
    message_decoded.append(int('0b'+str(message_bits[50:60]).replace('[', '').replace(']', '').replace(' ', ''), base=2)/10)
    X[18]=message_decoded[5]
    # 6 Decode position accuracy
    message_decoded.append(message_bits[60])
    # 7 Decode longitude
    message_decoded.append((int('0b'+str(message_bits[62:89]).replace('[', '').replace(']', '').replace(' ', ''), base=2) - np.power(2,27)*message_bits[61])/600000)
    X[0]=message_decoded[7]
    # 8 Decode latitude
    message_decoded.append((int('0b'+str(message_bits[90:116]).replace('[', '').replace(']', '').replace(' ', ''), base=2) - np.power(2,26)*message_bits[89])/600000)
    X[1]=message_decoded[8]
    # 9 Decode course over ground
    message_decoded.append(int('0b'+str(message_bits[116:128]).replace('[', '').replace(']', '').replace(' ', ''), base=2)/10)
    X[19]=message_decoded[9]
    # 10 Decode true heading
    message_decoded.append(int('0b'+str(message_bits[128:137]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    X[20]=message_decoded[10]
    # 11 Decode time stamp
    message_decoded.append(int('0b'+str(message_bits[137:143]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    # 12 Decode special manoeuvre indicator
    message_decoded.append(int('0b'+str(message_bits[143:145]).replace('[', '').replace(']', '').replace(' ', ''), base=2))
    temp = np.zeros(4)
    temp[message_decoded[12]]=1
    X[21:25]=temp
    # 13 Decode RAIM-flag
    message_decoded.append(message_bits[148])
    return X, MMSI, message_decoded


def encode(message_decoded):
    """
    Converts single AIS message from decimal form into binary. \n
    Argument: message_decoded - list/numpy array of AIS message fields decoded from binary to decimal, shape=(num_fields (14),). \n
    Returns: message_bits - numpy array of AIS message in binary form (1 column = 1 bit), shape=(num_bits (168),).
    """
    # 0 Encode the message type
    message_string = bin(int(message_decoded[0])).replace('0b', '').zfill(6)
    # 1 Encode repeat indicator
    message_string = message_string + bin(int(message_decoded[1])).replace('0b', '').zfill(2)
    # 2 Encode MMSI
    message_string = message_string + bin(int(message_decoded[2])).replace('0b', '').zfill(30)
    # 3 Encode navigational status
    message_string = message_string + bin(int(message_decoded[3])).replace('0b', '').zfill(4)
    # 4 Encode rate of turns
    if message_decoded[4]>=0:
        message_string = message_string + bin(int(message_decoded[4])).replace('0b', '').zfill(8)
    else:
        message_string = message_string + bin(int(-1 * message_decoded[4]-1)).replace('0b', '').zfill(8).replace('0','2').replace('1','0').replace('2','1')
    # 5 Encode speed over ground
    message_string = message_string + bin(int(message_decoded[5]*10)).replace('0b', '').zfill(10)
    # 6 Encode position accuracy
    message_string = message_string + bin(int(message_decoded[6])).replace('0b', '')
    # 7 Encode longitude
    if message_decoded[7]>=0:
        message_string = message_string + bin(int(message_decoded[7]*600000)).replace('0b', '').zfill(28)
    else:
        message_string = message_string + bin(int((-600000) * message_decoded[7]-1)).replace('0b', '').zfill(28).replace('0','2').replace('1','0').replace('2','1')
    # 8 Encode latitude
    if message_decoded[8]>=0:
        message_string = message_string + bin(int(message_decoded[8]*600000)).replace('0b', '').zfill(27)
    else:
        message_string = message_string + bin(int((-600000) * message_decoded[8]-1)).replace('0b', '').zfill(27).replace('0','2').replace('1','0').replace('2','1')
    # 9 Encode course over ground
    message_string = message_string + bin(int(message_decoded[9]*10)).replace('0b', '').zfill(12)
    # 10 Encode true heading
    message_string = message_string + bin(int(message_decoded[10])).replace('0b', '').zfill(9)
    # 11 Encode time stamp
    message_string = message_string + bin(int(message_decoded[11])).replace('0b', '').zfill(6)
    # 12 Encode special manoeuvre indicator
    message_string = message_string + bin(int(message_decoded[12])).replace('0b', '').zfill(2)
    # 13 Encode RAIM-flag
    message_string = message_string + bin(0).replace('0b', '').zfill(3)
    message_string = message_string + bin(int(message_decoded[13])).replace('0b', '')
    message_string = message_string + bin(0).replace('0b', '').zfill(35)
    # Convert to numpy array
    message_bits = []
    for i in message_string:  # For each encoded bit
        decimal = ord(i)-48  # convert it from ASCII to decimal
        if 0 <= decimal <= 9:  # if it is truly a decimal, not any other sign
            message_bits.append(decimal)  # add it to the line
    message_bits = np.array(message_bits)  # convert them to numpy array
    return message_bits


class Data:
    """
    Class that loads and preprocesses data (normalizes, splits into train and test)
    """
    message_bits_train = []
    message_bits_val = []
    message_bits = []
    message_decoded_train = []
    message_decoded_val = []
    message_decoded = []
    Xraw_train = []
    Xraw_val = []
    Xraw = []
    X_train = []
    X_val = []
    X = []
    MMSI_train = []
    MMSI_val = []
    MMSI = []
    timestamp_train = []
    timestamp_val = []
    timestamp = []

    def __init__(self, file):
        """
        Class initialization (class object creation) - reads all important sets from the file.
        Arguments: 
        - file - h5py file object to the file with the dataset,
        - train_percentage (optional) - scalar, int, how much data is to be in training set
            (default=70).
        """
        # Load the file and all data (as "test set")
        self.message_bits = np.array(file.get('message_bits'))
        self.message_decoded = np.array(file.get('message_decoded'))
        self.Xraw = np.array(file.get('X'))
        self.MMSI = np.array(file.get('MMSI')).tolist()
        self.timestamp = np.array(file.get('timestamp').asstr()).tolist()
        if file.filename == 'data/Gdansk.h5' or file.filename == 'data\Gdansk.h5':
            self.timestamp = [datetime.datetime.strptime(i, '%d-%b-%Y %H:%M:%S') for i in self.timestamp]
        elif file.filename == 'data/Gibraltar.h5' or file.filename == 'data\Gibraltar.h5':
            self.timestamp = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in self.timestamp]
        elif file.filename == 'data/Baltic.h5' or file.filename == 'data\Baltic.h5':
            self.timestamp = [datetime.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in self.timestamp]

    def standarize(self, X):
        """
        Changes the data distribution to have mean=0 and std=1. \n
        Argument: X - dataset to standarize, shape = (num_messages, num_features). \n
        Returns:
        - mu - numpy array, vector of means of each feature, shape=(num_features (115),),
        - sigma - numpy array, vector of standard deviations of each feature, shape=(num_features (115),).
        """
        if X.shape[0]:
            scaler = preprocessing.StandardScaler().fit(X)
            mu = scaler.mean_
            sigma = scaler.scale_
            X_norm = scaler.transform(X)
        else: 
            X_norm = X
            mu = 0
            sigma = 0
        return X_norm, mu, sigma

    def split(self, train_percentage, val_percentage):
        """
        Splits the dataset into train and test set according to given percentage. \n
        Arguments: 
        - train_percentage - scalar, int, how much data is to be in training set.
        - val_percentage - scalar, int, how much data is to be in validation set.
        """
        np.random.seed(1) #For reproducibility
        # Get the list of all individual trajectories
        K, MMSI_vec = count_number(self.MMSI)
        # Shuffle them
        MMSI_vec = np.random.permutation(MMSI_vec)
        # Take given percentage of them
        threshold = int(np.round(K*train_percentage/100))
        train = np.zeros((len(self.MMSI)), dtype=bool)
        for i in range(threshold):
            # Get indices of messages from each trajectory
            indices = np.array(self.MMSI) == MMSI_vec[i]
            # as a train set
            train =np.bitwise_or(train, indices)
        # Val_test set is everything not in train set
        val_test = np.bitwise_not(train)
        # Apply to all variables
        self.message_bits_train = self.message_bits[train, :]
        self.message_bits = self.message_bits[val_test, :]
        self.message_decoded_train = self.message_decoded[train, :]
        self.message_decoded = self.message_decoded[val_test, :]
        self.Xraw_train = self.Xraw[train, :]
        self.Xraw = self.Xraw[val_test, :]
        self.MMSI_train = np.array(self.MMSI)[train].tolist()
        self.MMSI = np.array(self.MMSI)[val_test].tolist()
        self.timestamp_train = np.array(self.timestamp)[train].tolist()
        self.timestamp = np.array(self.timestamp)[val_test].tolist()
        # Repeat the whole process to split into val and test
        K, MMSI_vec = count_number(self.MMSI)
        MMSI_vec = np.random.permutation(MMSI_vec)
        threshold = int(np.floor(K*val_percentage/(100-train_percentage)))
        val = np.zeros((len(self.MMSI)), dtype=bool)
        for i in range(threshold):
            indices = np.array(self.MMSI) == MMSI_vec[i]
            val = np.bitwise_or(val, indices)
        test = np.bitwise_not(val)
        self.message_bits_val = self.message_bits[val, :]
        self.message_bits = self.message_bits[test, :]
        self.message_decoded_val = self.message_decoded[val, :]
        self.message_decoded = self.message_decoded[test, :]
        self.Xraw_val = self.Xraw[val, :]
        self.Xraw = self.Xraw[test, :]
        self.MMSI_val = np.array(self.MMSI)[val].tolist()
        self.MMSI = np.array(self.MMSI)[test].tolist()
        self.timestamp_val = np.array(self.timestamp)[val].tolist()
        self.timestamp = np.array(self.timestamp)[test].tolist()