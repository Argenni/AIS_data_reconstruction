# ----------- Library of functions used in miscellaneous stages of AIS message reconstruction ----------
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy


def count_number(X):
    """
    Counts how many times a single item apprears in a list and creates a list of unique items. \n
    Argument: X - list of all items to compute. \n
    Returns: 
    - quantity - scalar, int, number of unique items in a list,
    - vec - list of uniqe items in X.
    """
    vec = []
    quantity = 0
    if len(X)>0:  # When X is not empty
        for i in X:  # Check items
            indices = np.where(np.array(vec)==i,1,0)
            if np.sum(indices)==0:  # if they are not already on the list
                vec.append(float(i))  # add them to the list
                quantity = quantity+1  # and increment the counter
    return quantity, vec


def visualize_trajectories(X, MMSI, MMSI_vec, goal):
    """
    Displays every trajectory/cluster/outlier. \n
    Arguments:
    - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
    - MMSI - list of MMSI identifier from each AIS message, len=num_messages,
    - MMSI_vec - list of uniqe items in MMSI,
    - goal - string, the purpose of visualization: 'data_visualization', 'clustering' or 'anomaly_detection'.
    """
    plt.figure()
    for i in MMSI_vec: 
        # Plot each trajectory/cluster with a different color
        indices = np.array(MMSI)==i
        plt.scatter(X[indices,0],X[indices,1])
    if goal == 'data_visualization':
        plt.title("Trajectory visualization")
    elif goal == 'clustering':
        plt.title("Clustering results")
    elif goal == 'anomaly_detection':
        plt.title("Outliers found")
        plt.legend(["Regular datapoints", "Detected outliers"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show(block=False)


class Corruption:
    """
    Class that introduces artificial damage to the dataset.
    """
    indices_corrupted = []  # numpy array indicating whether the message was corrupted (1) or not (0), shape=(num_messages,)

    def __init__(self, X):
        """
        Class initialization (class object creation). \n 
        Argument: X - numpy array, dataset to corrupt, shape=(num_messages, num_features(115)).
        """
        self.indices_corrupted = np.zeros(X.shape[0])

    def _choose_message(self):
        """
        Chooses a message to be artificially corrupted (for internal use only). \n
        Argument: message_bits - numpy array, AIS message in binary form to corrupt, shape=(num_messages, 168). \n
        Returns: message_idx - scalar, int, index of a message to corrupt.
        """
        choose = True
        while choose:
            message_idx = np.random.randint(0, self.indices_corrupted.shape[0])
            if not self.indices_corrupted[message_idx]:  # If the chosen message was not used before
                choose = False  # stop choosing
                self.indices_corrupted[message_idx]=1  # mark it so it will not be chosen again
            elif np.sum(self.indices_corrupted)==self.indices_corrupted.shape[0]:  # if all were used before
                self.reset()  # clear the indices to make them usable again
                print("Warning - the number of messages to choose exceeded the dataset size.")
        return message_idx
    
    def reset(self):
        """
        Sets the indices_corrupted back to all zeros to enable those messages to be chosen again.
        """
        self.indices_corrupted = np.zeros_like(self.indices_corrupted)

    def corrupt_bits(self, message_bits, bit_idx, message_idx=None):
        """
        Corrupts a randomly chosen message by swapping its desired bits. \n
        Arguments:
        - message_bits - numpy array, AIS message in binary form to corrupt, shape=(num_messages, 168),
        - bit_idx - scalar, int, index of a bit to corrupt,
        - message_idx - (optional) scalar, int, index of a message to corrupt. \n
        Returns:
        - message_bits_corr - numpy array, AIS message in binary form after corruption, shape=(num_messages, 168),
        - message_idx - scalar, int, index of a corrupted message.
        """
        message_bits_corr = copy.deepcopy(message_bits)
        # Choose a message to corrupt
        if message_idx is None:
            message_idx = self._choose_message()
        # Create an error seed
        seed = np.zeros(message_bits.shape[1],dtype=int)
        seed[bit_idx]=1
        # Apply the error to the chosen message bitwise
        message_bits_corr[message_idx,:] = np.bitwise_xor(message_bits_corr[message_idx,:],seed)
        return message_bits_corr, message_idx


class TimeWindow:
    """
    Class that enables cropping the dataset using a given time window.
    """
    _start = 0  # time (in mins) from a most recent message to the beginning of a time window
    _stop = 5  # time (in mins) from a most recent message to the end of a time window

    def __init__(self, start, stop):
        """
        Class initialization (class object creation). Arguments:
        - start - scalar, int, time (in mins) from a most recent message to the beginning of a time window,
        - stop - scalar, int, time (in mins) from a most recent message to the end of a time window.
        """
        self._start = start
        self._stop = stop
    
    def use_time_window(self, data_original, crop_train=True, crop_val=True, crop_test=True, verbose=True):
        """
        Chooses only messages from a time window with given start and stop. \n
        Arguments: 
        - data - Data object, dataset to crop including:
            - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_mesages, num_bits (168)),
            - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_mesages, num_fields (14)),
            - Xraw - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
            - MMSI - list of MMSI identifier from each AIS message, len=num_messages,
            - timestamp - list of strings with timestamp of each message, len=num_messages,
        - crop_train/val/test - (optional) Boolean, whether to use time window on a specific set or not (default=True),
        - verbose (optional) - Boolean, whether to print warnings if any of the aforementioned sets were not cropped (default=True). \n
        Returns: cropped data (message_bits, message_decoded, Xraw, MMSI, timestamp).
        """
        data = copy.deepcopy(data_original)
        # Crop train set
        if (len(data.timestamp_train)>0 and crop_train):
            indices = [datetime.timedelta(minutes=self._start) < max(data.timestamp_train)-time < datetime.timedelta(minutes=self._stop) for time in data.timestamp_train]
            data.timestamp_train = np.array(data.timestamp_train)[indices].tolist()
            data.MMSI_train = np.array(data.MMSI_train)[indices].tolist()
            data.Xraw_train = data.Xraw_train[indices, :]
            data.message_bits_train = data.message_bits_train[indices, :]
            data.message_decoded_train = data.message_decoded_train[indices, :]
        else: 
            if verbose: print("\n Warning: Training set not cropped with a time window!")
        # Crop validation set
        if (len(data.timestamp_val)>0 and crop_val):
            indices = [datetime.timedelta(minutes=self._start) < max(data.timestamp_val)-time < datetime.timedelta(minutes=self._stop) for time in data.timestamp_val]
            data.timestamp_val = np.array(data.timestamp_val)[indices].tolist()
            data.MMSI_val = np.array(data.MMSI_val)[indices].tolist()
            data.Xraw_val = data.Xraw_val[indices, :]
            data.message_bits_val = data.message_bits_val[indices, :]
            data.message_decoded_val = data.message_decoded_val[indices, :]
        else: 
            if verbose: print("\n Warning: Validation set not cropped with a time window!")
        # Crop test set
        if (len(data.timestamp)>0 and crop_test):
            indices = [datetime.timedelta(minutes=self._start) < max(data.timestamp)-time < datetime.timedelta(minutes=self._stop) for time in data.timestamp]
            data.timestamp = np.array(data.timestamp)[indices].tolist()
            data.MMSI = np.array(data.MMSI)[indices].tolist()
            data.Xraw = data.Xraw[indices, :]
            data.message_bits = data.message_bits[indices, :]
            data.message_decoded = data.message_decoded[indices, :]
        else: 
            if verbose: print("\n Warning: Test set not cropped with a time window!")
        return data

    def slide_time_window(self, start_new, stop_new):
        """
        Sets the new beginning and ending to the time window. \n
        Arguments: ints, scalars, new start and stop of a time window.
        """
        self._start = start_new
        self._stop = stop_new