# ----------- Library of functions used in miscellaneous phases of AIS message reconstruction ----------
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy


def count_number(X):
    """
    Counts how many times a single item apprears in a list and creates a list of unique items
    Argument: X - list of all items to compute 
    Returns: 
    - quantity - integer scalar, number of unique items in a list
    - vec - list of uniqe items in X
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


def visualize_trajectories(X,MMSI,MMSI_vec,goal):
    """
    Displays every trajectory/cluster/outlier
    Arguments:
    - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages, num_features (115))
    - MMSI - list of MMSI identifier from each AIS message, len = num_messages
    - MMSI_vec - list of uniqe items in MMSI
    - goal - string indicating the purpose of visualization:
    'data_visualization', 'clustering' or 'anomaly_detection'
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
    Class that introduces artificial damage to the dataset
    """
    indices_corrupted = []  # numpy array indicating whether the message was corrupted (1) or not (0), shape = (num_messages)
    _percentage = 2  # number of bits to corrupt

    def __init__(self, X, percentage):
        """
        Class initializer
        Arguments:
        - X - numpy array, dataset to corrupt, shape = (num_messages, num_features(115))
        - percentage - integer scalar, number of bits to corrupt in a message
        """
        self.indices_corrupted = np.zeros(X.shape[0])
        self._percentage = percentage

    def _choose_message(self):
        """
        Chooses a message to be artificially corrupted, for internal use only
        Argument: message_bits - numpy array, AIS message in binary form to corrupt, shape = (num_messages, 168)
        Returns: message_idx - integer scalar, index of a message to corrupt
        """
        choose = True
        while choose:
            message_idx = np.random.randint(0, self.indices_corrupted.shape[0])
            if not self.indices_corrupted[message_idx]:  # If the chosen message was not used before
                choose = False  # stop choosing
                self.indices_corrupted[message_idx]=1  # mark it so it will not be chosen again
            elif np.sum(self.indices_corrupted)==self.indices_corrupted.shape[0]:  # if all were used before
                self.reset()  # clear the indices to make them usable again
                print("Warning - the number of messages to choose exceeded the dataset size")
        return message_idx
    
    def reset(self):
        """
        Set the indices_corrupted back to all zeros to enable to choose those messages again
        """
        self.indices_corrupted = np.zeros_like(self.indices_corrupted)

    def corrupt_message(self, message_bits, message_idx=None):
        """
        Corrupt a randomly chosen message by swapping a given number (_percentage) of its bits
        Argument: message_bits - numpy array, AIS message in binary form to corrupt, shape = (num_messages, 168)
        Returns:
        - message_bits_corr - numpy array, AIS message in binary form after corruption, shape = (num_messages, 168)
        - message_idx - integer scalar, index of a message to corrupt
        - seed_idx - list of bits that were corrupted
        """
        message_bits_corr = copy.deepcopy(message_bits)
        # Choose a message to corrupt
        if message_idx is None:
            message_idx = self._choose_message()
        # Create an error seed
        seed = np.zeros(message_bits.shape[1],dtype=int)
        seed_idx = np.random.permutation(message_bits.shape[1])[0:self._percentage]
        for i in seed_idx:
            seed[i]=1
        # Apply the error to the chosen message bitwise
        message_bits_corr[message_idx,:] = np.bitwise_xor(message_bits_corr[message_idx,:],seed)
        return message_bits_corr, message_idx, seed_idx

    def corrupt_bits(self, message_bits, bit_idx, message_idx = None):
        """
        Corrupt a randomly chosen message by swapping its desired bits
        Arguments:
        - message_bits - numpy array, AIS message in binary form to corrupt, shape = (num_messages, 168)
        Returns:
        - message_bits_corr - numpy array, AIS message in binary form after corruption, shape = (num_messages, 168)
        - message_idx - integer scalar, index of a message to corrupt
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
    Class that enables cropping the dataset using a given time window
    """
    _start = 0  # time (in mins) from a most recent message to the beginning of a time window
    _stop = 5  # time (in mins) from a most recent message to the end of a time window

    def __init__(self, start, stop):
        """
        Class initializer
        Arguments:
        - start - scalar, time (in mins) from a most recent message to the beginning of a time window
        - stop - scalar, time (in mins) from a most recent message to the end of a time window
        """
        self._start = start
        self._stop = stop
    
    def use_time_window(self, data_original):
        """
        Chooses only messages from a time window with given start and stop
        Argument: data - Data object, dataset to crop including:
        - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_mesages, num_bits (168))
        - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages, num_fields (14))
        - Xraw - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages, num_features (115))
        - MMSI - list of MMSI identifier from each AIS message, len = num_messages
        - timestamp - list of strings with timestamp of each message, len = num_messages
        Returns: cropped data
        """
        data = copy.deepcopy(data_original)
        # Crop train set
        indices = [datetime.timedelta(minutes=self._start) < max(data.timestamp_train)-time < datetime.timedelta(minutes=self._stop) for time in data.timestamp_train]
        data.timestamp_train = np.array(data.timestamp_train)[indices].tolist()
        data.MMSI_train = np.array(data.MMSI_train)[indices].tolist()
        data.Xraw_train = data.Xraw_train[indices, :]
        data.message_bits_train = data.message_bits_train[indices, :]
        data.message_decoded_train = data.message_decoded_train[indices, :]
        # Crop validation set
        indices = [datetime.timedelta(minutes=self._start) < max(data.timestamp_train)-time < datetime.timedelta(minutes=self._stop) for time in data.timestamp_val]
        data.timestamp_val = np.array(data.timestamp_val)[indices].tolist()
        data.MMSI_val = np.array(data.MMSI_val)[indices].tolist()
        data.Xraw_val = data.Xraw_val[indices, :]
        data.message_bits_val = data.message_bits_val[indices, :]
        data.message_decoded_val = data.message_decoded_val[indices, :]
        # Crop test set
        indices = [datetime.timedelta(minutes=self._start) < max(data.timestamp)-time < datetime.timedelta(minutes=self._stop) for time in data.timestamp]
        data.timestamp = np.array(data.timestamp)[indices].tolist()
        data.MMSI = np.array(data.MMSI)[indices].tolist()
        data.Xraw = data.Xraw[indices, :]
        data.message_bits = data.message_bits[indices, :]
        data.message_decoded = data.message_decoded[indices, :]
        return data

    def slide_time_window(self, start_new, stop_new):
        """
        Sets the new beginning and ending to the time window
        Arguments: scalars, new start and stop of a time window
        """
        self._start = start_new
        self._stop = stop_new