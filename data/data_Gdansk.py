# -------------------------------- Load, interpret and save Gdansk AIS data ------------------------------------------
"""
Loads AIS message from ASCII file and prepares input feature vector based on it
Built for: AIS_data_MMSI_bin_hex.txt file (timestamp, MMSI, long, lat, AIS messages in binary and hex form in ASCII)
Creates Gdansk.h5 file with the following datasets: 
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_mesages (805), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages (805), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (805), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (805)
 - timestamp - list of strings with timestamp of each message, format '%d-%b-%Y %H:%M:%S', len = num_messages (805)
"""
print("\n ------------- Load, interpret and save Gdansk AIS data -------------- ")

# Important imports
import numpy as np
import h5py
import sys
sys.path.append('.')
from utils.initialization import decode  # pylint: disable=import-error

# Load AIS messages from ASCII file
print("Loading data...")
data = open('data/AIS_data_MMSI_bin_hex.txt','r')
data_string = data.readlines()
data.close()

# Convert all decimals from ASCII to binary
message_bits = []
timestamp = []
for i in data_string:  # For each string line
    message = []      
    data_string_split = i.split()  # split it to derive binary message[5] and a timestamp[0+1]
    for j in data_string_split[5]:  # read each decimal of the message
        decimal = ord(j)-48  # convert it from ASCII to decimal
        if 0 <= decimal <= 9:  # if it is truly a decimal, not any other sign
            message.append(decimal)  # add it to the line
    message_bits.append(message)  # and finally collect all lines
    timestamp.append(data_string_split[0]+ ' ' + data_string_split[1])  # as timestamp, concatenate two columns
message_bits = np.array(message_bits)  # convert them to numpy array

# Convert features from binary to decimal and add to the input vector
MMSI = []
X = []
message_decoded = []
for message in message_bits:
    X_0, MMSI_0, message_decoded_0 = decode(message)
    MMSI.append(MMSI_0)
    X.append(X_0)
    message_decoded.append(message_decoded_0)
X = np.array(X)
message_decoded = np.array(message_decoded)
print("Complete.")

# Filter out the obvious outliers:
# - where longitude is default (181)
MMSI=np.array(MMSI)[X[:,0] != 181].tolist()
message_bits=message_bits[X[:,0] != 181,:]
timestamp=np.array(timestamp)[X[:,0] != 181].tolist()
message_decoded=message_decoded[X[:,0] != 181,:]
X = X[X[:,0] != 181,:]
# - where latitude is default (91)
MMSI=np.array(MMSI)[X[:,1] != 91].tolist()
message_bits=message_bits[X[:,1] != 91,:]
timestamp=np.array(timestamp)[X[:,1] != 91].tolist()
message_decoded=message_decoded[X[:,1] != 91,:]
X = X[X[:,1] != 91,:]
# - all with latitude above 54.53 for clarity
MMSI=np.array(MMSI)[X[:,1] >54.53].tolist()
message_bits=message_bits[X[:,1] >54.53,:]
timestamp=np.array(timestamp)[X[:,1] >54.53].tolist()
message_decoded=message_decoded[X[:,1] >54.53,:]
X = X[X[:,1] >54.53,:]

# Save file
input("Press Enter to save the data and exit...")
File = h5py.File('data/Gdansk.h5', mode='x')
File.create_dataset('message_bits', data=message_bits)
File.create_dataset('message_decoded', data=message_decoded)
File.create_dataset('X', data=X)
File.create_dataset('MMSI', data=MMSI)
timestamp = [i.encode('ascii', 'ignore') for i in timestamp]
File.create_dataset('timestamp', data=timestamp)
File.close()