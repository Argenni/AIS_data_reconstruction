# -------------------------------- Load, interpret and save Baltic AIS data ------------------------------------------
"""
Loads AIS message from csv file and prepares input feature vector based on it
Built for: decoded_reports_baltic_all_ships_april_2018_micro_20k_sample_ais_data.csv file 
 (timestamp[0], mess_type[2], MMSI[4], long[6], lat[5], position_acc[7], speed[8], course[9])
Creates Baltic.h5 file with the following datasets: 
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape = (num_mesages (19999), num_bits (168))
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape = (num_mesages (19999), num_fields (14))
 - X - numpy array, AIS feature vectors (w/o normalization), shape = (num_messages (19999), num_features (115))
 - MMSI - list of MMSI identifier from each AIS message, len = num_messages (19999)
 - timestamp - list of strings with timestamp of each message, format '%d/%m/%Y %H:%M:%S', len = num_messages (19999)
 """
print("\n ------------- Load, interpret and save Baltic AIS data -------------- ")

# Important imports
import csv
import numpy as np
import h5py
import sys
sys.path.append('.')
from utils.initialization import decode, encode  # pylint: disable=import-error

# Load AIS message from csv file
print("Loading data...")
data = []
with open(
    'data/decoded_reports_baltic_all_ships_april_2018_micro_20k_sample_ais_data.csv',
    'r'
    ) as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        data.append(row)
data = data[1:len(data)]  # remove the first (title) row

# Convert from string to numeric
message_decoded = []
timestamp = []
MMSI = []
for i in data:
    message = []
    # 0 Get message type
    message.append(int(i[2]))
    # 1 Get (default) repeat indicator
    message.append(0)
    # 2 Get MMSI
    message.append(int(i[4]))
    MMSI.append(int(i[4]))
    # 3 Get (default) navigational status
    message.append(15)
    # 4 Get (default) rate of turns
    message.append(-128)
    # 5 Get speed over ground
    message.append(float(i[8]))
    # 6 Get position accuracy
    message.append(int(i[7]))
    # 7 Get longitude
    message.append(float(i[6]))
    # 8 Get latitude
    message.append(float(i[5]))
    # 9 Get course over ground
    message.append(float(i[9]))
    # 10 Get (default) true heading
    message.append(511)
    # 11 Get UTC timestamp
    message.append(int(i[0][17:19]))
    timestamp.append(i[0])
    # 12 Get (default) special manoeuvre indicator
    message.append(0)
    # 13 Get (default) RAIM-flag
    message.append(0)
    message_decoded.append(message)

# Sort by date
concatenation = np.concatenate(
    (np.array(timestamp).reshape(-1,1), np.array(MMSI).reshape(-1,1), np.array(message_decoded)), 
    axis=1)
concatenation = concatenation[concatenation[:, 0].argsort()]
timestamp = concatenation[:,0].tolist()
MMSI = np.array(concatenation[:,1], dtype=int).tolist()
message_decoded = np.array(concatenation[:,2:], dtype=float)

message_bits = []  # get binary representation message_bits
for message in message_decoded:
    message_bits.append(encode(message))
message_bits = np.array(message_bits)
X = []  # get feature vector X
for message in message_bits:
    X_0, _, _ = decode(message)
    X.append(X_0)
X = np.array(X)

print("Complete.")    

# Save file
input("Press Enter to save the data and exit...")
File = h5py.File('data/Baltic.h5', mode='x')
File.create_dataset('message_bits', data=message_bits)
File.create_dataset('message_decoded', data=message_decoded)
File.create_dataset('X', data=X)
File.create_dataset('MMSI', data=MMSI)
timestamp = [i.encode('ascii', 'ignore') for i in timestamp]
File.create_dataset('timestamp', data=timestamp)
File.close()