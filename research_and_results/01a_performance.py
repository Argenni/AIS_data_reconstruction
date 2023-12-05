"""
Checks the performace of AIS data reconstruction (damages 1 or 2 bits of a randomly selected AIS message). \n
Requires: Gdansk.h5 / Baltic.h5 / Gibraltar.h5 file with the following datasets (created by data_.py):
 - message_bits - numpy array of AIS messages in binary form (1 column = 1 bit), shape=(num_messages, num_bits (168)),
 - message_decoded - numpy array of AIS messages decoded from binary to decimal, shape=(num_messages, num_fields (14)),
 - X - numpy array, AIS feature vectors (w/o normalization), shape=(num_messages, num_features (115)),
 - MMSI - list of MMSI identifier from each AIS message, len=num_messages. \n
Creates 01a_performance_.h5 file, with OK_vec with:
- if clustering: for both DBSCAN and kmeans, percentage of correctly clustered messages,
- if anomaly detection: for each amount of dataset/corrupted bits:
    0. field classification recall,
    1. field classification precision,
    2. field classification Jaccard score,
    3. field classification Hamming score,
    4. message classification accuracy (only for 1-element-cluster anomaly detection).
"""
print("\n----------- AIS data reconstruction performance - one damaged message at a time --------- ")

# Important imports
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from scipy import signal
import copy
import os
import sys
sys.path.append('.')
from utils.initialization import Data, decode # pylint: disable=import-error
from utils.clustering import Clustering, check_cluster_assignment
from utils.anomaly_detection import AnomalyDetection, calculate_ad_accuracy
from utils.miscellaneous import count_number, Corruption

# ----------------------------!!! EDIT HERE !!! ---------------------------------  
np.random.seed(1)  # For reproducibility
distance = 'euclidean'
clustering_algorithm = 'DBSCAN'  # 'kmeans' or 'DBSCAN'
ad_algorithm = 'rf' # 'rf', 'xgboost' or 'threshold' (only for 1-element-cluster anomaly detection) 
stage = 'clustering' # 'clustering', 'ad_1element', 'ad_multielement'
num_metrics = {'clustering':2, 'ad_1element':5, 'ad_multielement':4}
num_bits = {'clustering':10, 'ad_1element':2, 'ad_multielement':2}
num_experiment = {'clustering':50, 'ad_1element':100, 'ad_multielement':10}
# --------------------------------------------------------------------------------
if ad_algorithm=='threshold' and stage!='ad_1element': ad_algorithm = 'xgboost' 

# Decide what to do
precomputed = 'start'
while precomputed != '1' and precomputed != '2':
    precomputed = input("Choose: \n1 - Run computations from scratch \n2 - Load precomputed values \n")
    if precomputed != '1' and precomputed != '2':
        print("Unrecognizable answer.")

# Load data
print(" Initialization... ")
if precomputed == '2':  # Load file with precomputed values
    if stage == 'clustering': file = h5py.File(name='research_and_results/01a_performance_clustering.h5', mode='r')
    elif stage == 'ad_1element': file = h5py.File(name='research_and_results/01a_performance_1element_' + ad_algorithm + '.h5', mode='r')
    elif stage == 'ad_multielement': file = h5py.File(name='research_and_results/01a_performance_multielement_' + ad_algorithm+ '.h5', mode='r')
    OK_vec = np.array(file.get('OK_vec'))
    file.close()

else:  # or run the computations
    if stage != 'clustering': 
        filename = ['Gdansk.h5', 'Baltic.h5', 'Gibraltar.h5']
        bits = np.array(np.arange(8,42).tolist() + np.arange(50,60).tolist() + np.arange(61,128).tolist() + np.arange(143,145).tolist())
    else: 
        filename = ['Gdansk.h5']
        bits = list(range(145))  
        bits.append(148)
    field_bits = np.array([6, 8, 38, 42, 50, 60, 61, 89, 116, 128, 137, 143, 145, 148])  # range of fields
    OK_vec = np.zeros((len(filename), num_bits[stage], num_metrics[stage], num_experiment[stage]))
    for file_num in range(len(filename)):      
        # Load the data from the right file
        file = h5py.File(name='data/' + filename[file_num], mode='r')
        data = Data(file)
        if stage != 'clustering': 
            data.split(train_percentage=50, val_percentage=25)
            data.X_train, _, _ = data.standarize(data.Xraw_train)
            data.X_val, _, _ = data.standarize(data.Xraw_val)
        file.close()
        data.X, _, _ = data.standarize(data.Xraw) 
        # First clustering
        clustering = Clustering()
        K = []
        idx = []
        if clustering_algorithm=='kmeans' or stage=='clustering':
            K_kmeans, _ = count_number(data.MMSI)  # Count number of groups/ships
            idx_kmeans, _ = clustering.run_kmeans(X=data.X, K=K_kmeans)
            K.append(K_kmeans)
            idx.append(idx_kmeans)
        if clustering_algorithm=='DBSCAN' or stage=='clustering':
            idx_DBSCAN, K_DBSCAN = clustering.run_DBSCAN(X=data.X, distance=distance)
            K.append(K_DBSCAN)
            idx.append(idx_DBSCAN)            
        print(" Complete.")

        # Damage data 
        for num_bit in range(num_bits[stage]):
            print(" Damaging one message at a time - " + str(file_num+1) + ". dataset, " + str(num_bit+1) + " damaged bits...")
            corruption = Corruption(data.X)
            np.random.seed(1) # for reproducibility
            for i in range(num_experiment[stage]):
                if stage != 'clustering': 
                    if ad_algorithm=='rf' or ad_algorithm=='xgboost': ad = AnomalyDetection(ad_algorithm=ad_algorithm)
                    else: ad = AnomalyDetection(ad_algorithm='xgboost') # for threshold only
                stop = False
                while not stop:
                    Xraw_corr = copy.deepcopy(data.Xraw)
                    MMSI_corr = copy.deepcopy(data.MMSI)
                    message_decoded_corr = copy.deepcopy(data.message_decoded)
                    message_bits_corr = copy.deepcopy(data.message_bits)
                    # choose random bits to damage
                    if stage=='ad_multielement':
                        field = np.random.choice(ad.fields_dynamic, size=num_bit+1, replace=False).tolist()
                        bit_idx = [np.random.randint(field_bits[field[j]-1], field_bits[field[j]]-1) for j in len(field)]
                    else: bit_idx = np.random.choice(bits, size=num_bit+1, replace=False).tolist()
                    # perform actual damage of randomly chosen message
                    message_idx = corruption.choose_message()
                    for bit in bit_idx:
                        message_bits_corr, _ = corruption.corrupt_bits(
                            message_bits=message_bits_corr, 
                            bit_idx=bit, message_idx=message_idx)
                    # put it back to the dataset
                    X_0, MMSI_0, message_decoded_0 = decode(message_bits_corr[message_idx,:])
                    Xraw_corr[message_idx,:] = X_0
                    MMSI_corr[message_idx] = MMSI_0
                    message_decoded_corr[message_idx,:] = message_decoded_0
                    X_corr, _, _ = data.standarize(Xraw_corr)
                    # cluster again to find new cluster assignment
                    K_corr = []
                    idx_corr = []
                    if clustering_algorithm=='kmeans' or stage=='clustering':
                        K_corr_kmeans, _ = count_number(MMSI_corr)  # Count number of groups/ships
                        idx_corr_kmeans, _ = clustering.run_kmeans(X=X_corr, K=K_corr_kmeans)
                        K_corr.append(K_corr_kmeans)
                        idx_corr.append(idx_corr_kmeans)
                    if clustering_algorithm=='DBSCAN' or stage=='clustering':
                        idx_corr_DBSCAN, K_corr_DBSCAN = clustering.run_DBSCAN(X=X_corr, distance=distance)
                        K_corr.append(K_corr_DBSCAN)
                        idx_corr.append(idx_corr_DBSCAN) 
                    # check if the conditions were met - the message is either in 1-element or multielement cluster
                    if stage=='ad_1element': 
                        # Check if the cluster is a 1-element cluster
                        ad.detect_in_1element_clusters(
                            idx=idx_corr[0],
                            idx_vec=range(-1, np.max(idx_corr)+1),
                            X=X_corr,
                            message_decoded=message_decoded_corr)
                        # if so, stop searching
                        stop = ad.outliers[message_idx][0]
                        # if not, allow this message to be chosen again
                        corruption.indices_corrupted[message_idx] = ad.outliers[message_idx][0]
                    elif stage=='ad_multielement':
                        # Check if the cluster is inside a proper cluster: if so, stop searching
                        if check_cluster_assignment(idx[0], idx_corr[0], message_idx):
                            stop = sum(idx_corr==idx_corr[message_idx])>2
                        # if not, allow this message to be chosen again
                        corruption.indices_corrupted[message_idx] = stop
                    else: stop = True
                # End experiment
                if stage=='ad_1element': 
                    idx_corr[0][message_idx] = ad.outliers[message_idx][1]
                    field = [sum(field_bits <= bit) for bit in np.sort(bit_idx)]
                    pred = ad.outliers[message_idx][2]
                    if ad_algorithm=='threshold':
                        pred = []
                        cwt_vec = []
                        for f in ad.fields:
                            field_diff = np.array(ad.compute_fields_diff(message_decoded_corr, idx_corr[0], message_idx, f))
                            cwt_vec.append(field_diff[0])
                            if field_diff[0]>0.5 and field_diff[1]>0.5: pred.append(f)
                        '''# Plot wavelet transform visualisation 
                        if j==0 and field[0]==7: # when longitude field is damaged
                            fields = ["MMSI","Navigational\nstatus", "Rate of turns","Speed over\nground",
                                "Longitude", "Latitude","Course over\nground","True\nheading", "Special\nmanouvre\nindicator"]
                            fig, ax = plt.subplots(ncols=2, nrows=2)
                            # Plot damaged fields message by message         
                            messages = np.zeros_like(corruption.indices_corrupted)
                            messages[message_idx] = 1
                            messages = messages[idx_corr[0]==ad.outliers[message_idx][1]]
                            messages = np.multiply(messages, message_decoded_corr[idx_corr[0]==ad.outliers[message_idx][1],field[0]])
                            messages[messages == 0] = None
                            ax[0,0].scatter(range(messages.shape[0]), message_decoded_corr[idx_corr[0]==ad.outliers[message_idx][1],field[0]])
                            ax[0,0].scatter(range(messages.shape[0]), messages, color='r')
                            ax[0,0].set_title("Damaged field: Longitude")
                            ax[0,0].set_xlabel("Number of a consecutive message")
                            ax[0,0].legend(["Original values", "Corrupted value"])
                            # Plot the result of a wavelet transform
                            with_ = message_decoded_corr[idx_corr[0]==ad.outliers[message_idx][1],7] # from all messages from the determined cluster
                            without = np.delete(message_decoded_corr,message_idx,axis=0)  # and without the damaged message
                            without = without[np.delete(idx_corr[0],message_idx,axis=0) == ad.outliers[message_idx][1],7]
                            if len(with_)>1: with_ = abs(with_[1:len(with_)]-with_[0:len(with_)-1]) # compute the derivative
                            scale = max(with_) # normalize
                            if scale and sum(with_): with_ = with_/scale 
                            with_cwt = abs(signal.cwt(with_, signal.morlet2, np.array([1,3])))
                            if len(without)>1: without = abs(without[1:len(without)]-without[0:len(without)-1])
                            if scale and sum(with_): without = without/scale
                            without_cwt = abs(signal.cwt(without, signal.morlet2, np.array([1,3])))
                            ax[0,1].plot(with_cwt[0,:])
                            ax[0,1].plot(without_cwt[0,:])
                            ax[0,1].set_title("Wavelet transform of a damaged field")
                            ax[0,1].set_xlabel("Time dimension")
                            ax[0,1].legend(["With corrupted value", "Without corrupted value"])
                            # Plot relative differences
                            gs = ax[1, 0].get_gridspec()
                            for ax in ax[1, 0:2]: # remove the underlying axes
                                ax.remove()
                            axbig = fig.add_subplot(gs[1, 0:2])
                            axbig.bar(fields, cwt_vec, width=0.5)
                            axbig.set_title("Relative difference in wavelet transform's max of each field")
                            axbig.set_xlabel("Fields")
                            axbig.set_xticks(range(len(fields)))
                            axbig.set_xticklabels(fields, rotation=45)
                            real = np.zeros_like(cwt_vec)
                            f = outliers.fields.index(field[0])
                            real[f] = 1
                            axbig.text(f-0.075, cwt_vec[f]+0.01*cwt_vec[f], str(round(cwt_vec[f],3)), color='r')
                            axbig.bar(fields, np.multiply(cwt_vec,real), width=0.5, color='r')
                            fig.set_tight_layout(True)
                            fig.show() '''                        
                elif stage=='ad_multielement':
                    ad.detect_in_multielement_clusters(
                        idx=idx_corr[0],
                        message_decoded=message_decoded_corr,
                        timestamp=data.timestamp)  
                    pred = ad.outliers[message_idx][2]

                # Compute results
                if stage=='clustering' or stage=='ad_1element':
                    for j in range(len(idx)):
                        OK_vec[file_num, num_bit, -1-j, i] = check_cluster_assignment(idx[j], idx_corr[j], message_idx)
                if stage=='ad_1element' or stage=='ad_multielement':
                    accuracies = calculate_ad_accuracy(field, pred)
                    OK_vec[file_num, num_bit, 0, i] = accuracies["recall"]
                    OK_vec[file_num, num_bit, 1, i] = accuracies["precision"]
                    OK_vec[file_num, num_bit, 2, i] = accuracies["jaccard"]
                    OK_vec[file_num, num_bit, 3, i] = accuracies["hamming"]
    OK_vec = np.mean(OK_vec, axis=3)*100
        

# Visualisation
print(" Complete.")
if stage=='clustering': 
    fig, ax = plt.subplots()
    ax.plot(np.arange(OK_vec.shape[1])+1, OK_vec[0,:,0], color='r')
    ax.plot(np.arange(OK_vec.shape[1])+1, OK_vec[0,:,1], color='b')
    #ax.set_title("Percentage of correctly assigned messages vs amount of damaged bits")
    ax.set_xlabel("Amount of damaged bits")
    ax.set_ylabel("Percentage of correctly assigned messages [%]")
    ax.legend(["DBSCAN", "k-means"])
    fig.show()
if stage == 'ad_1element' or stage == 'ad_multielement':
    print("For "+ ad_algorithm + ", with 1 bit damaged:")
    print("- Recall - Gdansk: " + str(round(OK_vec[0,0,0],2)) + "%, Baltic: " + str(round(OK_vec[1,0,0],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,0],2)) + "%")
    print("- Precision - Gdansk: " + str(round(OK_vec[0,0,1],2)) + "%, Baltic: " + str(round(OK_vec[1,0,1],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,1],2)) + "%")
    print("- Jaccard - Gdansk: " + str(round(OK_vec[0,0,2],2)) + "%, Baltic: " + str(round(OK_vec[1,0,2],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,2],2)) + "%")
    print("- Hamming - Gdansk: " + str(round(OK_vec[0,0,3],2)) + "%, Baltic: " + str(round(OK_vec[1,0,3],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,0,3],2)) + "%")
    if stage == 'ad_1element': print("- Cluster assignment accuracy - Gdansk: " + str(round(OK_vec[0,0,4],2)) + "%, Baltic: " 
    + str(round(OK_vec[1,0,4],2)) + "%, Gibraltar: " + str(round(OK_vec[2,0,4],2)) + "%")
    print("For "+ ad_algorithm + ", with 2 bits damaged:")
    print("- Recall - Gdansk: " + str(round(OK_vec[0,1,0],2)) + "%, Baltic: " + str(round(OK_vec[1,1,0],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,0],2)) + "%")
    print("- Precision - Gdansk: " + str(round(OK_vec[0,1,1],2)) + "%, Baltic: " + str(round(OK_vec[1,1,1],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,1],2)) + "%")
    print("- Jaccard - Gdansk: " + str(round(OK_vec[0,1,2],2)) + "%, Baltic: " + str(round(OK_vec[1,1,2],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,2],2)) + "%")
    print("- Hamming - Gdansk: " + str(round(OK_vec[0,1,3],2)) + "%, Baltic: " + str(round(OK_vec[1,1,3],2)) + "%, Gibraltar: "
    + str(round(OK_vec[2,1,3],2)) + "%")
    if stage == 'ad_1element': print("- Cluster assignment accuracy - Gdansk: " + str(round(OK_vec[0,1,4],2)) + "%, Baltic: " 
    + str(round(OK_vec[1,1,4],2)) + "%, Gibraltar: " + str(round(OK_vec[2,1,4],2)) + "%")


# Save results
if precomputed == '2':
    input("Press Enter to exit...")
else:
    input("Press Enter to save and exit...")
    if stage == 'clustering':
        if os.path.exists('research_and_results/01a_performance_clustering.h5'):
            os.remove('research_and_results/01a_performance_clustering.h5')
        file = h5py.File('research_and_results/01a_performance_clustering.h5', mode='a')
    elif stage == 'ad_1element':
        if os.path.exists('research_and_results/01a_performance_1element_'+ad_algorithm+'.h5'):
            os.remove('research_and_results/01a_performance_1element_'+ad_algorithm+'.h5')
        file = h5py.File('research_and_results/01a_performance_1element_'+ad_algorithm+'.h5', mode='a')
    elif stage == 'ad_multielement':
        if os.path.exists('research_and_results/01a_performance_multielement_'+ad_algorithm+'.h5'):
            os.remove('research_and_results/01a_performance_multielement_'+ad_algorithm+'.h5')
        file = h5py.File('research_and_results/01a_performance_multielement_'+ad_algorithm+'.h5', mode='a')
    file.create_dataset('OK_vec', data=OK_vec)
    file.close()