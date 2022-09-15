# ----------- Library of functions used in research regarding of AIS message reconstruction ----------
# --------------------- not necessarily in the main pipeline -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import sys
sys.path.append(".")
from utils.miscellaneous import count_number


# ----------------------- For clustering phase ------------------------------
def check_cluster_assignment(idx, idx_corr, message_idx):
    """
    Checks if the corrupted message is assigned together with other messages from its vessel
    Arguments:
    - idx - list of indices of clusters assigned to each message, len = num_messages,
    - idx_corr - list of indices of clusters assigned to each message in a corrupted dataset, 
        len = num_messages
    - message_idx - integer scalar, index of a message that was corrupted
    """
    idx_before = idx[message_idx]
    idx_now = idx_corr[message_idx]
    # Find all messages originally clustered with the corrupted message
    indices_original = np.where(idx == idx_before)
    # Find a cluster that contains most of those messages after the corruption
    percentage = []
    _, idx_corr_vec = count_number(idx_corr)
    for i in idx_corr_vec:  # for each cluster in corrupted data
        indices_cluster = np.where(idx_corr == i)  # find messages from that cluster
        intersection = set(indices_original[0]).intersection(indices_cluster[0])  # find messages both in original cluster and examined cluster
        percentage.append(len(intersection)/len(indices_original[0]))  # calculate how many messages from the original cluster are in examined cluster
    idx_preferable = idx_corr_vec[percentage.index(max(percentage))]  # the cluster with the biggest percentage is probably the right one
    # Check if that cluster is the same as before
    result = idx_now == idx_preferable
    return result


# ------------------- For anomaly detection phase ---------------------------
def visualize_corrupted_bits(OK_vec_all, titles):
    """
    Plots the results of damaging certain bits in AIS message
    Arguments: 
    - OK_vec_all - numpy array containing percentages of correctness regarding each bit
        shape = (num_subplots, num_bits)
    - titles - dictionary with titles for each subplot,
        keys are the number of a subplot {'0':"title_for_first_subplot", ...}
    """
    bits = list(range(145))  # create a list of meaningful bits to visualize
    bits.append(148)
    fig, ax = plt.subplots(OK_vec_all.shape[0], sharex=True, sharey=True)
    for i in range(OK_vec_all.shape[0]):
        OK_vec = OK_vec_all[i, :]
        ax[i].set_title(titles[str(i)])  # get titles from the dictionary
        # Plot each meesage fields with a different color - other bits are 0s
        ax[i].bar(bits, np.concatenate((OK_vec[0:6], np.zeros((140))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((6)), OK_vec[6:8], np.zeros((138))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((8)), OK_vec[8:38], np.zeros((108))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((38)), OK_vec[38:42], np.zeros((104))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((42)), OK_vec[42:50], np.zeros((96))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((50)), OK_vec[50:60], np.zeros((86))), axis=0))
        temp = np.zeros(146)
        temp[60] = OK_vec[60]
        ax[i].bar(bits, temp)
        ax[i].bar(bits, np.concatenate((np.zeros((61)), OK_vec[61:89], np.zeros((57))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((89)), OK_vec[89:116], np.zeros((30))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((116)), OK_vec[116:128], np.zeros((18))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((128)), OK_vec[128:137], np.zeros((9))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((137)), OK_vec[137:143], np.zeros((3))), axis=0))
        ax[i].bar(bits, np.concatenate((np.zeros((143)), OK_vec[143:145], np.zeros((1))), axis=0))
        temp = np.zeros(146)
        temp[145] = OK_vec[145]
        ax[i].bar(bits, temp)
        box = ax[i].get_position()
        ax[i].set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax[i].set_xlabel("Index of a damaged bit")
    fig.legend([
                "Message type","Repeat indicator","MMSI","Navigational status", 
                "Rate of turns","Speed over ground","Position accuracy","Longitude", 
                "Latitude","Course over ground","True heading","Time stamp", 
                "Special manouvre indicator", "RAIM-flag"], loc=7)
    fig.show()