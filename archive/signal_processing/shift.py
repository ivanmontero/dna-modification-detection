# Ivan Montero

# ========== Imports ==========

# Boilerplate
import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os

# SciKit Learn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, KernelPCA
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# plt.ioff()

# Commandline arguments.
from argparse import ArgumentParser

# ========== Command Arguments ==========
parser = ArgumentParser()

# Data related -- Required
parser.add_argument("-c", "--centers", required=True,
                    help="The file containing center IPD info.")
parser.add_argument("-s", "--sequences", required=True,
                    help="The file containing sequences.")
args = parser.parse_args()

# ========== Run Setup ==========

# Definitely not coverage or score
FEATURES = [
            "Bottom Coverage",
            "Bottom IPD Ratio",
            # "Bottom Null Prediction",
            # "Bottom Score",
            "Bottom Trimmed Error",
            # "Bottom Trimmed Mean",
            "Top Coverage",
            "Top IPD Ratio",
            # "Top Null Prediction",
            # "Top Score",
            "Top Trimmed Error",
            # "Top Trimmed Mean",
]

def prepare_input(sequences):
    ss = []
    for s in sequences:
        if s in FEATURES:
            ss.append(scale(sequences[s]))
    ss.append(pd.get_dummies(pd.DataFrame(sequences["Base"])).values)
    data = np.concatenate(ss, axis=1)
    return data

def get_resources():
    c = pd.read_csv(args.centers)
    s = np.load(args.sequences)[()]
    return c, s

def get_offsets(ipd_sequences):
    # ipd_sequences = scale(ipd_sequences)
    ipd_sequences = (ipd_sequences - np.mean(ipd_sequences)) / np.std(ipd_sequences)
    # ipd_sequences /= np.std(ipd_sequences)
    # ipd_sequences = (ipd_sequences - np.mean(ipd_sequences, axis=1)[:,np.newaxis]) / np.std(ipd_sequences, axis=1)[:,np.newaxis]
    radius = ipd_sequences.shape[1] // 2
    cc_sum = []
    for start in range(len(ipd_sequences)):
        cc = 0.0
        for i in range(len(ipd_sequences)):
            cc += np.squeeze(np.correlate(ipd_sequences[start], ipd_sequences[i]))
            # res = np.correlate(sequences[start], sequences[i], "same")
            # cc_val += np.max(res)
        cc_sum.append(cc)
    max_cc = np.argmax(cc_sum)

    print(max_cc)
    print(f"pre: {np.argmax(ipd_sequences[max_cc]) - radius}")
    max_seq = np.roll(ipd_sequences[max_cc], -(np.argmax(ipd_sequences[max_cc]) - radius))
    # plt.xticks(np.arange(-ipd_sequences.shape[1]//2+1, ipd_sequences.shape[1]//2+1, 1.0))
    # plt.plot(np.arange(-ipd_sequences.shape[1]//2+1, ipd_sequences.shape[1]//2+1, 1.0), max_seq)
    # plt.show()
    print(f"post: {np.argmax(max_seq) - radius}")
    # print(max_cc)
    # TODO: MOve so that 0 is the max.
    
    cc_i = []
    for i in range(len(ipd_sequences)):
        res = np.correlate(max_seq, ipd_sequences[i], "same")
        cc_i.append(np.argmax(res) - radius)
        # print(np.argmax(np.roll(ipd_sequences[i], cc_i[-1])))
    # print(cc_i)
    return cc_i

# CLips half the radius
def shift_clip(sequences, shifts):
    shifted = []
    for i in range(sequences.shape[0]):
        shifted.append(np.roll(sequences[i], shifts[i]))
    radius = sequences.shape[1] // 2
    print(np.array(shifted)[:, radius // 2 : radius + radius // 2+1].shape)
    return np.array(shifted)[:, radius // 2 : radius + radius // 2+1]

def view_base_dist(bases):
    # bases = sequences["Base"][idx]
    # new_bases = shift_pad(bases, radius, offsets, "", keep_dims=True)
    # print(new_bases)
    # bases = bases[bases[:, bases.shape[1]//2+1] == "T"]
    
    # f = open("tp_top_shifted_bases.txt", "w+")
    # for i in range(len(bases)):
    #     for j in range(len(bases[i])):
    #         f.write(bases[i,j])
    #     f.write("\n")
    # f.close()
    vals = {}
    for base in ["T", "A", "C", "G"]:
        vals[base] = np.sum(np.where(bases == base, 1, 0), axis=0)
        vals[base] = (vals[base] - np.mean(vals[base])) / np.std(vals[base])
        plt.plot(np.arange(-bases.shape[1]//2+1, bases.shape[1]//2+1), vals[base], label=base)
    plt.xticks(np.arange(-bases.shape[1]//2+1, bases.shape[1]//2+1, 1.0))
    plt.legend()
    plt.show()

def view_average(sequences, name):
    sequences = (sequences - np.mean(sequences, axis=1)[:,np.newaxis]) / np.std(sequences, axis=1)[:,np.newaxis]
    print(sequences.shape)
    # sequences = sequences[sequences[:, sequences.shape[1]//2+1] == "T"]
    average = np.mean(sequences, axis=0)
    figure(num=1, figsize=(20, 6))
    plt.plot(np.arange(-sequences.shape[1]//2+1, sequences.shape[1]//2+1), average)
    plt.xticks(np.arange(-sequences.shape[1]//2+1, sequences.shape[1]//2+1, 1.0))
    # plt.show()
    plt.title(name)
    plt.savefig(name + ".png", dpi=400)
    plt.clf()

# Give a larger radius, it will return a shifted one with half the radius.
def run():
    radius = sequences["Top IPD Ratio"].shape[1] // 2
    new_radius = radius // 2

    strand = {
        "ts" : (c["Top IPD Ratio"] > c["Bottom IPD Ratio"]),
        "bs" : (c["Top IPD Ratio"] < c["Bottom IPD Ratio"])
    }

    label = {
        "tp" : (c["Fold Change"] > 10),
        "fp" : (c["Fold Change"] < 10)
    }

    for l in label:
        for s in strand:
            idx = strand[s] & label[l]
            st = "Top IPD Ratio" if s == "ts" else "Bottom IPD Ratio"
            offsets = get_offsets(sequences[st][idx][:,radius-new_radius-1:radius+new_radius])
            print("o")
            shifted = shift_clip(sequences['Base'][idx], offsets)
            print("s")
            view_average(shift_clip(sequences[st][idx], offsets), f"{l}_{s}_full_demean")
            
            f = open(f"{l}_{s}_shifted_bases_full_demean.txt", "w+")
            for i in range(len(shifted)):
                for j in range(len(shifted[i])):
                    f.write(shifted[i,j])
                f.write("\n")
            f.close()
            

    # for idx_name in idxs:
    #     idx = idxs[idx_name]
    #     offsets = get_offsets(sequences["Top IPD Ratio"][idx][:,radius-new_radius-1:radius+new_radius])
    #     # print(f"shift: {np.sum(sequences['Base'][idx][:, radius // 2 : radius + radius // 2+1] != shift_clip(sequences['Base'][idx], offsets))}")
    #     f = open(idx + "_shifted_bases.txt", "w+")
    #     for i in range(len(bases)):
    #         for j in range(len(bases[i])):
    #             f.write(bases[i,j])
    #         f.write("\n")
    #     f.close()
    
    # view_base_dist(shift_clip(sequences["Base"][idx], offsets))
    # view_average(shift_clip(sequences["Top IPD Ratio"][idx], offsets))
    # print(offsets)
    new_seq = {}
    for column in sequences:
        new_seq[column] = sequences[column]

    labels = c["Fold Change"].map(lambda x: 1 if x > 10 else 0).values
    # dim_red(sequences["Top IPD Ratio"], offsets, labels, radius)


# ========== Main ==========
def init(c, s):
    global centers, sequences, padding
    centers = c
    sequences = s

if __name__ == "__main__":
    c, s = get_resources()
    init(c, s)
    run()