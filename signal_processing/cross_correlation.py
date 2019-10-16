# Ivan Montero

# ========== Imports ==========

# Boilerplate
import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os
from scipy.stats import mode
import matplotlib.pyplot as plt
# Commandline arguments.
from argparse import ArgumentParser

# ========== Command Arguments ==========

parser = ArgumentParser()
parser.add_argument("-r", "--radii", nargs="+", required=True,
                    help="Radii to run classfier on.")
parser.add_argument("-c", "--centers", required=True,
                    help="The file containing center IPD info.")
parser.add_argument("-ts", "--topsequences", required=True,
                    help="The file containing top IPD sequences.")
parser.add_argument("-bs", "--bottomsequences", required=True,
                    help="The file containing bottom IPD sequences.")
parser.add_argument("-b", "--bases", required=True,
                    help="The file containing base sequences.")
parser.add_argument("-o", "--outdir", required=True,
                    help="The directory to output.")

# Data related -- Optional
parser.add_argument("--parallel", action="store_true",
                    help="Run on a multithreaded environment.")
parser.add_argument("--interactive", action="store_true",
                    help="Makes plots show to the user.")
parser.add_argument("-top", "--top", action="store_true",
                    help="Analyze only top")
parser.add_argument("-bottom", "--bottom", action="store_true",
                    help="Analyze only bottom")
args = parser.parse_args()

def resize_and_verify(centers, topsequences, bottomsequences, bases, radius):
    mr = len(topsequences.iloc[0]) // 2
    ts = topsequences.iloc[:,mr-radius:mr+radius+1]
    bs = bottomsequences.iloc[:,mr-radius:mr+radius+1]
    b = bases.iloc[:,mr-radius:mr+radius+1]

    to_keep = ts.isin(ts.dropna()).iloc[:,0]

    c = centers[to_keep]
    ts = ts[to_keep]
    bs = bs[to_keep]
    b = b[to_keep]

    return c, ts, bs, b

def prepare_input(topsequences, bottomsequences, bases):
    if args.top:
        data = topsequences
    elif args.bottom:
        data = bottomsequences
    else:
        data = np.concatenate((topsequences, bottomsequences), axis=1)
    data = np.concatenate((data, pd.get_dummies(bases).values), axis=1)
    return data

def init(l, c, ts, bs, b, r, o):
    global lock, centers, topsequences, bottomsequences, bases, radii, params, outdir
    lock = l
    centers = c
    topsequences = ts
    bottomsequences = bs
    bases = b
    radii = list(map(int,r))
    outdir = o
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

def get_resources():
    # Load in tables
    c = pd.read_csv(args.centers)
    ts = pd.read_csv(args.topsequences)
    bs = pd.read_csv(args.bottomsequences)
    b = pd.read_csv(args.bases)

    # Filter based on params.
    if args.top:
        strand = c["IPD Top Ratio"] > c["IPD Bottom Ratio"]
    elif args.bottom:
        strand = c["IPD Top Ratio"] < c["IPD Bottom Ratio"]
    else:
        strand = c["IPD Top Ratio"] == c["IPD Top Ratio"]   #always true

    return c[strand], ts[strand], bs[strand], b[strand]

# ========== Run Routine ==========

def run():
    # create_cc_matrix()
    # create_padded_cc_matrix()
    # single_cc_matrix()
    # pass

    # def create_cc_matrix():
    radius = radii[0]

    c, ts, bs, b = resize_and_verify(centers, topsequences, bottomsequences, bases, radii[0]*2)    
    idx = (c["Fold Change"] > 10) & (c["IPD Top Ratio"] > c["IPD Bottom Ratio"])
    sequences = ts[idx].values
    sequences = (sequences - np.mean(sequences)) / np.std(sequences)

    cc_sum = []
    cc_i = []
    for start in range(len(sequences)):
        # print(start/len(sequences))
        cc_val = 0.0
        for i in range(len(sequences)):
            # cc += np.squeeze(np.correlate(sequences.iloc[start], sequences.iloc[i]))
            res = np.correlate(sequences[start], sequences[i], "same")
            cc_val += np.max(res)
        cc_sum.append(cc_val)
    max_cc = np.argmax(cc_sum)
    print(max_cc)
    for i in range(len(sequences)):
        res = np.correlate(sequences[max_cc], sequences[i], "same")
        cc_i.append(np.argmax(res) - radius*2)
    print(cc_i)
    res = np.sum(sequences[:,radius:3*radius+1], axis=0) / len(sequences)
    plt.plot([i for i in range(-len(res)//2, len(res)//2)], res)
    shifted = shift_pad(sequences, radius*2, cc_i)
    sequences = np.array(shifted)[:, 3*radius:5*radius+1]

    res = np.sum(sequences, axis=0) / len(sequences)
    plt.plot([i for i in range(-len(res)//2, len(res)//2)], res)
    plt.show()
    
    print(cc_sum)
    print(max(cc_sum))
    print(print(np.argmax(cc_sum)))

def shift_pad(sequences, radius, shifts):
    padded = np.pad(sequences, pad_width=((0,0),(radius, radius)), mode='constant', constant_values=(0, 0))
    shifted = []
    for i in range(padded.shape[0]):
        shifted.append(np.roll(padded[i], shifts[i]))
    return shifted


# ========== Main ==========

if __name__ == "__main__":
    l = Lock()
    c, ts, bs, b = get_resources()

    # if args.parallel:
    #     pool = Pool(os.cpu_count(),
    #                 initializer=init,
    #                 initargs=(l, c, ts, bs, b, args.radii, arg.outdir))
    #     pool.map(run, params)
    #     pool.close()
    #     pool.join()
    # else:   
    #     init(l, c, ts, bs, b, args.radii, args.outdir)
    #     run()
    init(l, c, ts, bs, b, args.radii, args.outdir)
    run()

    

# def create_padded_cc_matrix():

#     cc_total = []
#     cc_i_total = []
#     radius = radii[0]
#     c, ts, bs, b = resize_and_verify(centers, topsequences, bottomsequences, bases, radius)    
#     idx = (c["Fold Change"] > 10) & (c["IPD Top Ratio"] > c["IPD Bottom Ratio"])
#     sequences = ts[idx]

#     cc_mat = np.loadtxt("cc_mat_single.txt")
#     plt.matshow(np.log(cc_mat))
#     plt.show()
#     cc_i_mat = np.loadtxt("cc_i_mat_single.txt", dtype=int)
#     shifts = mode(cc_i_mat, axis=0)[0][0]
#     print(shifts.shape[0])
#     print(len(sequences))
#     shifted = shift_pad(sequences, radius, shifts)
#     sequences = np.array(shifted)[:, radius:3*radius+1]
#     print(sequences.shape)

#     for start in range(len(sequences)):
#         print(start/len(sequences))
#         cc = []
#         cc_i = []

#         for i in range(len(sequences)):
#             res = np.correlate(sequences[start], sequences[i], "same")
#             # cc.append(np.max(res))
#             cc.append(res[radius])
#             cc_i.append(np.argmax(res) - radius)
#         # print(cc)
#         # print(cc_i)
#         # print(mode(cc_i))
#         cc_total.append(cc)
#         cc_i_total.append(cc_i)
#     plt.matshow(np.log(np.array(cc_total)))
#     plt.show()
#     np.savetxt(args.outdir + "cc_mat_shift_single.txt", np.array(cc_total))

# def create_cc_matrix():
#     # plt.matshow(cc_mat)
#     # plt.show()

#     cc_total = []
#     cc_i_total = []
#     radius = radii[0]
#     c, ts, bs, b = resize_and_verify(centers, topsequences, bottomsequences, bases, radius)    
#     idx = (c["Fold Change"] > 10) & (c["IPD Top Ratio"] > c["IPD Bottom Ratio"])
#     sequences = ts[idx]

#     cc_i_mat = np.loadtxt("cc_i_mat.txt", dtype=int)
#     shifts = mode(cc_i_mat, axis=0)[0][0]
#     print(shifts.shape[0])
#     print(len(sequences))
#     # print(shifts.tolist())
#     # print(shifts)
#     # shifted = shift_pad(sequences, radius, shifts)
#     # plt.matshow(np.array(shifted))
    
#     # plt.show()


#     for start in range(len(sequences)):
#         print(start/len(sequences))
#         cc = []
#         cc_i = []


#         # positives = c[(c["Fold Change"] > 10) & (c["IPD Top Ratio"] > c["IPD Bottom Ratio"])]["IPD Top Ratio"].values
#         # p_idx = np.where((c["Fold Change"].values > 10) & (c["IPD Top Ratio"].values >c["IPD Bottom Ratio"].values))


#         for i in range(len(sequences)):
#             res = np.correlate(sequences.iloc[start], sequences.iloc[i], "same")
#             # cc.append(np.max(res))
#             cc.append(res[radius])
#             cc_i.append(np.argmax(res) - radius)
#         # print(cc)
#         # print(cc_i)
#         # print(mode(cc_i))
#         cc_total.append(cc)
#         cc_i_total.append(cc_i)
#     np.savetxt(args.outdir + "cc_mat_single.txt", np.array(cc_total))
#     np.savetxt(args.outdir + "cc_i_mat_single.txt", np.array(cc_i_total), fmt="%d")

#     # padded = np.pad(sequences, pad_width=(radius, radius), mode='constant', constant_values=(0, 0))
#     # # print(padded)
#     # shifted = []
#     # for i in range(padded.shape[0]):
#     #     shifted.append(np.roll(padded[i], cc_i))
#     # # shifted = np.array(shifted)
#     # for i in range(len(shifted)):
#     #     res = np.correlate(shifted[0], shifted[i], "same")
#     #     cc.append(np.max(res))
#     #     cc_i.append(np.argmax(res))
#     # print(cc)
#     # print(cc_i)
#     # print(mode(cc_i))
#     # print(len(cc_i))
