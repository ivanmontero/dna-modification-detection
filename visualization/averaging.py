# TODO: Refactor to use sequences

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np 
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
parser.add_argument("-o", "--outdir", help="The directory to hold output.")
args = parser.parse_args()

# The chromosome file to parse
FILE = "l_tarentolae.tsv"
if args.file:
    FILE = args.file

# The directory to store plots
PLOT_DIR = "averaging/"
if args.outdir:
    PLOT_DIR = args.outdir
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Create the pandas file
DATA = pd.read_table(FILE).dropna()

def plot_and_save(avg, name, title):
    """
    Plots and saves the sequences with reduced dimensions.

    Args:
        dim_red: The sequences, after dimensionality reduction.
        name: The name of the graph
    """
    print("plot_s " + name)

    # Plot lock, if we are plotting in a parallel manner.
    lock.acquire()
    # colors = is_peak.map(lambda x: 'r' if x else 'b')
    plt.plot([i - int(len(avg)/2) for i in range(int(len(avg)))],
              avg)
    # plt.scatter(dim_red[:,0], dim_red[:,1], c=colors, s=1)
    plt.suptitle(title)
    # Save the plot
    plt.savefig(PLOT_DIR + name + ".png", dpi=1600)
    plt.cla()
    plt.close("all")
    lock.release()

# f (Filter):
#   0 - None
#   1 - True Positives
#   2 - False Positives
def plot_chromosome(data, peaks=1000, base=None, f=0, sequence_radius=10):
    print("plot_chromosome")
    
    if f == 1:
        data = data[data["Max IPD"] > 10]
    elif f == 2:
        data = data[data["Max IPD"] <= 10]

    # ===== Create Sequences ===== 
    sequences, positives = get_peaks(data, peaks, sequence_radius, base)

    # ===== Dimensionality Reduction =====
    # reduced = reduce_dimensions(sequences, dim_red_type)
    avg = [0.0] * (sequence_radius*2 + 1)
    index = 0
    for sequence in sequences:
        for i in range(sequence_radius*2 + 1):
            avg[i] += sequence.iloc[i]
        if (index+1) % (peaks//10) == 0:
            avg_i = map(lambda x: x / (index+1), avg)
            name = \
                "%d_%s_%s" \
                % (index+1, \
                "all" if base is None else base, \
                "all" if f == 0 else \
                ("tp" if f == 1 else
                "fp"))
            title = \
                "Peaks: %s, Base: %s, %s" \
                % (index+1, \
                "All" if base is None else base, \
                "All Peaks" if f == 0 else \
                ("True Positives" if f == 1 else \
                "False Positives"))
            plot_and_save(avg_i, name, title)
        index += 1

def get_window(center, window_radius, table):
    window = table[(table["Chromosome"] == center["Chromosome"])
                   & (table["Position"] >= center["Position"] - window_radius)
                   & (table["Position"] <= center["Position"] + window_radius)]
    return window if len(window) == (2*window_radius+1) else None

# get_peaks(data, peaks, sequence_radius, base)
def get_peaks(data, num_windows, window_radius, base):
    print("get_peaks")
    positives = data.sort_values(by=["Max IPD"], ascending=False)
    if base is not None:
        positives = positives[positives["Base"] == base]
    j_windows = []
    j_window_index = 0
    while len(j_windows) < num_windows:
        center = positives.iloc[j_window_index]
        ipd_selected = \
              "IPD Top Ratio" \
              if center["IPD Top Ratio"] > center["IPD Bottom Ratio"] else \
              "IPD Bottom Ratio"
        window = get_window(center, window_radius, data)
        if window is not None:
            j_windows.append(window[ipd_selected])
            j_window_index += 1
        else:
            positives.drop(positives.index[j_window_index], inplace=True)
    return j_windows, positives.iloc[:num_windows]

def wrapper(args):
    plot_chromosome(*args)

def init(l):
    global lock 
    lock = l

print("apply")
# data["Max IPD"] = data.apply(
#     lambda row: max(row["IPD Top Ratio"], row["IPD Bottom Ratio"]), axis=1)
# data["Max IPD"] = np.max(row["IPD Top Ratio"], row["IPD Bottom Ratio"])
DATA["Max IPD"] = pd.concat(
    [DATA["IPD Top Ratio"], DATA["IPD Bottom Ratio"]], axis=1).max(axis=1)


# PEAKS = [100, 200, 300, 400, 500, 750, 1000]
BASES = [None, "T", "A", "C", "G"]
FILTERS = [0, 1, 2]
RADIUS = 200
from multiprocessing import Pool, Lock
if __name__ == "__main__":
    data_in = []
    # for p in PEAKS:
    for b in BASES:
        for f in FILTERS:
            data_in.append((DATA, 1000, b, f, RADIUS))
    print(data_in)
    pl = Lock()
    pool = Pool(os.cpu_count(), initializer=init, initargs=(pl,))
    pool.map(wrapper, data_in)
    pool.close()
    pool.join()
# for f in FILTERS:
#     wrapper((DATA, p, None, f, RADIUS))

# for f in FILTERS:
#     print("filter: " + str(f))
#     plot_chromosome(DATA, f=f)

# wrapper((DATA, 20, None, 0, 25))

# wrapper((20, "pca", 50))
