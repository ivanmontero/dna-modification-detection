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
PLOT_DIR = "peak_clustering_plots/"
if args.outdir:
    PLOT_DIR = args.outdir
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Create the pandas file
data = pd.read_table(FILE).dropna()

def reduce_dimensions(sequences, red_type):
    """
    Runs dimensionality reduction on the following array of sequences.
    
    Args:
        sequences: The sequences which to run dimensionality reduction on.
        red_type: The name of the type of dimensionality reduction. Must be
            either "pca" or "tsne"

    Returns:
        reduced: The reduced form of the sequences.
    """
    print("dim_red")
    reduced = []

    if red_type == "pca":
        pca = PCA()
        reduced = pca.fit_transform(sequences)
    
    if red_type == "tsne":
        tsne = TSNE()
        reduced = tsne.fit_transform(sequences)
    
    return reduced

def plot_and_save(dim_red, is_peak, name):
    """
    Plots and saves the sequences with reduced dimensions.

    Args:
        dim_red: The sequences, after dimensionality reduction.
        name: The name of the graph
    """
    print("plot_s " + name)

    # Plot lock, if we are plotting in a parallel manner.
    lock.acquire()
    colors = is_peak.map(lambda x: 'r' if x else 'b')
        
    plt.scatter(dim_red[:,0], dim_red[:,1], c=colors, s=1)
        
    # Save the plot
    plt.savefig(PLOT_DIR + name + ".png")
    plt.cla()
    plt.close("all")
    lock.release()

def plot_chromosome(sequence_radius=10, dim_red_type="pca", peaks=100):
    print("plot_chromosome")

    # ===== Create Sequences ===== 
    j_windows_rows, sequences, positives = get_peaks(peaks, sequence_radius)

    # ===== Dimensionality Reduction =====
    reduced = reduce_dimensions(sequences, dim_red_type)

    name = str(sequence_radius) + "_" + dim_red_type
    name += "_" + str(peaks)
    
    is_peak = positives["Fold Change"].map(lambda x: x > 10)

    plot_and_save(reduced, is_peak, name)

def get_window(center, window_radius, table=data):
    window = data[(data["Chromosome"] == center["Chromosome"])
                   & (data["Position"] >= center["Position"] - window_radius)
                   & (data["Position"] <= center["Position"] + window_radius)]
    return window if len(window) == (2*window_radius+1) else None

def get_peaks(num_windows, window_radius):
    print("get_peaks")
    positives = data.sort_values(by=["Max IPD"], ascending=False)
    j_windows = []
    j_windows_rows = []
    j_window_index = 0
    while len(j_windows_rows) < num_windows:
        center = positives.iloc[j_window_index]
        ipd = "IPD Top Ratio" \
              if center["IPD Top Ratio"] > center["IPD Bottom Ratio"] else \
              "IPD Bottom Ratio"
        window = get_window(center, window_radius)
        if window is not None:
            j_windows_rows.append(window)
            j_windows.append(window[ipd])
            j_window_index += 1
        else:
            positives.drop(positives.index[j_window_index], inplace=True)
    return j_windows_rows, j_windows, positives.iloc[:num_windows]

def wrapper(args):
    plot_chromosome(*args)

def init(l):
    global lock 
    lock = l

print("apply")
# data["Max IPD"] = data.apply(
#     lambda row: max(row["IPD Top Ratio"], row["IPD Bottom Ratio"]), axis=1)
# data["Max IPD"] = np.max(row["IPD Top Ratio"], row["IPD Bottom Ratio"])
data["Max IPD"] = pd.concat(
    [data["IPD Top Ratio"], data["IPD Bottom Ratio"]], axis=1).max(axis=1)

print("filter")
#data = data[data["Chromosome"] == "LtaP_01"]
SEQUENCE_RADII = [5, 10, 15, 20, 40, 80, 160]
PEAKS = [100, 200, 300, 400, 500, 750, 1000]
#SEQUENCE_RADII = [20]
#PEAKS = [50]
from multiprocessing import Pool, Lock
if __name__ == "__main__":
    data_in = []
    for s in SEQUENCE_RADII:
        for p in PEAKS:
            data_in.append((s, "pca", p))
            data_in.append((s, "tsne", p))
    print(data_in)
    pl = Lock()
    pool = Pool(os.cpu_count(), initializer=init, initargs=(pl,))
    pool.map(wrapper, data_in)
    pool.close()
    pool.join()

# wrapper((20, "pca", 50))
