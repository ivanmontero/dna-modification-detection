import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# The chromosome file to parse
FILE = "l_tarentolae.tsv"

# The directory to store plots
PLOT_DIR = "plots/"

# Create the pandas file
TABLE = pd.read_table(FILE)

# = Parameters for the sliding window =
# The length of a sequence
SEQUENCE_LENGTH = 50
# The distance moved between sequences
STRIDE = 10
# When a fold change value is considered part of a "peak"
PEAK_THRESHOLD = 10

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
    
from threading import Lock
PLOT_LOCK = Lock()
def create_plot(dim_red, labels, name):
    global PLOT_LOCK, PLOT_DIR
    PLOT_LOCK.acquire()
    # Find max values to normalize:
    np_2d = np.asarray(dim_red)
    m_0 = max(abs(np.min(np_2d, axis=0)[0]), np.max(np_2d, axis=0)[0])
    m_1 = max(abs(np.min(np_2d, axis=0)[1]), np.max(np_2d, axis=0)[1])
        
    # Plot all the normalized values
    for i in range(len(dim_red)):
        peak = labels[i] == 1
        c = 'r' if peak else 'b'
        plt.scatter(dim_red[i, 0] / m_0, dim_red[i, 1] / m_1, c=c, s=1)
        
    # Display the graph.
    plt.savefig(PLOT_DIR + name + ".png")
    PLOT_LOCK.release()

# For each chromosome, run this:
# for chromosome in table["Chromosome"].unique():

def plot_chromosome(chromosome):
    global TABLE, SEQUENCE_LENGTH, STRIDE, PEAK_THRESHOLD
    # Filter out the single LtaP_01 chromosome file
    c = TABLE[TABLE["Chromosome"] == chromosome].dropna()
    
    # ===== Create Sequences ===== 
    # The sequences to visualize
    sequences = []
    # For each sequence, 1 if peak, 0 o.w.
    labels = []
    
    # curr* keeps track of current sequence.
    curr = []
    curr_peaks = []
    for index, row in c.iterrows(): 
        curr.append(row["IPD Top Ratio"])
        curr_peaks.append(1 if row["Fold Change"] > PEAK_THRESHOLD else 0)
    
        # Make sure we have a sequence of appropriate length
        if len(curr) == SEQUENCE_LENGTH:
            # Add to our sequences if the stride is followed.
            if index % STRIDE == 0:
                sequences.append(curr)
                labels.append(sum(curr_peaks) == SEQUENCE_LENGTH)
            
            curr.pop(0)
            curr_peaks.pop(0)
    
    
    # ===== Dimensionality Reduction =====
    import numpy as np 
    
    # PCA
    pca = PCA()
    sequences_2d_pca = pca.fit_transform(sequences)
    create_plot(sequences_2d_pca, labels, chromosome + "_pca")
    
    # T-SNE
    tsne = TSNE()
    sequences_2d_tsne = tsne.fit_transform(sequences)
    create_plot(sequences_2d_tsne, labels, chromosome + "_tsne")

from multiprocessing import Pool
import os
if __name__ == "__main__":
    pool = Pool(os.cpu_count())
    pool.map(plot_chromosome, TABLE["Chromosome"].unique())
