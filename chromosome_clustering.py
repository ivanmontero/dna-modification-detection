import pandas as pd
import matplotlib.pyplot as plt

# The chromosome file to parse
FILE = "l_tarentolae.tsv"

# Create the pandas file
table = pd.read_table(FILE)

# Filter out the single LtaP_01 chromosome file
c1 = table[table["Chromosome"] == "LtaP_01"].dropna()

# ===== Create Sequences =====

# = Parameters for the sliding window =
# The length of a sequence
SEQUENCE_LENGTH = 50
# The distance moved between sequences
STRIDE = 10
# When a fold change value is considered part of a "peak"
PEAK_THRESHOLD = 10

# The sequences to visualize
sequences = []
# For each sequence, 1 if peak, 0 o.w.
labels = []

# curr* keeps track of current sequence.
curr = []
curr_peaks = []
for index, row in c1.iterrows():
    # Choose a subset to examine.
    if index < 30000:
        continue
    if index > 40000:
        continue
    
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
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
sequences_2d = pca.fit_transform(sequences)

# T-SNE
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
# sequences_2d = tsne.fit_transform(sequences)

# Find max values to normalize:
np_2d = np.asarray(sequences_2d)
m_0 = max(abs(np.min(np_2d, axis=0)[0]), np.max(np_2d, axis=0)[0])
m_1 = max(abs(np.min(np_2d, axis=0)[1]), np.max(np_2d, axis=0)[1])

# Plot all the normalized values
for i in range(len(sequences)):
    peak = labels[i] == 1
    c = 'r' if peak else 'b'
    plt.scatter(sequences_2d[i, 0] / m_0, sequences_2d[i, 1] / m_1, c=c)

# Display the graph.
plt.show()