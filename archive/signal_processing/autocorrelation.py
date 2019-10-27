import pandas as pd
import numpy as np 
import os
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

# The chromosome file to parse
FILE = "l_tarentolae.tsv"

# The chromosome to run cross correlation on.
chromosome = "LtaP_01"

# # The directory to store plots
# PLOT_DIR = "plots/"
# if not os.path.exists(PLOT_DIR):
#     os.makedirs(PLOT_DIR)

# Create the pandas file
TABLE = pd.read_table(FILE)

def create_sequences(table):
    fold_change = [0] * (table["Position"].max() + 1)
    top_ratio = [0] * (table["Position"].max() + 1)
    bottom_ratio = [0] * (table["Position"].max() + 1)
    for index, row in table.iterrows():
        position = row["Position"]
        fold_change[position] = row["Fold Change"]
        top_ratio[position] = row["IPD Top Ratio"]
        bottom_ratio[position] = row["IPD Bottom Ratio"]
    return fold_change, top_ratio, bottom_ratio

def plot_autocorrelation(sequence, name):
    cc = np.correlate(sequence, sequence, "full")
    cc = cc[len(cc)//2 : ]
    plt.plot(cc)
    plt.title(name)
    plt.xlabel("Shift")
    plt.ylabel("Cross Correlation")
    plt.show()


t = TABLE[TABLE["Chromosome"] == chromosome].dropna()
fold_sequence, top_sequence, bottom_sequence = create_sequences(t)
# plot_autocorrelation(fold_sequence, chromosome + " Fold Change Autocorrelation")
# plot_autocorrelation(top_sequence, chromosome + " IPD Top Ratio Autocorrelation")
plot_autocorrelation(bottom_sequence, chromosome + " IPD Bottom Ratio Autocorrelation")




