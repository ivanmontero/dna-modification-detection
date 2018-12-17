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
    for index, row in table.iterrows():
        position = row["Position"]
        fold_change[position] = row["Fold Change"]
        top_ratio[position] = row["IPD Top Ratio"]
    return fold_change, top_ratio

t = TABLE[TABLE["Chromosome"] == chromosome].dropna()
fold_sequence, top_sequence = create_sequences(t)

## Find cross correlation
#f1 = fft(fold_sequence)
#f2 = fft(np.flipud(top_sequence))
#cc = np.real(ifft(f1 * f2))

cc = np.correlate(top_sequence, fold_sequence, "same")

zero_index = int(len(cc)/2)

print(np.argmax(cc) - zero_index)

plt.plot([i - zero_index for i in range(len(cc))],cc)
plt.title(chromosome + " Cross Correlation")
plt.xlabel("Shift")
plt.ylabel("Cross Correlation")
plt.show()
