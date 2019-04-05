import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
parser.add_argument("-o", "--outdir", help="The directory to output to.")
args = parser.parse_args()

# The chromosome file to parse
FILE = "l_tarentolae.tsv" if not args.file else args.file

# Load in the data.
data = pd.read_table(FILE).dropna()
data.describe()

PLOT_DIR = "./peak_plots_full/" if not args.outdir else args.outdir
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Plot a gray rectangle showing where N/A values are.
def plot_na(table, ax):
    color = "tab:grey"
    positions = table["Position"]
    for i in range(1, positions.size):
        if positions.iloc[i] - positions.iloc[i-1] > 1:
            ax.axvspan(positions.iloc[i-1], positions.iloc[i], alpha=.5, facecolor=color)

def plot(table, title, name):
    # Set up a plot for the Fold Change
    color = "tab:orange"
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Fold Change')
    ax1.plot(table["Position"], table["Fold Change"], color=color, label="Fold Change", linewidth=1.0)
    ax1.tick_params(axis='y')

    # Create a second plot that shares the same x axis
    ax2 = ax1.twinx()

    # Set up the plot for the IPD Values
    color = "tab:red"
    ax2.set_ylabel("IPD Value")
    ax2.plot(table["Position"], table["IPD Top Ratio"], color=color, label="IPD Top Ratio", linewidth=1.0)
    color = "tab:blue"
    ax2.plot(table["Position"], table["IPD Bottom Ratio"], color=color, label="IPD Bottom Ratio", linewidth=1.0)
    ax2.tick_params(axis='y')
    
    plot_na(table, ax2)

    plt.suptitle(title)
    plt.legend()
    plt.savefig(PLOT_DIR + name + ".png", dpi=600)
    plt.cla()
    plt.close()

def get_window(center, window_radius, table=data):
    window = data[(data["Chromosome"] == center["Chromosome"])
                   & (data["Position"] >= center["Position"] - window_radius)
                   & (data["Position"] <= center["Position"] + window_radius)]
    # return window if len(window) == (2*window_radius+1) else None
    return window

J_WINDOWS = 500
WINDOW_RADIUS = 200

def get_peaks(num_windows, window_radius,  sort):
    positives = data.sort_values(by=[sort], ascending=False)
    j_windows_rows = []
    j_window_index = 0
    while len(j_windows_rows) < num_windows:
        window = get_window(positives.iloc[j_window_index], window_radius)
        j_windows_rows.append(window)
        j_window_index += 1
    return j_windows_rows, positives

data["Max IPD"] = data.apply(
    lambda row: max(row["IPD Top Ratio"], row["IPD Bottom Ratio"]), axis=1)
windows, positives = get_peaks(J_WINDOWS, WINDOW_RADIUS, "Max IPD")

for i in range(J_WINDOWS):
    center = positives.iloc[i]
    title = \
        "Peak #%d, Chromosome: %s, Strand: %s, Position: %d, Base: %s" \
        % (i, \
           center["Chromosome"], \
           "Top" if center["IPD Top Ratio"] > center["IPD Bottom Ratio"] else "Bottom", \
           center["Position"], \
           center["Base"])
    name = \
        "%d_%s_%s_%d_%s" \
        % (i, \
           center["Chromosome"], \
           "Top" if center["IPD Top Ratio"] > center["IPD Bottom Ratio"] else "Bottom", \
           center["Position"], \
           center["Base"])
    if i % 100 == 0:
        print(i)
    plot(windows[i], title, name)

