import pandas as pd
import numpy as np 
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
parser.add_argument("-o", "--outdir", help="The directory to hold output.")
parser.add_argument("-b", "--bases", help="Output base sequences.", action="store_true")
parser.add_argument("-nc", "--no_centers", help="Don't output centers.", action="store_true")
args = parser.parse_args()

# The chromosome file to parse
FILE = "l_tarentolae.tsv"
if args.file:
    FILE = args.file

# The directory to store plots
DIR = "windows/"
if args.outdir:
    DIR = args.outdir
if not os.path.exists(DIR):
    os.makedirs(DIR)

SEQUENCES_FILE = "sequences_r_100.csv" if not args.bases else "bases_r_100.csv"
CENTERS_FILE = "centers_r_100.csv"

# Create the pandas file
DATA = pd.read_table(FILE).dropna()

def get_window(center, window_radius):
    window = DATA[(DATA["Chromosome"] == center["Chromosome"])
                   & (DATA["Position"] >= center["Position"] - window_radius)
                   & (DATA["Position"] <= center["Position"] + window_radius)]
    return window if len(window) == (2*window_radius+1) else None

# Sequence: an array of sequences
# Centers: a dataframe
def save(sequences, centers, window_radius):
    f = open(DIR + SEQUENCES_FILE, "w+")
    f.write(
        ",".join(
            map(str,
                [i for i in range(-window_radius, window_radius+1)]))
        + "\n")
    for s in sequences:
        f.write(",".join(map(str, s)) + "\n")
    f.close()
    if not args.no_centers:
        centers.to_csv(DIR + CENTERS_FILE, index=False)

print("apply")
DATA["Max IPD"] = pd.concat(
    [DATA["IPD Top Ratio"], DATA["IPD Bottom Ratio"]], axis=1).max(axis=1)

CHECKPOINT = 1000
IPD_THRESHOLD = 5
RADIUS = 100

print("filtering")
peaks = DATA[DATA["Max IPD"] > IPD_THRESHOLD].sort_values(by=["Max IPD"], ascending=False)

sequences = []
centers = pd.DataFrame() # create empty dataframe

print("creating peaks")
print(peaks.shape[0])
for i in range(peaks.shape[0]):
    row = peaks.iloc[i]
    selected = \
        ("IPD Top Ratio" \
        if row["IPD Top Ratio"] > row["IPD Bottom Ratio"] else \
        "IPD Bottom Ratio") if not args.bases else \
            "Base"
    window = get_window(row, RADIUS)
    if window is not None:
        sequences.append(window[selected].tolist())
        print(window[selected].tolist())
        centers = centers.append(row, ignore_index=True)
        if len(sequences) % CHECKPOINT == 0:
            print(len(sequences))
            save(sequences, centers, RADIUS)
save(sequences, centers)
    