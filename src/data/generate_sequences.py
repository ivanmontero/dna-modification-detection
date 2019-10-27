import pandas as pd
import numpy as np 
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.", default="../../data/processed/l_tarentolae.tsv")
parser.add_argument("-o", "--outdir", help="The directory to hold output.", default="../../data/processed/sequences/")
parser.add_argument("-t", "--ipdthreshold", help="The threshold of which produced sequences' centers must be greater than.", type=float, default=5)
parser.add_argument("-n", "--name", help="The postfix to the name of the files produced", default="")
# parser.add_argument("-b", "--bases", help="Output base sequences.", action="store_true")
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

RADIUS = 25
SEQUENCES_FILE = f"sequences{args.name}.npy"
# SEQUENCES_FILE = "sequences_r_25.csv" if not args.bases else "bases_r_25.csv"
CENTERS_FILE = f"centers{args.name}.csv"

# Create the pandas file
DATA = pd.read_table(FILE).dropna()
DATA_BY_CHROMOSOME = {c:DATA[DATA["Chromosome"] == c] for c in DATA["Chromosome"].unique()}
def get_window(center, window_radius):
    df = DATA_BY_CHROMOSOME[center["Chromosome"]]
    window = df[(df["Position"] >= center["Position"] - window_radius)
              & (df["Position"] <= center["Position"] + window_radius)]
    return window if len(window) == (2*window_radius+1) else None

# Sequence: an array of sequences
# Centers: a dataframe
def save(sequences, centers):
    # f = open(DIR + SEQUENCES_FILE, "w+")
    # f.write(
    #     ",".join(
    #         map(str,
    #             [i for i in range(-window_radius, window_radius+1)]))
    #     + "\n")
    # for s in sequences:
    #     f.write(",".join(map(str, s)) + "\n")
    # f.close()
    for col in sequences:
        sequences[col] = np.array(sequences[col])
    np.save(DIR + SEQUENCES_FILE, sequences)
    if not args.no_centers:
        centers.to_csv(DIR + CENTERS_FILE, index=False)

print("apply")
DATA["Max IPD"] = pd.concat(
    [DATA["Top IPD Ratio"], DATA["Bottom IPD Ratio"]], axis=1).max(axis=1)

CHECKPOINT = 1000
IPD_THRESHOLD = args.ipdthreshold

print("filtering")
peaks = DATA[DATA["Max IPD"] > IPD_THRESHOLD].sort_values(by=["Max IPD"], ascending=False)

sequences = {col:[] for col in DATA.columns}
centers = pd.DataFrame() # create empty dataframe

print("creating peaks")
print(peaks.shape[0])
for i in range(peaks.shape[0]):
    print(i)
    row = peaks.iloc[i]
    # selected = \
    #     ("IPD Top Ratio" \
    #     if row["IPD Top Ratio"] > row["IPD Bottom Ratio"] else \
    #     "IPD Bottom Ratio") if not args.bases else \
    #         "Base"
    window = get_window(row, RADIUS)
    if window is not None:
        for column in DATA.columns:
            if column not in ["Chromosome", "Max IPD"]:
                sequences[column].append(window[column].tolist())
        # sequences.append(window[selected].tolist())
        # print(window[selected].tolist())
        centers = centers.append(row, ignore_index=True)
        if len(sequences) % CHECKPOINT == 0:
            print(len(sequences))
            save(sequences, centers, RADIUS)

negatives = DATA[(DATA["Fold Change"] < 10) & (DATA["Max IPD"] < IPD_THRESHOLD)].sample(peaks.shape[0] // 2)
for i in range(negatives.shape[0]):
    print(i)
    row = negatives.iloc[i]
    # selected = \
    #     ("IPD Top Ratio" \
    #     if row["IPD Top Ratio"] > row["IPD Bottom Ratio"] else \
    #     "IPD Bottom Ratio") if not args.bases else \
    #         "Base"
    window = get_window(row, RADIUS)
    if window is not None:
        for column in DATA.columns:
            if column not in ["Chromosome", "Position", "Max IPD"]:
                sequences[column].append(window[column].tolist())
        # sequences.append(window[selected].tolist())
        # print(window[selected].tolist())
        centers = centers.append(row, ignore_index=True)
        if len(sequences) % CHECKPOINT == 0:
            print(len(sequences))
            save(sequences, centers, RADIUS)

save(sequences, centers)


# columns = ["Chromosome",
#            "Position",
#            "Base", 
#            "Fold Change",
#            "Top Score",
#            "Top Trimmed Mean",
#            "Top Trimmed Error",
#            "Top Null Prediction",
#            "Top IPD Ratio",
#            "Top Coverage",
#            "Bottom Score",
#            "Bottom Trimmed Mean",
#            "Bottom Trimmed Error",
#            "Bottom Null Prediction",
#            "Bottom IPD Ratio",
#            "Bottom Coverage"]