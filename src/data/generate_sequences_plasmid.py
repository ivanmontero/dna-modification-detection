import pandas as pd
import numpy as np 
import os

# The chromosome file to parse
FILE = "../../data/processed/plasmid_and_j.csv"

# The directory to store sequences
DIR = "../../data/processed/"
if not os.path.exists(DIR):
    os.makedirs(DIR)

RADIUS = 25
TOP_SEQUENCES_FILE = f"plasmid_top_sequences.npy"
BOTTOM_SEQUENCES_FILE = f"plasmid_bottom_sequences.npy"
# SEQUENCES_FILE = "sequences_r_25.csv" if not args.bases else "bases_r_25.csv"
TOP_CENTERS_FILE = f"plasmid_top_centers.csv"
BOTTOM_CENTERS_FILE = f"plasmid_bottom_centers.csv"

# Create the pandas file
DATA = pd.read_csv(FILE).dropna()
DATA_BY_CHROMOSOME = {c:DATA[DATA["plasmid"] == c] for c in DATA["plasmid"].unique()}
def get_window(center, window_radius):
    df = DATA_BY_CHROMOSOME[center["plasmid"]]
    window = df[(df["position"] >= center["position"] - window_radius)
              & (df["position"] <= center["position"] + window_radius)
              & (df["strand"] == center["strand"])]
    return window if len(window) == (2*window_radius+1) else None

# Sequence: an array of sequences
# Centers: a dataframe
def save(sequences, centers, is_top):
    # f = open(DIR + SEQUENCES_FILE, "w+")
    # f.write(
    #     ",".join(
    #         map(str,
    #             [i for i in range(-window_radius, window_radius+1)]))
    #     + "\n")
    # for s in sequences:
    #     f.write(",".join(map(str, s)) + "\n")
    # f.close()
    seq_f = TOP_SEQUENCES_FILE if is_top else BOTTOM_SEQUENCES_FILE
    cen_f = TOP_CENTERS_FILE if is_top else BOTTOM_CENTERS_FILE

    for col in sequences:
        sequences[col] = np.array(sequences[col])
    np.save(DIR + seq_f, sequences)
    centers.to_csv(DIR + cen_f, index=False)

# CHECKPOINT = 1000
IPD_THRESHOLD = 0.0

print("filtering")
peaks = DATA
# plasmid,position,strand,base,ipdRatio,J
sequences = ["position", "base", "ipdRatio"]
top_sequences = {col:[] for col in sequences}
bottom_sequences = {col:[] for col in sequences}
top_centers = pd.DataFrame() # create empty dataframe
bottom_centers = pd.DataFrame()

print("creating peaks")
print(peaks.shape[0])
for i in range(peaks.shape[0]):
    # print(i)
    row = peaks.iloc[i]
    is_top = row["strand"] == 0
    # selected = \
    #     ("IPD Top Ratio" \
    #     if row["IPD Top Ratio"] > row["IPD Bottom Ratio"] else \
    #     "IPD Bottom Ratio") if not args.bases else \
    #         "Base"
    window = get_window(row, RADIUS)
    if window is not None:
        for column in sequences:
            if column not in ["plasmid", "strand"]:
                if is_top:
                    top_sequences[column].append(window[column].tolist())
                else:
                    bottom_sequences[column].append(window[column].tolist())

        # sequences.append(window[selected].tolist())
        # print(window[selected].tolist())
        if is_top:
            top_centers = top_centers.append(row, ignore_index=True)
        else:
            bottom_centers = bottom_centers.append(row, ignore_index=True)
            
        print(f"{i/peaks.shape[0]}")
        # if len(sequences) % CHECKPOINT == 0:
        #     print(len(sequences))
        #     save(sequences, centers, RADIUS)

save(top_sequences, top_centers, True)
save(bottom_sequences, bottom_centers, False)

# negatives = DATA[(DATA["Fold Change"] < 10) & (DATA["Max IPD"] < IPD_THRESHOLD)].sample(peaks.shape[0] // 2)
# for i in range(negatives.shape[0]):
#     print(i)
#     row = negatives.iloc[i]
#     # selected = \
#     #     ("IPD Top Ratio" \
#     #     if row["IPD Top Ratio"] > row["IPD Bottom Ratio"] else \
#     #     "IPD Bottom Ratio") if not args.bases else \
#     #         "Base"
#     window = get_window(row, RADIUS)
#     if window is not None:
#         for column in DATA.columns:
#             if column not in ["Chromosome", "Position", "Max IPD"]:
#                 sequences[column].append(window[column].tolist())
#         # sequences.append(window[selected].tolist())
#         # print(window[selected].tolist())
#         centers = centers.append(row, ignore_index=True)
#         if len(sequences) % CHECKPOINT == 0:
#             print(len(sequences))
#             save(sequences, centers, RADIUS)

# save(top_sequences, top_centers)


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








