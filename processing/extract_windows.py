import pandas as pd
import numpy as np 
import argparse
import time

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Create a TSV file with a set of peaks.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.version = 0.1

    parser.add_argument(
        '-i', 
        '--infile', 
        required = True,
        help = 'Input file.')

    parser.add_argument(
        '-w', 
        '--window', 
        default = 50,
        help = 'The size of the window used for predictions.')

    parser.add_argument(
        '-c',
        '--columns',
        default = ['top_A',
                   'top_T',
                   'top_C',
                   'top_G', 
                   'top_ipd', 
                   'bottom_ipd'],
        help = 'List of columns to include as features.')

    parser.add_argument(
        '-p',
        '--ipd',
        default = 2,
        help = 'IPD threshold value.')

    parser.add_argument(
        '-f',
        '--fold-change',
        default = 10,
        help = 'Fold change threshold value.')

    parser.add_argument(
        '-e',
        '--examples',
        default = 50000,
        help = 'Max number of examples from each class.')

    parser.add_argument(
        '-o', 
        '--outfile', 
        default = 'outfile.tsv',
        help = 'Output file.')
    
    return parser.parse_args()

def sample(data, examples):
    if len(data) <= examples:
        return data.index.values
    else:
        return data.sample(examples).index.values

def windows(index, data, window, columns):
    radius = int(window/2)
    features = []

    for i in range(len(index)):
        chromosome = index[i][0]
        position = index[i][1]

        start = time.time()

        vector = {}
        for column in columns:
            vector[column] = []

        for i in range(lower_bound, upper_bound):
            selection =  data.loc[chromosome, position]
            for column in columns:
                vector[column].append(selection[column])

        concatenation = []
        for column in columns:
            concatenation += vector[column]

        features.append(concatenation) 

        print (time.time() - start)

    return features

def main():
    start = time.time()
    arguments = setup()

    data = pd.read_csv('/active/myler_p/People/Sur/J-IP/LtaP/ivan-pacbio/merged_data.csv', 
        usecols = list(arguments.columns) + ['chromosome','position','fold_change'], 
        index_col = ['chromosome','position'])

    data = data.dropna()

    ipd = arguments.ipd
    fold_change = arguments.fold_change

    true_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] > fold_change)]
    false_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] < fold_change)]
    true_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] < fold_change)]
    false_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] > fold_change)]

    true_positive = sample(true_positive, arguments.examples)
    false_positive = sample(false_positive, arguments.examples)
    true_negative = sample(true_negative, arguments.examples)
    false_negative = sample(false_negative, arguments.examples)

    true_positive_features = windows(true_positive, data, arguments.window, arguments.columns)

    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

if __name__ == '__main__':
    main()







# # Create the pandas file
# DATA = pd.read_table(FILE).dropna()
# DATA_BY_CHROMOSOME = {c:DATA[DATA["Chromosome"] == c] for c in DATA["Chromosome"].unique()}
# def get_window(center, window_radius):
#     df = DATA_BY_CHROMOSOME[center["Chromosome"]]
#     window = df[(df["Position"] >= center["Position"] - window_radius)
#               & (df["Position"] <= center["Position"] + window_radius)]
#     return window if len(window) == (2*window_radius+1) else None

# # Sequence: an array of sequences
# # Centers: a dataframe
# def save(sequences, centers):
#     for col in sequences:
#         sequences[col] = np.array(sequences[col])
#     np.save(DIR + SEQUENCES_FILE, sequences)
#     if not args.no_centers:
#         centers.to_csv(DIR + CENTERS_FILE, index=False)

# print("apply")
# DATA["Max IPD"] = pd.concat(
#     [DATA["Top IPD Ratio"], DATA["Bottom IPD Ratio"]], axis=1).max(axis=1)

# CHECKPOINT = 1000
# IPD_THRESHOLD = 5

# print("filtering")
# peaks = DATA[DATA["Max IPD"] > IPD_THRESHOLD].sort_values(by=["Max IPD"], ascending=False)

# sequences = {col:[] for col in DATA.columns}
# centers = pd.DataFrame() # create empty dataframe

# print("creating peaks")
# print(peaks.shape[0])
# for i in range(peaks.shape[0]):
#     print(i)
#     row = peaks.iloc[i]
#     window = get_window(row, RADIUS)
#     if window is not None:
#         for column in DATA.columns:
#             if column not in ["Chromosome", "Max IPD"]:
#                 sequences[column].append(window[column].tolist())
#         # sequences.append(window[selected].tolist())
#         # print(window[selected].tolist())
#         centers = centers.append(row, ignore_index=True)
#         if len(sequences) % CHECKPOINT == 0:
#             print(len(sequences))
#             save(sequences, centers, RADIUS)

# negatives = DATA[(DATA["Fold Change"] < 10) & (DATA["Max IPD"] < IPD_THRESHOLD)].sample(peaks.shape[0] // 2)
# for i in range(negatives.shape[0]):
#     print(i)
#     row = negatives.iloc[i]

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

# save(sequences, centers)

