import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Train a model on the features and save it.')

    parser.version = 0.1

    parser.add_argument(
        '-of',
        '--original_file',
        required = True,
        help = 'Input preprocessed h5 original data file.'
    )

    parser.add_argument(
        '-pf',
        '--predictions_file',
        required = True,
        help = 'Input preprocessed h5 original data file.'
    )

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    return parser.parse_args()

# Loads the extracted features file, and relevant data.
def load(original_file, predictions_file):
    original = pd.read_hdf(original_file)
    predictions = pd.read_csv(predictions_file, names=["chromosome", "position", "drop"], index_col=[0,1])
    data = original.merge(predictions, on= ["chromosome", "position"])
    return data

def get_peaks(data, threshold, min_peak_length=50): # Add another threshold for peak "length"
    data["peak_id"] = 0
    peak_id = 1
    in_peak = False
    for index, row in data.iterrows():
        if row["fold_change"] > threshold:
            if not in_peak:
                in_peak = True
            data.loc[index, "peak_id"] = peak_id
        elif in_peak:
            in_peak = False
            if (data["peak_id"] == peak_id).sum() < min_peak_length:
                data[data["peak_id"] == peak_id] = 0
            else:
                peak_id += 1
    return peak_id if in_peak else peak_id - 1
    # print(peak_id)

if __name__ == "__main__":
    args = setup()
    data = load(args.original_file, args.predictions_file)
    # print(data)
    total_peaks = get_peaks(data, 10)

    # for p in range(1, total_peaks+1):
    #     print((data["peak_id"] == p).sum())
    # # print("here")
    threshold = .2
    pj = []
    jp = []
    for threshold in np.linspace(0, data["drop"].max(), 100):
        # print(threshold)
        js = (data["drop"] >= threshold).sum()
        js_in_peak = ((data["drop"] >= threshold) & (data["peak_id"] != 0)).sum()
        peaks_w_j = 0
        for p in range(1, total_peaks+1):
            peaks_w_j += float(((data["peak_id"] == p) & (data["drop"] >= threshold)).sum() > 0)
        pj.append(peaks_w_j / total_peaks)
        jp.append(js_in_peak / js)
    plt.plot(pj, jp)
    plt.xlabel("peaks w/ J")
    plt.ylabel("J inside peaks")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()

    # print(f"% of peaks w/ J: {peaks_w_j / total_peaks}")
    # print(f"% of J inside peaks: {js_in_peak / js}")




    # print(data[data["peak_id"] == 1])
    # print(data[data["fold_change"] > 10])
