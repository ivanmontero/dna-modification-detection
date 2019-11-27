# Boilerplate
import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os
import math

# SciKit Learn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from itertools import islice

# Keras
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, Dropout

# Create training data
c = pd.read_csv("../../data/processed/centers_r_25_n.csv")
s = np.load("../../data/processed/sequences_r_25_n.npy")[()]

print(s)
ss = []
ss.append((s["Top IPD Ratio"] - np.mean(s["Top IPD Ratio"])) /  np.std(s["Top IPD Ratio"]))
ss.append((s["Bottom IPD Ratio"] - np.mean(s["Bottom IPD Ratio"])) /  np.std(s["Bottom IPD Ratio"]))
# print(pd.get_dummies(pd.DataFrame(s["Base"])))
onehot = pd.get_dummies(pd.DataFrame(s["Base"]))
print(onehot)
for i in range(51):
    col = f"{i}_N"
    if col in onehot.columns:
        onehot.drop(col, axis=1, inplace=True)
ss.append(onehot.values)
X_train = np.concatenate(ss, axis=1)
print(X_train.shape)
y_train = c["Fold Change"].map(lambda x: 1 if x > 10 else 0).values

# Create model
sizes = [int(float(i)*len(X_train[0])) for i in [1.0, 0.5, 0.25]]
do = .6
model = Sequential()
model.add(Dense(sizes[0], activation="relu", input_shape=(len(X_train[0]),)))
model.add(Dropout(do))
for i in range(1, len(sizes)):
    model.add(Dense(sizes[i], activation="relu"))
    model.add(Dropout(do))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=5) #CHANGE TO 25

# Load plasmid data
p = pd.read_csv("../../data/interim/plasmid.tsv", sep="\t")
# p = p[p["strand"] == 0].reset_index(drop=True)

def create_window(seq, radius):
    windows = []
    for i in range(radius, len(seq) - radius):
        windows.append(seq[i-radius:i+radius+1])
    # return np.array(windows)
    temp = np.array(windows)
    print(temp.shape)
    return temp
def create_sequence(bases, ipd):
    # print(scale(sequences["Top IPD Ratio"][index]))
    # print(pd.get_dummies(pd.DataFrame(bases)).values.flatten())
    return np.concatenate([ipd,
        pd.get_dummies(pd.DataFrame(bases)).values.flatten()])

# We want to see which T, when removed, drops the prediction value considerably
def determine_t(bases, ipd, model):
    # Returns the offset of the T in the sequence.

    # List of T positions and their values.
    t_pos = []
    c_val = []
    sequence = bases
    # print(sequence)
    for i, c in enumerate(sequence):
        if c == "T":
            t_pos.append(i - len(sequence) // 2)
            te = []
            for n in ["A", "C", "G"]:
                # print(sequence)
                # print(create_sequence(np.concatenate([sequence[:i], [n], sequence[i+1:]]), index))
                te.append(create_sequence(np.concatenate([sequence[:i], [n], sequence[i+1:]]), ipd))
                # print(te[-1].shape[0])
                if te[-1].shape[0] != 306:
                    del te[-1]
                    continue
                # print(te[-1].shape)
            # print(np.array(te))
            if len(te) != 3:
                continue
            try:
                vals = model.predict(np.array(te))
            except:
                print("ERROR")
                print(te)
                print(te[0].shape)
                print(te[1].shape)
                print(te[2].shape)
                continue
            # print(vals)
            c_val.append(min(vals))
    if len(c_val) == 0:
        return math.inf, math.inf
    m = np.argmin(c_val)
    return t_pos[m], c_val[m]

THRESHOLD = 0.5
radius = 25
p["J_pred"] = 0.0
p["J_delta"] = 0.0
p["J_detections"] = 0
for c in p["refName"].unique():
    # print(f"===== {c} =====")
    c_vals = p[p["refName"] == c]
    # print(c_vals.shape)
    # print(c_vals)
    # print(c_vals)
    ipd_windows_top = create_window(c_vals[c_vals["strand"] == 0]["ipdRatio"], radius)
    ipd_windows_bottom = create_window(c_vals[c_vals["strand"] == 1]["ipdRatio"], radius)
    ipd_windows_top = (ipd_windows_top - np.average(ipd_windows_top))/np.std(ipd_windows_top)
    ipd_windows_bottom = (ipd_windows_bottom - np.average(ipd_windows_bottom))/np.std(ipd_windows_bottom)
    base_windows = create_window(c_vals[c_vals["strand"] == 0]["base"], radius)
    base_onehot = pd.get_dummies(pd.DataFrame(base_windows)).values
    # print(base_onehot.shape)
    X_test = np.concatenate([ipd_windows_top, ipd_windows_bottom, base_onehot], axis=1)
    # print(X_test.shape)
    y_pred = model.predict(X_test)
    # print(c_vals.iloc[radius:c_vals.shape[0] - radius, :].values.shape)
    # print(y_pred.shape)
    for i, pred_row in enumerate(zip(y_pred, c_vals[c_vals["strand"] == 0].iloc[radius:c_vals.shape[0] - radius, :].iterrows())):
        # print(pred_row)
        # print(pred_row[1][0])
        p.at[pred_row[1][0],"J_pred"] = pred_row[0]
        # print(pred_row[1][0])
        if pred_row[0] > THRESHOLD:
            # pass
            offset, min_t = determine_t(base_windows[i], np.concatenate([ipd_windows_top[i], ipd_windows_bottom[i]]), model)
            # print(min_t - pred_row[0])
            if offset == math.inf or min_t == math.inf:
                continue
            J_pos = pred_row[1][0] + offset*2
            # print(J_pos)
            p.at[J_pos,"J_detections"] += 1
            # p.at[J_pos,"J_delta"] = min(p.at[J_pos, "J_delta"], min_t - pred_row[0])
            p.at[J_pos,"J_delta"] += min_t - pred_row[0]
        if i % (c_vals[c_vals["strand"] == 0].shape[0] // 100) == 0:
            print(i / c_vals[c_vals["strand"] == 0].shape[0])
    # print(c_vals.head(100))
    # print(p.head(100))

p.to_csv("../../data/processed/plasmid_w_predictions_dual.tsv", sep="\t", index=False)
