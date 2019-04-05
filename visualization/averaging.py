
import pandas as pd
import numpy as np 
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-c", "--centers", help="The file containing center files.")
parser.add_argument("-s", "--sequences", help="The file containing sequences")
parser.add_argument("-o", "--outdir", help="The directory to hold output.")
args = parser.parse_args()

sequences = pd.read_csv(args.sequences)
centers = pd.read_csv(args.centers)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

tp_top = centers[(centers["Fold Change"] > 10) &
                 (centers["IPD Top Ratio"] > centers["IPD Bottom Ratio"])]

tp_bottom = centers[(centers["Fold Change"] > 10) &
                    (centers["IPD Top Ratio"] < centers["IPD Bottom Ratio"])]

fp_top = centers[(centers["Fold Change"] <= 10) &
                 (centers["IPD Top Ratio"] > centers["IPD Bottom Ratio"])]

fp_bottom = centers[(centers["Fold Change"] <= 10) &
                    (centers["IPD Top Ratio"] < centers["IPD Bottom Ratio"])]

def plot(c, title, name, r=100):
    to_drop = centers.isin(c).iloc[:,0]
    s = sequences[to_drop]
    avg = s.sum(axis=0) / s.shape[0]
    # avg.plot()
    avg = avg.iloc[100-r:100+r+1]
    # print(avg)
    plt.plot([i-r for i in range(2*r+1)], avg)
    plt.title(title)
    # plt.show()
    plt.savefig(args.outdir + name + ".png", dpi=400)
    plt.cla()
    plt.close("all")

# c1 - c2
def plot_diff(c1, c2, title, name, r=100):
    to_drop = centers.isin(c1).iloc[:,0]
    s1 = sequences[to_drop]
    avg1 = s1.sum(axis=0) / s1.shape[0]

    to_drop = centers.isin(c2).iloc[:,0]
    s2 = sequences[to_drop]
    avg2 = s2.sum(axis=0) / s2.shape[0]

    avg = avg1 - avg2
    # print(avg)

    # avg.dropna()

    # print(avg)

    # avg.plot()
    avg = avg.iloc[100-r:100+r+1]
    plt.plot([i-r for i in range(2*r+1)], avg)
    plt.title(title)
    # plt.show()
    plt.savefig(args.outdir + name + ".png", dpi=400)
    plt.cla()
    plt.close("all")

plot(tp_top, "True Positive Top Ratio", "tp_top")
plot(tp_bottom, "True Positive Bottom Ratio", "tp_bottom")
plot(fp_top, "False Positive Top Ratio", "fp_top")
plot(fp_bottom, "False Positive Bottom Ratio", "fp_bottom")

plot_diff(tp_top, fp_top, "TP - FP, Top Ratio", "tp_minus_fp_top")
plot_diff(tp_bottom, fp_bottom, "TP - FP, Bottom Ratio", "tp_minus_fp_bottom")

plot(tp_top, "True Positive Top Ratio", "tp_top_r15", r=15)
plot(tp_bottom, "True Positive Bottom Ratio", "tp_bottom_r15", r=15)
plot(fp_top, "False Positive Top Ratio", "fp_top_r15", r=15)
plot(fp_bottom, "False Positive Bottom Ratio", "fp_bottom_r15", r=15)

plot_diff(tp_top, fp_top, "TP - FP, Top Ratio", "tp_minus_fp_top_r15", r=15)
plot_diff(tp_bottom, fp_bottom, "TP - FP, Bottom Ratio", "tp_minus_fp_bottom_r15", r=15)

# # Create 50-50 distribution of fp and tp
# n_pos = c[c["Fold Change"] > 10].shape[0]
# if args.rand:
#     neg = c[c["Fold Change"] <= 10].sample(n=n_pos, random_state=0)
# else:
#     neg = c[c["Fold Change"] <= 10].nlargest(n_pos, "Max IPD")
# to_drop = c.isin(pd.concat([neg, c[c["Fold Change"] > 10]])).iloc[:,0]
# c = c[to_drop]
# s = s[to_drop]
# print(n_pos)

# avg.plot()
# plt.show()
# print(avg)
# plt.plot(avg.values)