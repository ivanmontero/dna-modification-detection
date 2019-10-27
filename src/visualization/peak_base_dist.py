
import pandas as pd
import numpy as np 
from operator import add, truediv, sub
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-c", "--centers", help="The file containing center files.")
parser.add_argument("-b", "--bases", help="The file containing bases")
parser.add_argument("-o", "--outdir", help="The directory to hold output.")
args = parser.parse_args()

centers = pd.read_csv(args.centers)
bases = pd.read_csv(args.bases)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
print(bases)

bases_flatten = list(bases.values.flatten())
total_bases = len(bases_flatten)
avg_base = {
    "A" : bases_flatten.count("A") / total_bases,
    "T" : bases_flatten.count("T") / total_bases,
    "C" : bases_flatten.count("C") / total_bases,
    "G" : bases_flatten.count("G") / total_bases
}

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
    s = bases[to_drop]
    # avg = s.sum(axis=0) / s.shape[0]
    bs = {
        "T" : [],
        "A" : [],
        "C" : [],
        "G" : []
    }
    s = s.iloc[:,100-r:100+r+1]
    print(s)
    for col in s:
        # c = s[col].
        # bs[b].append(c[c["Base"] == b].size)
        counts = s[col].value_counts()
        for b in ["T", "A", "C", "G"]:
            bs[b].append(counts[b])
    
    domain = [i-r for i in range(2*r+1)]
    total = s.shape[0]
    for b in bs:
        bs[b] = np.array(bs[b])
        plt.plot(domain, bs[b]/total - avg_base[b], label=b, linewidth=3.0)

    
    # tp = list(map(truediv, bs["T"], [total] * len(bs["T"])))
    # tavg = sum(tp) / len(tp)
    # plt.plot(domain, list(map(sub, tp, [tavg] * len(bs["T"]))), color="r", label="T", linewidth=3.0)
    
    # ap = list(map(truediv, bs["A"], [total] * len(bs["A"])))
    # aavg = sum(ap) / len(ap)
    # plt.plot(domain, list(map(sub, ap, [aavg] * len(bs["A"]))), color="y", label="A", linewidth=3.0)
    
    # cp = list(map(truediv, bs["C"], [total] * len(bs["C"])))
    # cavg = sum(cp) / len(cp)
    # plt.plot(domain, list(map(sub, cp, [cavg] * len(bs["C"]))), color="b", label="C", linewidth=3.0)
    
    # gp = list(map(truediv, bs["G"], [total] * len(bs["G"])))
    # gavg = sum(gp) / len(gp)
    # plt.plot(domain, list(map(sub, gp, [gavg] * len(bs["G"]))), color="g", label="G", linewidth=3.0)
    
    plt.legend()
    plt.title(title)
    plt.savefig(args.outdir + name + ".png", dpi=800)
    plt.cla()
    plt.close("all")

plot(tp_top, "True Positive Top Ratio", "tp_top")
plot(tp_bottom, "True Positive Bottom Ratio", "tp_bottom")
plot(fp_top, "False Positive Top Ratio", "fp_top")
plot(fp_bottom, "False Positive Bottom Ratio", "fp_bottom")

plot(tp_top, "True Positive Top Ratio", "tp_top_r20", r=20)
plot(tp_bottom, "True Positive Bottom Ratio", "tp_bottom_r20", r=20)
plot(fp_top, "False Positive Top Ratio", "fp_top_r20", r=20)
plot(fp_bottom, "False Positive Bottom Ratio", "fp_bottom_r20", r=20)