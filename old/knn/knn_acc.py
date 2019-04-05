import pandas as pd
# from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Pool, Lock
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
parser.add_argument("-top", "--top", help="Analyze only top", action="store_true")
parser.add_argument("-bottom", "--bottom", help="Analyze only bottom", action="store_true")
args = parser.parse_args()
# finish
def knn(sequences, labels, k):
    print("knn %d" % (k,))
    # return KMeans(n_clusters=n_clusters, random_state=0).fit_predict(sequences)
    return KNeighborsClassifier(n_neighbors=k).fit(sequences, labels).predict(sequences)

def save(classifications, is_peak, radius, k):
    print("s r%d k%d" % (radius, k))

    lock.acquire()
    f = open(outfile, "a")
    f.write("%d,%d,%f\n"
            % (radius, k, accuracy_score(classifications, is_peak)))
    f.close()
    lock.release()

def run(radius=100, k=2):
    print("run " + str(k) + " " + str(radius))

    # Max radius in the dataset
    mr = len(sequences.iloc[0]) // 2

    r_seq = sequences.iloc[:,mr-radius:mr+radius+1]

    is_peak = centers["Fold Change"].map(lambda x: 1 if x > 10 else 0).values
    classifications = knn(r_seq, is_peak, k)

    save(classifications, is_peak, radius, k)

def wrapper(args):
    run(*args)

def init(l, s, c, o):
    global lock, sequences, centers, outdir, outfile
    lock = l
    sequences = s
    centers = c
    outfile = o

# ========== Main ==========
if __name__ == "__main__":
    # Prepare necessary resources.
    s = pd.read_csv(args.sequences)
    c = pd.read_csv(args.centers)
    l = Lock()
    o = args.outdir
    o += "knn_acc"
    if args.top:
        o += "_top"
        strand = c["IPD Top Ratio"] > c["IPD Bottom Ratio"]
    elif args.bottom:
        o += "_bottom"
        strand = c["IPD Top Ratio"] < c["IPD Bottom Ratio"]
    else:
        strand = c["IPD Top Ratio"] == c["IPD Top Ratio"] #always true
    o += ".csv"

    # Create 50-50 distribution of fp and tp
    n_pos = c[(c["Fold Change"] > 10) & strand].shape[0]
    neg = c[(c["Fold Change"] <= 10) & strand].sample(n=n_pos, random_state=0)
    to_drop = c.isin(pd.concat([neg, c[(c["Fold Change"] > 10) & strand]])).iloc[:,0]
    c = c[to_drop]
    s = s[to_drop]
    print(n_pos)

    # Prepare outputs.
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    f = open(o, "w+")
    f.write("%s,%s,%s\n" \
            % ("radius", "k", "acc"))
    f.close()

    # Prepare job parameters.
    params = []
    for r in [100, 50, 25, 20, 15, 10, 5]:
        # Different radii
        for k in [i for i in range(2, 13)]:
            # Different cluster sizes
            params.append((r,k))

    # Create pool
    pool = Pool(os.cpu_count(), initializer=init, initargs=(l, s, c, o))

    # Run
    pool.map(wrapper, params)
    pool.close()
    pool.join()

    # Sort
    print("sorting")
    result = pd.read_csv(o)
    result = result.sort_values(by=["radius","k"])
    result.to_csv(o, index=False)