import pandas as pd
from sklearn.cluster import KMeans
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
parser.add_argument("-t", "--trunc", help="Whether to truncate peak.", action="store_true")
parser.add_argument("-r", "--rand", help="Whether to randomize negatives.", action="store_true")
args = parser.parse_args()

def k_means(sequences, n_clusters):
    print("k_means %d" % (n_clusters,))
    return KMeans(n_clusters=n_clusters, random_state=0).fit_predict(sequences)

def save(clusters, is_peak, radius, n_clusters):
    print("s r%d c%d" % (radius, n_clusters))

    c = {i:{j: 0 for j in ["total", "tp", "fp"]} for i in range(n_clusters)}

    for i in range(len(clusters)):
        c[clusters[i]]["total"] += 1
        c[clusters[i]]["tp" if is_peak.iloc[i] else "fp"] += 1

    lock.acquire()
    f = open(outfile, "a")
    for cluster in c:
        c_d = c[cluster]
        f.write("%d,%d,%d,%d,%d,%d\n"
                % (radius, n_clusters, cluster, c_d["total"], c_d["tp"], c_d["fp"]))
    f.close()
    lock.release()

def run(radius=100, n_clusters=2):
    print("run " + str(n_clusters) + " " + str(radius))

    # Max radius in the dataset
    mr = len(sequences.iloc[0]) // 2

    r_seq = sequences.iloc[:,mr-radius:mr+radius+1]

    clusterids = k_means(r_seq, n_clusters)
    is_peak = centers["Fold Change"].map(lambda x: x > 10)

    save(clusterids, is_peak, radius, n_clusters)

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
    s = pd.read_csv(args.sequences, header=None)
    if args.trunc:
        mr = len(s.iloc[0]) // 2
        s.iloc[:, mr].values[:] = 0
    c = pd.read_csv(args.centers)
    l = Lock()
    o = args.outdir
    o += "kmeans_trunc" if args.trunc else "kmeans"
    if args.rand:
        o += "_r"
    o += ".csv"

    # Create 50-50 distribution of fp and tp
    n_pos = c[c["Fold Change"] > 10].shape[0]
    if args.rand:
        neg = c[c["Fold Change"] <= 10].sample(n=n_pos, random_state=0)
    else:
        neg = c[c["Fold Change"] <= 10].nlargest(n_pos, "Max IPD")
    to_drop = c.isin(pd.concat([neg, c[c["Fold Change"] > 10]])).iloc[:,0]
    c = c[to_drop]
    s = s[to_drop]
    print(n_pos)

    # Prepare outputs.
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    f = open(o, "w+")
    f.write("%s,%s,%s,%s,%s,%s\n" \
            % ("radius", "n_clusters", "cluster_id", "total", "tp", "fp"))
    f.close()

    # Prepare job parameters.
    params = []
    for r in [100, 50, 25, 20, 15, 10, 5]:
        # Different radii
        for n_c in [i for i in range(2, 13)]:
            # Different cluster sizes
            params.append((r,n_c))

    # Create pool
    pool = Pool(os.cpu_count(), initializer=init, initargs=(l, s, c, o))

    # Run
    pool.map(wrapper, params)
    pool.close()
    pool.join()

    # Sort
    print("sorting")
    result = pd.read_csv(o)
    result = result.sort_values(by=["radius", "n_clusters","cluster_id"])
    result.to_csv(o, index=False)