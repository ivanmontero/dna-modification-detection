import pandas as pd
from sklearn.cluster import KMeans
from multiprocessing import Pool, Lock
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

def dim_red(sequences, red_type):
    reduced = []
    if red_type == "pca":
        pca = PCA()
        reduced = pca.fit_transform(sequences)
    
    if red_type == "tsne":
        tsne = TSNE()
        reduced = tsne.fit_transform(sequences)
    return reduced

# reduced, is_peak, radius, red_type
def save(dim_red, is_peak, radius, red_type):
    print("s r%d %s" % (radius, red_type))

    lock.acquire()
    colors = is_peak.map(lambda x: 'r' if x else 'b')

    plt.scatter(dim_red[:,0], dim_red[:,1], c=colors, s=1)

    # Save the plot
    name = "r%d_%s" % (radius, red_type)
    plt.savefig(outdir + name + ".png", dpi=600)
    plt.cla()
    plt.close("all")
    lock.release()

def run(radius=100, red_type="pca"):
    print("run " + str(radius) + " " + red_type)

    # Max radius in the dataset
    mr = len(sequences.iloc[0]) // 2

    r_seq = sequences.iloc[:,mr-radius:mr+radius+1]

    reduced = dim_red(r_seq, red_type)
    is_peak = centers["Fold Change"].map(lambda x: x > 10)

    save(reduced, is_peak, radius, red_type)

def wrapper(args):
    run(*args)

def init(l, s, c, o):
    global lock, sequences, centers, outdir, outdir
    lock = l
    sequences = s
    centers = c
    outdir = o

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

    # Create 50-50 distribution of fp and tp
    # n_pos = c[c["Fold Change"] > 10].shape[0]
    # if args.rand:
    #     neg = c[c["Fold Change"] <= 10].sample(n=n_pos, random_state=0)
    # else:
    #     neg = c[c["Fold Change"] <= 10].nlargest(n_pos, "Max IPD")
    # to_drop = c.isin(pd.concat([neg, c[c["Fold Change"] > 10]])).iloc[:,0]
    # c = c[to_drop]
    # s = s[to_drop]
    # print(n_pos)


    # Prepare outputs.
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Prepare job parameters.
    params = []
    for r in [100, 50, 25, 20, 15, 10, 5]:
        # Different radii
        for t in ["pca", "tsne"]:
            # Different cluster sizes
            params.append((r,t))

    # Create pool
    pool = Pool(os.cpu_count(), initializer=init, initargs=(l, s, c, o))

    # Run
    pool.map(wrapper, params)
    pool.close()
    pool.join()