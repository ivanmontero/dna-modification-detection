# Ivan Montero

# ========== Imports ==========

# Boilerplate
import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os

# SciKit Learn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Commandline arguments.
from argparse import ArgumentParser

# ========== Command Arguments ==========

parser = ArgumentParser()

# Classfication related -- Required
parser.add_argument("method",
                    help=("The method of classification. "
                          "Must be one of the following: "
                          "knn, log_reg, svc, mlpc"))
parser.add_argument("-r", "--radii", nargs="+", required=True,
                    help="Radii to run classfier on.")
parser.add_argument("-p", "--param", nargs="*", action='append',
                    help="Parameters to the classfier.")

# Data related -- Required
parser.add_argument("-c", "--centers", required=True,
                    help="The file containing center IPD info.")
parser.add_argument("-ts", "--topsequences", required=True,
                    help="The file containing top IPD sequences.")
parser.add_argument("-bs", "--bottomsequences", required=True,
                    help="The file containing bottom IPD sequences.")
parser.add_argument("-b", "--bases", required=True,
                    help="The file containing base sequences.")
parser.add_argument("-o", "--outdir", required=True,
                    help="The directory to output.")

# Data related -- Optional
parser.add_argument("--parallel", action="store_true",
                    help="Run on a multithreaded environment.")
parser.add_argument("--interactive", action="store_true",
                    help="Makes plots show to the user.")
parser.add_argument("-top", "--top", action="store_true",
                    help="Analyze only top")
parser.add_argument("-bottom", "--bottom", action="store_true",
                    help="Analyze only bottom")
parser.add_argument("-e", "--even", action="store_true",
                    help="Use a 50-50 distribution of true and false examples.")

args = parser.parse_args()

# ========== Classification Methods ==========
def knn(X_train, y_train, X_test, params):
    return KNeighborsClassifier(int(params[0])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def log_reg(X_train, y_train, X_test, params):
    return LogisticRegression(max_iter=int(params[0])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def svc(X_train, y_train, X_test, params):
    return SVC(kernel=params[0], probability=True).fit(X_train, y_train).predict_proba(X_test)[:,1]

def mlpc(X_train, y_train, X_test, params):
    return MLPClassifier(tuple([int(float(i)*len(X_train[0])) for i in params[0].split(",")]), max_iter=int(params[1])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def rfc(X_train, y_train, X_test, params):
    return RandomForestClassifier(n_estimators=int(params[0]), n_jobs=-1).fit(X_train, y_train).predict_proba(X_test)

def keras_model(X_train, y_train, X_test, params):
    # layer sizes
    sizes = [int(float(i)*len(X_train[0])) for i in params[0].split(",")]
    model = Sequential()
    model.add(Dense(sizes[0], activation="relu", input_shape=(len(X_train[0]),)))
    model.add(Dropout(float(params[1])))
    for i in range(1, len(sizes)):
        model.add(Dense(sizes[i], activation="relu"))
        model.add(Dropout(float(params[1])))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X_train, y_train, epochs=int(params[2]))

    return np.squeeze(model.predict(X_test))

def get_classfication_name(method):
    if method == "knn":
        return "K-Nearest Neighbors"
    elif method == "log_reg":
        return "Logistic Regression"
    elif method == "svc":
        return "Support Vector"
    elif method == "mlpc":
        return "Multi-Layer Perception"
    elif method == "keras_model":
        return "Keras Model"
    else:
        return "Unknown"

# ========== Run Setup ==========

def resize_and_verify(centers, topsequences, bottomsequences, bases, radius):
    mr = len(topsequences.iloc[0]) // 2
    ts = topsequences.iloc[:,mr-radius:mr+radius+1]
    bs = bottomsequences.iloc[:,mr-radius:mr+radius+1]
    b = bases.iloc[:,mr-radius:mr+radius+1]

    to_keep = ts.isin(ts.dropna()).iloc[:,0]

    c = centers[to_keep]
    ts = scale(ts[to_keep])
    bs = scale(bs[to_keep])
    b = b[to_keep]

    return c, ts, bs, b

def prepare_input(topsequences, bottomsequences, bases):
    if args.top:
        data = topsequences
    elif args.bottom:
        data = bottomsequences
    else:
        data = np.concatenate((topsequences, bottomsequences), axis=1)
    data = np.concatenate((data, pd.get_dummies(bases).values), axis=1)
    return data

def plot_roc(fpr, tpr, auc, label, title, name):
    if args.interactive:
        import matplotlib.pyplot as plt
    else:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()

    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], label=f"{label[i]} (AUC={auc[i]:.2f})")
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(title)

    if args.interactive:
        plt.show()
    else:
        plt.savefig(outdir + name + ".png", dpi=600)
    plt.close("all")
    plt.cla()

# ========== Main Setup ==========

def init(l, c, ts, bs, b, r, p, o):
    global lock, centers, topsequences, bottomsequences, bases, radii, params, outdir
    lock = l
    centers = c
    topsequences = ts
    bottomsequences = bs
    bases = b
    radii = list(map(int,r))
    params = p
    outdir = o
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

def get_resources():
    # Load in tables
    c = pd.read_csv(args.centers)
    ts = pd.read_csv(args.topsequences)
    bs = pd.read_csv(args.bottomsequences)
    b = pd.read_csv(args.bases)

    # Filter based on params.
    if args.top:
        strand = c["IPD Top Ratio"] > c["IPD Bottom Ratio"]
    elif args.bottom:
        strand = c["IPD Top Ratio"] < c["IPD Bottom Ratio"]
    else:
        strand = c["IPD Top Ratio"] == c["IPD Top Ratio"]   #always true

    if args.even:
        # Create 50-50 distribution of fp and tp
        n_pos = c[(c["Fold Change"] > 10) & strand].shape[0]
        neg = c[(c["Fold Change"] <= 10) & strand].sample(n=n_pos, random_state=0)
        to_drop = c.isin(
            pd.concat([neg, c[(c["Fold Change"] > 10) & strand]])).iloc[:,0]
        strand = to_drop

    return c[strand], ts[strand], bs[strand], b[strand]

# ========== Run Routine ==========

def run(index):
    print(f"[START] {params[index]}")
    fprs = []
    tprs = []
    aucs = []
    labels = []

    for radius in radii:
        c, ts, bs, b = resize_and_verify(centers, topsequences, bottomsequences, bases, radius)     
        s =  prepare_input(ts, bs, b)
        y = c["Fold Change"].map(lambda x: 1 if x > 10 else 0).values
        X_train, X_test, y_train, y_test = train_test_split(s, y, test_size=0.2)

        y_scores = globals()[args.method](X_train, y_train, X_test, params[index])

        fpr, tpr, threshold = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)
        labels.append(f"r={radius}")

    title = f"{get_classfication_name(args.method)} ROC Plot"
    if params:
        title += f". Params: {', '.join(params[index])}"
    name = f"{args.method}_{'_'.join(params[index])}"

    plot_roc(fprs, tprs, aucs, labels, title, name)
    print(f"[FINISH] {params[index]}")

# ========== Main ==========

if __name__ == "__main__":
    l = Lock()
    c, ts, bs, b = get_resources()

    if args.parallel:
        pool = Pool(os.cpu_count(),
                    initializer=init,
                    initargs=(l, c, ts, bs, b, args.radii, args.param, arg.outdir))
        pool.map(run, params)
        pool.close()
        pool.join()
    else:   
        init(l, c, ts, bs, b, args.radii, args.param, args.outdir)
        for i in range(len(args.param)):
            run(i)