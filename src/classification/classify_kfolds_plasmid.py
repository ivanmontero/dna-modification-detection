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

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# XGBoost
import xgboost as xgb

# Commandline arguments.
from argparse import ArgumentParser

# ========== Command Arguments ==========

parser = ArgumentParser()

# Classfication related -- Required
parser.add_argument("method",
                    help=("The method of classification. "
                          "Must be one of the following: "
                          "knn, log_reg, svc, mlpc"),
                    default="keras_model")
parser.add_argument("-r", "--radius",
                    help="Radius to run classfier on.",
                    default="25")
parser.add_argument("-p", "--param", nargs="*", action='append',
                    help="Parameters to the classfier.",
                    default="1.0,0.5,0.25 0.6 5")

# Data related -- Required
parser.add_argument("-c", "--centers",
                    help="The file containing center IPD info.",
                    default="../../data/processed/plasmid_top_centers.csv")
parser.add_argument("-s", "--sequences",
                    help="The file containing sequences.",
                    default="../../data/processed/plasmid_top_sequences.npy")
parser.add_argument("-o", "--outdir",
                    help="The directory to output.",
                    default="../../graphs/classification/classify_kfolds_plasmid/")

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
parser.add_argument("-folds", "--folds", default=5, type=int,
                    help="Number of folds to use for cross validation.")
parser.add_argument("-n", "--note", default="",
                    help="Post-fix to file name.")
args = parser.parse_args()

# ========== Classification Methods ==========
def knn(X_train, y_train, X_test, params):
    return KNeighborsClassifier(int(params[0])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def log_reg(X_train, y_train, X_test, params):
    return LogisticRegression(max_iter=int(params[0])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def ridge(X_train, y_train, X_test, params):
    return RidgeClassifier(alpha=float(params[0])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def svc(X_train, y_train, X_test, params):
    return SVC(kernel=params[0], probability=True).fit(X_train, y_train).predict_proba(X_test)[:,1]

# Input relative
def mlpc(X_train, y_train, X_test, params):
    return MLPClassifier(tuple([int(float(i)*len(X_train[0])) for i in params[0].split(",")]), max_iter=int(params[1])).fit(X_train, y_train).predict_proba(X_test)[:,1]

def rfc(X_train, y_train, X_test, params):
    return RandomForestClassifier(n_estimators=int(params[0]), n_jobs=-1).fit(X_train, y_train).predict_proba(X_test)[:,1]

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

def boosting(X_train, y_train, X_test, params):
    # dtrain = xgb.DMatrix(np.array(X_train), label=np.array(y_train))
    # bparams = {
    #     "verbosity": 2,
    #     # "num_parallel_tree": 2
    #     "max_depth": int(params[1])
    # }
    # bst = xgb.train(bparams, dtrain, int(params[0]))
    # dtest = xgb.DMatrix(np.array(X_test))
    # return np.squeeze(bst.predict(dtest, ntree_limit=bst.best_ntree_limit))
    return xgb.XGBClassifier(max_depth=10, n_estimators=int(params[0]), n_jobs=-1,).fit(X_train, y_train).predict_proba(X_test)[:,1]

def get_classfication_name(method):
    if method == "knn":
        return "K-Nearest Neighbors"
    elif method == "log_reg":
        return "Logistic Regression"
    elif method == "svc":
        return "Support Vector"
    elif method == "mlpc":
        return "Multi-Layer Perception"
    elif method == "rfc":
        return "Random Forest"
    elif method == "keras_model":
        return "Keras Model"
    elif method == "boosting":
        return "XGBoost Random Forest"
    elif method == "ridge":
        return "Ridge Classifier"
    else:
        return "Unknown"

# ========== Run Setup ==========

# def resize_and_verify(centers, sequences):
#     mr = len(topsequences.iloc[0]) // 2
#     ts = topsequences.iloc[:,mr-radius:mr+radius+1]
#     bs = bottomsequences.iloc[:,mr-radius:mr+radius+1]
#     b = bases.iloc[:,mr-radius:mr+radius+1]

#     to_keep = ts.isin(ts.dropna()).iloc[:,0]

#     c = centers[to_keep]
#     ts = scale(ts[to_keep])
#     bs = scale(bs[to_keep])
#     b = b[to_keep]

#     return c, ts, bs, b

# Definitely not coverage or score
FEATURES = [
            # "Bottom Coverage",
            # "Bottom IPD Ratio",
            # "Bottom Null Prediction",
            # "Bottom Score",
            # "Bottom Trimmed Error",
            # "Bottom Trimmed Mean",
            # "Top Coverage",
            "ipdRatio",
            # "Top Null Prediction",
            # "Top Score",
            # "Top Trimmed Error",
            # "Top Trimmed Mean",
            ]

def prepare_input(sequences):
    ss = []
    for s in sequences:
        if s in FEATURES:
            f_seq = sequences[s]
            ss.append((f_seq - np.mean(f_seq))/np.std(f_seq))
    ss.append(pd.get_dummies(pd.DataFrame(sequences["base"])).values)
    data = np.concatenate(ss, axis=1)
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
        plt.savefig(outdir + name + args.note + ".png", dpi=600)
    plt.close("all")
    plt.cla()

# ========== Main Setup ==========

def init(l, c, s, p, o):
    global lock, centers, sequences, params, outdir
    lock = l
    centers = c
    sequences = s
    params = p
    outdir = o
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

def get_resources():
    # Load in tables
    c = pd.read_csv(args.centers)
    # ts = pd.read_csv(args.topsequences)
    # bs = pd.read_csv(args.bottomsequences)
    # b = pd.read_csv(args.bases)
    s = np.load(args.sequences)[()]

    # # Filter based on params.
    # if args.top:
    #     strand = c["IPD Top Ratio"] > c["IPD Bottom Ratio"]
    # elif args.bottom:
    #     strand = c["IPD Top Ratio"] < c["IPD Bottom Ratio"]
    # else:
    #     strand = c["IPD Top Ratio"] == c["IPD Top Ratio"]   #always true

    # if args.even:
    #     # Create 50-50 distribution of fp and tp
    #     n_pos = c[(c["Fold Change"] > 10) & strand].shape[0]
    #     neg = c[(c["Fold Change"] <= 10) & strand].sample(n=n_pos, random_state=0)
    #     to_drop = c.isin(
    #         pd.concat([neg, c[(c["Fold Change"] > 10) & strand]])).iloc[:,0]
    #     strand = to_drop

    return c, s

# ========== Run Routine ==========

def run(index):
    print(f"[START] {params[index]}")

    radius = args.radius
    # c, f = resize_and_verify(centers, topsequences, bottomsequences, bases, radius)     
    s = prepare_input(sequences)
    y = c["J"].values

    fprs = []
    tprs = []
    aucs = []
    labels = []

    cv = StratifiedKFold(n_splits=args.folds, random_state=0)

    i = 0
    mean_fpr = np.linspace(0, 1, 100)
    for train, test in cv.split(s, y):
        y_scores = globals()[args.method](s[train], y[train], s[test], params[index])
        fpr, tpr, thresholds = roc_curve(y[test], y_scores)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"ROC fold {i} (AUC={roc_auc:.2f})")
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label='$\pm$ 1 std. dev.')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title("Classification ROC Plot: 50% Dropout, 25 Epochs")
    plt.title(f"{get_classfication_name(args.method)} ROC Plot. Radius={radius}, Params: {' '.join(params[index])}")
    plt.legend(loc="lower right")

    if args.interactive:
        plt.show()
    else:
        plt.savefig(outdir + f"{args.method}_kfolds_r_{radius}_p_{'_'.join(params[index]).replace(',','-').replace('.','-')}{args.note}", dpi=600)
    plt.close("all")
    plt.cla()

    print(f"[FINISH] {params[index]}")

# ========== Main ==========

if __name__ == "__main__":
    l = Lock()
    c, s = get_resources()

    if args.parallel:
        pool = Pool(os.cpu_count(),
                    initializer=init,
                    initargs=(l, c, s, args.param, arg.outdir))
        pool.map(run, params)
        pool.close()
        pool.join()
    else:   
        init(l, c, s, args.param, args.outdir)
        for i in range(len(args.param)):
            run(i)