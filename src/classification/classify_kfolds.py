import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from keras.models import Sequential
from keras.layers import Dense, Dropout
import xgboost as xgb
from argparse import ArgumentParser
from pickle import dump, load

# ========== Command Arguments ==========

parser = ArgumentParser()

# Classfication related -- Required
parser.add_argument("method",
                    help=("The method of classification. "
                          "Must be one of the following: "
                          "knn, log_reg, svc, mlpc"),
                    default="keras_model")
parser.add_argument("-p", "--params", nargs="*",
                    help="Parameters to the classfier.",
                    default=["1.0,0.5,0.25", "0.6", "5"])
parser.add_argument("-op", "--outfile",
                    help="Where to output the ROC plot.",
                    default="../../graphs/classification/roc.png")
parser.add_argument("-d", "--data",
                    help="The file containing data.",
                    default="../../data/processed/data_0.npy")
parser.add_argument("-l", "--labels",
                    help="The file containing labels.",
                    default="../../data/processed/labels_0.npy")
parser.add_argument("-t", "--title",
                    help="Title of the plot")

# Data related -- Optional
parser.add_argument("-folds", "--folds", default=5, type=int,
                    help="Number of folds to use for cross validation.")
parser.add_argument("-n", "--note", default="",
                    help="Post-fix to file name.")
parser.add_argument("-s", "--save", action="store_true",
                    help="Whether to save the model. Does not run k-folds.")
parser.add_argument("-mf", "--model_file", default="model.h5",
                    help="The path of where to save the model.")
args = parser.parse_args()

# ========== Classification Methods ==========

# params: [num_neighbors]
def knn(X_train, y_train, X_test, params):
    model = KNeighborsClassifier(int(params[0])).fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1], model

# params: [num_iterations]
def log_reg(X_train, y_train, X_test, params):
    model = LogisticRegression(max_iter=int(params[0])).fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1]

# params: [regularization_strength]
def ridge(X_train, y_train, X_test, params):
    model = RidgeClassifier(alpha=float(params[0])).fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1]

# params: [kernel_type]
#   kernel_type = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
def svc(X_train, y_train, X_test, params):
    model = SVC(kernel=params[0], probability=True).fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1], model

# params: [l1_percentage],[l2_percentage],[l3_percentage],... [max_iter]
#   li_percentage = the number of nodes in layer i, as a percentage of the input size
def mlpc(X_train, y_train, X_test, params):
    model = MLPClassifier(
        tuple([int(float(i)*len(X_train[0])) for i in params[0].split(",")]),
        max_iter=int(params[1]))
    return model.fit(X_train, y_train).predict_proba(X_test)[:,1], model

# params: [num_estimators]
def rfc(X_train, y_train, X_test, params):
    model = RandomForestClassifier(n_estimators=int(params[0]), n_jobs=-1).fit(X_train, y_train)
    return model.predict_proba(X_test)[:,1]

# params: [l1_percentage],[l2_percentage],[l3_percentage],... [dropout] [epochs]
#   li_percentage = the number of nodes in layer i, as a percentage of the input size
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
    return np.squeeze(model.predict(X_test)), model

# params: [num_estimators]
def boosting(X_train, y_train, X_test, params):
    model = xgb.XGBClassifier(max_depth=10, n_estimators=int(params[0]), n_jobs=-1,)
    return model.fit(X_train, y_train).predict_proba(X_test)[:,1], model

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

def save(model, path, method):
    if method == "keras_model":
        model.save(path)
    else:
        dump(mode, path)

if __name__ == "__main__":
    print("Loading data.")
    x, y = np.load(args.data), np.load(args.labels)

    if not args.save:
        fprs = []
        tprs = []
        aucs = []
        labels = []

        cv = StratifiedKFold(n_splits=args.folds, random_state=0)

        print("Begining k-folds.")
        i = 0
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(x, y):
            y_scores, _ = globals()[args.method](x[train], y[train], x[test], args.params)
            fpr, tpr, thresholds = roc_curve(y[test], y_scores)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"ROC fold {i} (AUC={roc_auc:.2f})")
            print(f"Finished fold {i}.")
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
        plt.title(f"{get_classfication_name(args.method)} ROC Plot, Params: {' '.join(args.params)}" if args.title is None else args.title)
        plt.legend(loc="lower right")

        print(f"Saving plot to {args.outfile}")
        plt.savefig(args.outfile, dpi=600)
        plt.close("all")
        plt.cla()
    else:
        print(f"Training model {args.method} with params: {args.params}")
        y_scores, model = globals()[args.method](x, y, x, args.params)
        print(f"Average train loss: {np.average(y_scores)}")
        print(f"Saving to {args.model_file}")
        save(model, args.model_file, args.method)

