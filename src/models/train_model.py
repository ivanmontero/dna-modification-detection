from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from tensorflow import keras
import numpy as np
import argparse
import json
import time



def load(filename):
    with open(filename) as infile:
        contents = json.load(infile)
        data = np.array(contents['vectors'])
        labels = np.array(contents['labels'])

    return data, labels

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(150, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(50, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ['accuracy'])

    return model

def train_network(model, x_train, y_train, x_test = None, y_test = None, filename = None):
    if x_test is not None:
        history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, verbose = 0)
        y_scores = model.predict(x_test).reshape((-1))
        
        return y_scores, history.history
    else:
        model.fit(x_train, y_train, epochs = 10, verbose = 0)
        model.save(filename, save_format = 'tf')

def shortest_row(array):
    current = 0
    length = len(array[0])
    for i in range(len(array)):
        if len(array[i]) < length:
            current = i
    return array[i]

def interpolate_curve(x, y, area):
    
    interpolate_x = shortest_row(x)
    interpolate_y = []
    
    for i in range(len(x)):
        new_y = np.interp(interpolate_x, x[i], y[i])
        interpolate_y.append(new_y)
    
    mean_y = np.mean(interpolate_y, axis = 0)
    mean_area = np.mean(area)

    std_y = np.std(interpolate_y, axis = 0)
    std_area = np.std(area)

    lower_y = mean_y - std_y
    upper_y = mean_y + std_y
    
    return interpolate_x, mean_y, lower_y, upper_y, mean_area, std_area
    
def plot(x, y, name, filename):
    fprs = []
    tprs = []
    roc_aucs = []
    
    recalls = []
    precisions = []
    pr_aps = []
    
    training = []
    validation = []

    folds = 5
    i = 1
    for train, test in model_selection.KFold(n_splits = folds).split(x):
        print (f'Starting Fold {i}')
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        model = create_model()
        y_scores, history = train_network(model, x_train, y_train, x_test, y_test)
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)
        roc_auc = metrics.auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
        
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_scores)
        pr_ap = metrics.average_precision_score(y_test, y_scores)
        recalls.append(recall[::-1])
        precisions.append(precision[::-1])
        pr_aps.append(pr_ap)

        training.append(history['accuracy'])
        validation.append(history['val_accuracy'])

        i += 1
        
    mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(fprs, tprs, roc_aucs)
    
    plt.figure(figsize = (8, 4), dpi = 150, facecolor = 'white')
    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], linewidth = 1, alpha = 0.3, label = f"ROC Fold {i} (AUC = {roc_aucs[i]:.2f})")
    plt.fill_between(mean_x, lower_y, upper_y, color = 'grey', alpha = 0.2, label = r'$\pm \sigma$')
    plt.plot(mean_x, mean_y, color = 'C0', label = fr'Mean ROC (AUC = {mean_area:.2f} $\pm$ {std_area:.2f})', linewidth = 2)
    plt.plot([0, 1], [0, 1], linestyle = '--', color = 'black')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.title(f'{folds} Fold ROC with {name}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.savefig('test_roc.png')
    plt.close()
    
    mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(recalls, precisions, pr_aps)
    
    plt.figure(figsize = (8, 4), dpi = 150, facecolor = 'white')
    for i in range(len(recalls)):
        plt.plot(recalls[i], precisions[i], linewidth = 1, alpha = 0.3, label = f"PR Fold {i} (AP = {pr_aps[i]:.2f})")
    plt.fill_between(mean_x, lower_y, upper_y, color = 'grey', alpha = 0.2, label = r'$\pm \sigma$')
    plt.plot(mean_x, mean_y, color = 'C0', label = fr'Mean PR (AP = {mean_area:.2f} $\pm$ {std_area:.2f})', linewidth = 2)
    plt.plot([0, 1], [0.5, 0.5], linestyle = '--', color = 'black')
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.title(f'{folds} Fold PR with {name}')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.ylim([-0.1,1.1])
    plt.tight_layout()
    plt.savefig('test_pr.png')
    plt.close()
    
    mean_training = np.mean(training, axis = 0)
    mean_validation = np.mean(validation, axis = 0)
    epochs = range(1, len(mean_training) + 1)

    std_training = np.std(training, axis = 0)
    std_validation = np.std(validation, axis = 0)

    plt.figure(figsize = (8, 4), dpi = 150, facecolor = 'white')
    plt.plot(epochs, mean_training, color = 'C0', label = f'Training Accuracy (Final = {mean_training[-1]:.2f})', linewidth = 2)
    plt.fill_between(epochs, mean_training + std_training, mean_training - std_training, color = 'C0', alpha = 0.2)
    plt.plot(epochs, mean_validation, color = 'C1', label = f'Validation Accuracy (Final = {mean_validation[-1]:.2f})', linewidth = 2)
    plt.fill_between(epochs, mean_validation + std_training, mean_validation - std_training, color = 'C1', alpha = 0.2)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
    plt.title(f'{name} Training History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([-0.1,1.1])
    plt.tight_layout()
    plt.savefig('test_history.png')
    plt.close()

def main():
    print ('Reading data.')
    start = time.time()
    x, y = load('/active/myler_p/People/Sur/J-IP/LtaP/ivan-pacbio/chip_only.json')
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')
    
    print ('Testing Model')
    start = time.time()
    plot(x, y, 'Neural Network', 'test')
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Training Model')
    start = time.time()
    model = create_model()
    train_network(model, x, y, filename = 'test.h5')
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

if __name__ == '__main__':
    main()