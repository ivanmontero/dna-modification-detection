from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import tqdm
import json
import time

# Import Progress Bars
import progress_bars

# Import Helper Functions
import os
import sys
current_path = os.path.dirname(__file__)
utils_path = os.path.join(current_path, '..', 'utils')
sys.path.append(utils_path )
import utils

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Train a neural network and save it.')

    parser.version = 0.5

    parser.add_argument(
        '-i', 
        '--input', 
        required = True,
        help = 'Input HDF table with feature vectors.')

    parser.add_argument(
        '--fold-change',
        default = 10,
        type = float, 
        help = 'Fold change threshold value.')

    parser.add_argument(
        '-e',
        '--examples',
        default = 1000,
        type = int,
        help = 'Number of examples from each class.')
    
    parser.add_argument(
        '--holdout',
        default = None, 
        help = 'Which chromosome to holdout for testing.')
    
    parser.add_argument(
        '-c',
        '--center',
        default=False,
        action='store_true',
        help = 'Whether to only center on As and Ts')

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    return parser.parse_args()

def train_dataset(data, n_examples, threshold, holdout = None):
    # Drop rows where there are any nulls.
    data.dropna(inplace = True)

    # Add a label column based on fold change threshold.
    data.loc[data['fold_change'] >= threshold, 'labels'] = 1
    data.loc[data['fold_change'] < threshold, 'labels'] = 0

    # Remove test holdout chromosome. 
    chromosomes = data.index.unique(level = 'chromosome').to_list()
    if holdout:
        chromosomes = chromosomes.remove(holdout)
        data = data.loc[[chromosomes], :, :]

    training = []
    validation = []
    # TODO: replace with variable
    model = create_model(306)
    for chromosome in chromosomes:
        results = train_fold(data, n_examples, model, chromosome)
        training_history
        
        training.append(training_history)
        validation.append(validation_history)

    return training, validation

def train_fold(data, n_examples, model, holdout, batch_size = 32):

    # Separate into training and validation. 
    chromosomes = data.index.unique(level = 'chromosome').to_list()
    chromosomes.remove(holdout)
    training_fold = data.loc[chromosomes, :, :]
    validation_fold = data.loc[holdout, :, :]

    # Filter on labels.
    positive = training_fold.loc[training_fold['labels'] == 1]
    negative = training_fold.loc[training_fold['labels'] == 0]

    # Sample n examples.
    positive = sample(positive, n_examples)
    negative = sample(negative, n_examples)

    # Convert to numpy.
    # Unfortunately there's this weird thing where if the numpy array has lists
    # of uneven length, or in our case, Nones, then it makes a weird array that
    # tensorflow doesn't like. We first have to convert the numpy array of numpy
    # arrays into a list of numpy arrays then convert it back. There must be 
    # something more elegant but this is what works for now.
    positive_examples = np.array(list(positive['vectors'].to_numpy()))
    positive_labels = positive['labels'].to_numpy()
    negative_examples = np.array(list(negative['vectors'].to_numpy()))
    negative_labels = negative['labels'].to_numpy()
    validation_examples = np.array(list(validation_fold['vectors'].to_numpy()))
    validation_labels = validation_fold['labels'].to_numpy()

    # Aggregate training examples and labels.
    train_examples = np.vstack([positive_examples, negative_examples])
    train_labels = np.hstack([positive_labels, negative_labels])

    # Shuffle the order of examples.
    train_length = len(train_examples)
    index = np.random.permutation(train_length)
    train_examples = train_examples[index]
    train_labels = train_labels[index]

    # Convert to tensorflow dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_examples)
    validation_dataset = validation_dataset.batch(batch_size)

    # Compute the number of batches.
    train_length = int(np.ceil(train_length/batch_size))
    validation_length = int(np.ceil(len(validation_examples)/batch_size))

    # Train the network, then validate, then reset for the next fold.
    training_history, validation_history = train_network(model, train_dataset, train_length)
    validation_scores = validate_network(model, validation_dataset, validation_length)
    model.reset_states()
    
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(validation_labels, validation_scores)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    
    precision, recall, thresholds = metrics.precision_recall_curve(validation_labels, validation_scores)
    pr_ap = metrics.average_precision_score(y_test, y_scores)

    return (
        training_history, 
        validation_history, 
        false_positive_rate, 
        true_positive_rate, 
        roc_auc, precision[::-1], 
        recall[::-1], 
        pr_ap)

def sample(data, n_examples):
    if len(data) <= n_examples:
        return data
    else:
        return data.sample(n_examples)

# We will define our model as a multi-layer densely connected neural network
# with dropout between the layers.
def create_model(input_dim):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            300, 
            input_dim = input_dim, 
            activation="relu"
        ))
    model.add(
        keras.layers.Dropout(
            0.5
        ))
    model.add(
        keras.layers.Dense(
            150, 
            activation="relu"
        ))
    model.add(
        keras.layers.Dropout(
            0.5
        ))
    model.add(
        keras.layers.Dense(
            50, 
            activation="relu"
        ))
    model.add(
        keras.layers.Dense(
            1, 
            activation="sigmoid"
        ))
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics = ['accuracy'], 
        )

    return model

# Begin training the neural network, with optional validation data, and model saving.
def train_network(model, training_dataset, length, validation_split = 0.1):
    # TODO: There doesn't seem to be a great way to get the length of a
    # tf.DataSet, so instead we resort to passing the variable. 
    n_validation = int(length*validation_split)
    validation_dataset = training_dataset.take(n_validation) 
    training_dataset = training_dataset.skip(n_validation)

    # Create out custom TQDM progress bar for training. 
    callback = progress_bars.train_progress(length)

    history = model.fit(
        training_dataset,
        validation_data = validation_dataset,
        epochs = 10, 
        verbose = 0,
        use_multiprocessing = True, 
        callbacks = [callback])
    
    return history.history['accuracy'], history.history['val_accuracy']

def validate_network(model, validation_dataset, length):
    callback = progress_bars.predict_progress(length)

    scores = model.predict(
        validation_dataset,
        use_multiprocessing = True,
        callbacks = [callback])

    return scores

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
    
# Run k-folds cross validation on the provided data and write the
# ROC, Accuracy vs. Epoch, and Precision vs. Recall curves.
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

        model = create_model(len(x_train[0]))
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

    with PdfPages(filename) as pdf: 
        mean_training = np.mean(training, axis = 0)
        mean_validation = np.mean(validation, axis = 0)
        epochs = range(1, len(mean_training) + 1)

        std_training = np.std(training, axis = 0)
        std_validation = np.std(validation, axis = 0)

        plt.figure(
            figsize = (8, 4), 
            dpi = 150, 
            facecolor = 'white')
        plt.plot(
            epochs, 
            mean_training, 
            color = 'C0',
            linewidth = 2,
            label = f'Training Accuracy (Final = {mean_training[-1]:.2f})')
        plt.fill_between(
            epochs, 
            mean_training + std_training, 
            mean_training - std_training, 
            color = 'C0', 
            alpha = 0.2)
        plt.plot(
            epochs, 
            mean_validation, 
            color = 'C1',  
            linewidth = 2,
            label = f'Validation Accuracy (Final = {mean_validation[-1]:.2f})')
        plt.fill_between(
            epochs, 
            mean_validation + std_training, 
            mean_validation - std_training,
            color = 'C1', 
            alpha = 0.2)
        plt.legend(
            bbox_to_anchor = (1.05, 1), 
            loc = 'upper left')
        plt.title(f'{name} Training History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim([-0.1,1.1])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(fprs, tprs, roc_aucs)
    
        plt.figure(
            figsize = (8, 4), 
            dpi = 150, 
            facecolor = 'white')
        for i in range(len(fprs)):
            plt.plot(
                fprs[i], 
                tprs[i], 
                linewidth = 1, 
                alpha = 0.3, 
                label = f"ROC Fold {i} (AUC = {roc_aucs[i]:.2f})")
        plt.fill_between(
            mean_x, 
            lower_y, 
            upper_y, 
            color = 'grey', 
            alpha = 0.2, 
            label = r'$\pm \sigma$')
        plt.plot(
            mean_x, 
            mean_y, 
            color = 'C0',
            linewidth = 2,
            label = fr'Mean ROC (AUC = {mean_area:.2f} $\pm$ {std_area:.2f})')
        plt.plot(
            [0, 1], 
            [0, 1], 
            linestyle = '--', 
            color = 'black')
        plt.legend(
            bbox_to_anchor = (1.05, 1), 
            loc = 'upper left')
        plt.title(f'{folds} Fold ROC with {name}')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
        mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(recalls, precisions, pr_aps)
        
        plt.figure(
            figsize = (8, 4), 
            dpi = 150, 
            facecolor = 'white')
        for i in range(len(recalls)):
            plt.plot(
                recalls[i], 
                precisions[i], 
                linewidth = 1,
                alpha = 0.3,
                label = f"PR Fold {i} (AP = {pr_aps[i]:.2f})")
        plt.fill_between(
            mean_x, 
            lower_y, 
            upper_y, 
            color = 'grey', 
            alpha = 0.2, 
            label = r'$\pm \sigma$')
        plt.plot(
            mean_x, 
            mean_y, 
            color = 'C0', 
            linewidth = 2,
            label = fr'Mean PR (AP = {mean_area:.2f} $\pm$ {std_area:.2f})')
        plt.plot(
            [0, 1], 
            [0.5, 0.5], 
            linestyle = '--', 
            color = 'black')
        plt.legend(
            bbox_to_anchor = (1.05, 1), 
            loc = 'upper left')
        plt.title(f'{folds} Fold PR with {name}')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.ylim([-0.1,1.1])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def main():

    # Get rid of random tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    data = pd.read_pickle('/active/myler_p/People/Sur/software/dna-modification-detection/data/processed/test_data.p')
    training, validation = train_dataset(data, 10000, 10, None)
    
    # total_start = utils.start_time()
    # # Get argparse arguments. 
    # arguments = setup()

    # start = utils.start_time('Reading data.')
    # data = pd.read_hdf(arguments.infile)
    # utils.end_time(start)
        
    # start = utils.start_time('Testing Model')

    # project_folder = utils.project_path()
    # reports_folder = os.path.join(project_folder, 'reports')
    # if arguments.prefix:
    #     filename = os.path.join(reports_folder, f'{arguments.prefix}_model_performance.pdf')
    # else:
    #     filename = os.path.join(reports_folder, 'model_performance.pdf')

    # plot(x, y, 'Neural Network', filename)
    # utils.end_time(start) 

    # start = utils.start_time('Training Model')

    # models_folder = os.path.join(project_folder, 'models')
    # if arguments.prefix:
    #     filename = os.path.join(models_folder, f'{arguments.prefix}_model.h5')
    # else:
    #     filename = os.path.join(models_folder, 'model.h5')

    # model = create_model(len(x[0]))
    # train_network(model, x, y, filename = filename)
    # utils.end_time(start)

    # total_time = utils.end_time(total_start, True)
    # print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()