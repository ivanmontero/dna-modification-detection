import argparse

def import_modules():
    print ('Importing Modules')
    # Make the local imports global
    global os
    global sys
    global utils
    global progress_bars
    global PdfPages
    global plt
    global keras 
    global metrics
    global tf
    global pd
    global np
    global json

    # Import Local Helper Functions
    import os
    import sys
    current_path = os.path.dirname(__file__)
    utils_path = os.path.join(current_path, '..', 'utils')
    sys.path.append(utils_path)
    import utils
    
    start = utils.start_time()
    # Local Import
    import progress_bars

    # Import Everything Else
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    from tensorflow import keras
    from sklearn import metrics
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import json
    utils.end_time(start)

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Train a neural network and save it.', 
        formatter_class = argparse.RawTextHelpFormatter)

    parser.version = 0.6

    parser.add_argument(
        '-i', 
        '--input', 
        required = True,
        help = 'Input numpy array of feature vectors.')

    parser.add_argument(
        '-d', 
        '--data-frame', 
        required = True,
        help = 'Input pandas dataframe.')

    parser.add_argument(
        '-m', 
        '--metadata',
        required = True,
        help = 'Metadata for input table.')

    parser.add_argument(
        '-f',
        '--fold-change',
        default = 10,
        type = float, 
        help = 'Fold change threshold value.\n(Default: 10)')

    parser.add_argument(
        '-n',
        '--n-examples',
        default = 5000,
        type = int,
        help = 'Number of total examples to train on.\n(Default: 5,000)')
    
    parser.add_argument(
        '-o',
        '--holdout',
        default = None, 
        help = 'Which chromosome to holdout for testing.\n(Default: None)') 

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    # Whether to only center on As and Ts'
    parser.add_argument(
        '--center',
        default = False,
        action = 'store_true',
        help = argparse.SUPPRESS)

    # Skip the final train step.
    parser.add_argument(
        '--skip-final',
        default = False,
        action = 'store_true',
        help = argparse.SUPPRESS)

    # Train on all examples instead of sampling. 
    parser.add_argument(
        '--train-all',
        default = False,
        action = 'store_true',
        help = argparse.SUPPRESS)

    # Turn the progress bars off.
    parser.add_argument(
        '--min-peak-length',
        default = 20,
        type = int,
        help = argparse.SUPPRESS)

    return parser.parse_args()

def label_peaks(dataframe, threshold, min_peak_length):
    # TODO: Make this better. 
    chromosome_list = dataframe.index.get_level_values('chromosome').to_list()
    chromosome_length = [0]
    for chromosome in dataframe.index.unique('chromosome'):
        chromosome_length.append(chromosome_list.count(chromosome))
    chromosome_length = np.cumsum(chromosome_length)

    fold_change = dataframe['fold_change'].to_numpy()
    dataframe = dataframe.assign(peak_id = 0)
    peak_id = 1

    for i in range(len(chromosome_length) - 1):
        start = chromosome_length[i]
        end = chromosome_length[i+1]

        in_peak = False
        for j in range(start, end):
            current_fold_change = fold_change[j]

            # If you are currently in a peak.
            if current_fold_change > threshold:
                in_peak = True
                dataframe.iat[j, -1] = peak_id

            # If you have just exited a peak.
            elif in_peak:
                in_peak = False
                
                # If the peak was too small. 
                if (dataframe['peak_id'] == peak_id).sum() < min_peak_length:
                    dataframe.loc[dataframe['peak_id'] == peak_id, 'peak_id'] = 0
                else:
                    peak_id += 1

        # At the end of each chromosome.             
        if in_peak:
            peak_id = peak_id + 1

    return dataframe

def train_dataset(vectors, dataframe, threshold, n_examples, holdout, window, min_peak_length, center, train_all):
    # Drop rows where there are any nulls.
    condition = vectors.any(axis = 1)
    vectors = vectors[condition]
    dataframe = dataframe.loc[condition]

    # Add a label column based on fold change threshold.
    label = (dataframe['fold_change'] >= threshold).astype(int)
    dataframe = dataframe.assign(label = label)

    # Add a peak column based on where ChIP peaks occur.
    dataframe = label_peaks(dataframe, threshold, min_peak_length)

    # Remove test holdout chromosome.
    index = dataframe.index.get_level_values('chromosome')
    chromosomes = dataframe.index.unique(level = 'chromosome').to_list()
    if holdout:
        chromosomes = chromosomes.remove(holdout)
        condition = (index != holdout)
        vectors = vectors[condition]
        dataframe = dataframe.loc[condition]

    # Filter out non-centers if specified
    if center:
        condition = ((dataframe['top_A'] == 1) | (dataframe['top_T'] == 1))
        vectors = vectors[condition]
        dataframe = dataframe.loc[condition]

    # Percent of rows which are in peaks.
    peak_percent = dataframe['label'].sum()/len(dataframe)

    # Training History
    training_history = []
    validation_history = []

    # ROC Curve
    false_positive_rate = []
    true_positive_rate = []
    roc_auc = []
    
    # PR Curve
    recall = []
    precision = []
    average_precision = []

    # Peak vs. J curve
    peaks_with_j = []
    js_in_peak = []
    peak_auc = []

    model = create_model(window)
    for chromosome in chromosomes:
        results = train_fold(vectors, dataframe, model, n_examples, chromosome, train_all = train_all)
        training_history.append(results['training_history'])
        validation_history.append(results['validation_history'])
        false_positive_rate.append(results['false_positive_rate'])
        true_positive_rate.append(results['true_positive_rate'])
        roc_auc.append(results['roc_auc'])
        recall.append(results['recall'])
        precision.append(results['precision'])
        average_precision.append(results['average_precision'])
        peaks_with_j.append(results['peaks_with_j'])
        js_in_peak.append(results['js_in_peak'])
        peak_auc.append(results['peak_auc'])

    return (
        vectors,
        dataframe,
        model,
    {
        'training_history': training_history, 
        'validation_history': validation_history, 
        'false_positive_rate': false_positive_rate, 
        'true_positive_rate': true_positive_rate, 
        'roc_auc':roc_auc,
        'precision': precision, 
        'recall': recall, 
        'average_precision': average_precision,
        'peak_percent': round(peak_percent, 4),
        'peaks_with_j': peaks_with_j,
        'js_in_peak': js_in_peak,
        'peak_auc': peak_auc
    })

def train_fold(vectors, dataframe, model, n_examples, holdout, batch_size = 32, train_all = False):

    # Separate into training and validation.
    index = dataframe.index.get_level_values('chromosome')
    labels = dataframe['label'].to_numpy()

    # Validation Data
    validation_examples = vectors[index == holdout]
    validation_labels = labels[index == holdout]
    peak_ids = dataframe['peak_id'].to_numpy()[index == holdout]

    # TODO: Takes about 40 seconds on the whole dataset, kinda slow. 
    # Training Data
    train_examples = vectors[index != holdout]
    training_labels = labels[index != holdout]

    # TODO: Also takes about 40 seconds
    # Filter on labels.
    positive_examples = train_examples[training_labels == 1]
    negative_examples = train_examples[training_labels == 0]
    
    # Sample n examples.
    n_examples = int(n_examples/2)
    if not train_all: 
        positive_examples = sample(positive_examples, n_examples)
        negative_examples = sample(negative_examples, n_examples)

    # Aggregate training examples and labels.
    train_examples = np.vstack([positive_examples, negative_examples])
    positive_labels = np.ones(len(positive_examples))
    negative_labels = np.zeros(len(negative_examples))
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
    training_history, validation_history = train_network(train_dataset, model, train_length)
    validation_scores = validate_network(validation_dataset, model, validation_length)
    model.reset_states()

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(validation_labels, validation_scores)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    precision, recall, thresholds = metrics.precision_recall_curve(validation_labels, validation_scores)
    average_precision = metrics.average_precision_score(validation_labels, validation_scores)

    peaks_with_j, js_in_peak = peak_j_curve(peak_ids, validation_scores)
    peak_auc = metrics.auc(peaks_with_j, js_in_peak)

    return {
        'training_history': training_history, 
        'validation_history': validation_history, 
        'false_positive_rate': false_positive_rate, 
        'true_positive_rate': true_positive_rate, 
        'roc_auc':roc_auc,
        'precision': precision[::-1], 
        'recall': recall[::-1], 
        'average_precision': average_precision,
        'peaks_with_j': peaks_with_j,
        'js_in_peak': js_in_peak,
        'peak_auc': peak_auc
    }

# Begin training the neural network, with optional validation data, and model saving.
def train_network(training_dataset, model, length, validation_split = 0.1):
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
        callbacks = [callback])
    
    return history.history['accuracy'], history.history['val_accuracy']

def validate_network(validation_dataset, model, length):
    callback = progress_bars.predict_progress(length)

    scores = model.predict(
        validation_dataset,
        callbacks = [callback])

    return scores.reshape(-1)

# TODO: Right now it trains on everything in the end. Maybe that's too much? 
# Maybe it won't generalize? Tough to say, we should probably check using the 
# cross validation step. 
def train_final(vectors, dataframe, model, batch_size = 32):
    # Extract examples and labels.
    labels = dataframe['label'].to_numpy()

    # Shuffle the order of examples.
    index = np.random.permutation(len(vectors))
    vectors = vectors[index]
    labels = labels[index]

    # Create tensorflow dataset and batch.
    data = tf.data.Dataset.from_tensor_slices((vectors, labels))
    data = data.batch(batch_size)

    # Create progress bar.
    length = int(np.ceil(len(vectors)/batch_size))
    callback = progress_bars.train_progress(length)

    model.fit(
        data,
        epochs = 10, 
        verbose = 0, 
        callbacks = [callback])

def sample(vectors, n_examples):
    if len(vectors) <= n_examples:
        return vectors
    else:
        index = np.arange(len(vectors))
        selection = np.random.choice(index, n_examples, replace = False)
        return vectors[selection]

# We will define our model as a multi-layer densely connected neural network
# with dropout between the layers.
def create_model(input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(300, input_dim = input_dim, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(150, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics = ['accuracy'])

    return model

def peak_j_curve(peak_ids, y_scores):
    # Sort the y_scores largest to smallest and order peak_ids by its index.
    sort_index = np.argsort(y_scores, kind="mergesort")[::-1]
    peak_ids = peak_ids[sort_index]
    y_scores = y_scores[sort_index]

    # Compute cumulative number of unique peaks as we lower the threshold.
    unique_peaks = {0}
    count_peaks = np.zeros(len(peak_ids))
    for i in range(len(peak_ids)):
        unique_peaks.add(peak_ids[i])
        count_peaks[i] = len(unique_peaks) - 1

    # Compute percent peaks with J.
    min_peak = min(peak_ids[peak_ids > 0])
    max_peak = max(peak_ids)
    total_peaks = max_peak - min_peak + 1
    peaks_with_j = count_peaks / total_peaks

    # Compute number of J's in peaks.
    in_peak = (peak_ids > 0).astype(int)
    count_js = np.cumsum(in_peak)
    js_with_peak = count_js / np.arange(1, len(y_scores) + 1)

    # Compute unique thresholds and add a 0 threshold.
    distinct_index = np.where(np.diff(y_scores))[0]
    threshold_index = np.concatenate([distinct_index, [len(y_scores) - 1]])

    peaks_with_j = np.concatenate([[0], peaks_with_j[threshold_index]])
    js_with_peak = np.concatenate([[1], js_with_peak[threshold_index]])

    return peaks_with_j, js_with_peak

def shortest_row(array):
    current = 0
    length = len(array[0])
    for i in range(len(array)):
        if len(array[i]) < length:
            current = i
    return array[i]

def interpolate_curve(x, y, area, average = False):
    
    if not average:
        mean_x = shortest_row(x)
        interpolate_y = []
            
        for i in range(len(x)):
            new_y = np.interp(mean_x, x[i], y[i])
            interpolate_y.append(new_y)
        
        mean_y = np.mean(interpolate_y, axis = 0)
        std_y = np.std(interpolate_y, axis = 0)

    else:
        mean_x = np.mean(x, axis = 0)
        mean_y = np.mean(y, axis = 0)
        std_y = np.std(y, axis = 0)

    mean_area = np.mean(area)
    std_area = np.std(area)

    lower_y = mean_y - std_y
    upper_y = mean_y + std_y
    
    return mean_x, mean_y, lower_y, upper_y, mean_area, std_area

def plot(
    filename,
    training_history, 
    validation_history, 
    false_positive_rate, 
    true_positive_rate, 
    roc_auc, 
    precision,
    recall, 
    average_precision,
    peak_percent,
    peaks_with_j,
    js_in_peak,
    peak_auc,
    name = 'Neural Network'):

    with PdfPages(filename) as pdf:
        folds = len(training_history)
        mean_training = np.mean(training_history, axis = 0)
        mean_validation = np.mean(validation_history, axis = 0)
        epochs = range(1, len(mean_training) + 1)

        std_training = np.std(training_history, axis = 0)
        std_validation = np.std(validation_history, axis = 0)

        # Training Plot
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
        plt.title(f'{folds} Fold Training with {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([-0.1,1.1])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(false_positive_rate, true_positive_rate, roc_auc)

    
        # ROC Curve
        plt.figure(
            figsize = (8, 4), 
            dpi = 150, 
            facecolor = 'white')
        for i in range(len(false_positive_rate)):
            plt.plot(
                false_positive_rate[i], 
                true_positive_rate[i], 
                linewidth = 1, 
                alpha = 0.3)
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
        plt.xlabel('False Positive Rate')
        plt.xlim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.ylim([-0.1,1.1])
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    
        mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(recall, precision, average_precision)
        
        # Precision Recall Curve
        plt.figure(
            figsize = (8, 4), 
            dpi = 150, 
            facecolor = 'white')
        for i in range(len(recall)):
            plt.plot(
                recall[i], 
                precision[i], 
                linewidth = 1,
                alpha = 0.3)
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
            [peak_percent, peak_percent], 
            linestyle = '--', 
            color = 'black')
        plt.legend(
            bbox_to_anchor = (1.05, 1), 
            loc = 'upper left')
        plt.title(f'{folds} Fold PR with {name}')
        plt.xlabel('Recall')
        plt.xlim([-0.1, 1.1])
        plt.ylabel('Precision')
        plt.ylim([-0.1,1.1])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        mean_x, mean_y, lower_y, upper_y, mean_area, std_area = interpolate_curve(peaks_with_j, js_in_peak, peak_auc)
        
        # Peak J Curve
        plt.figure(
            figsize = (8, 4), 
            dpi = 150, 
            facecolor = 'white')
        for i in range(len(peaks_with_j)):
            plt.plot(
                peaks_with_j[i], 
                js_in_peak[i], 
                linewidth = 1,
                alpha = 0.3)
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
            label = fr'Mean AUC = {mean_area:.2f} $\pm$ {std_area:.2f}')
        plt.plot(
            [0, 1], 
            [peak_percent, peak_percent], 
            linestyle = '--', 
            color = 'black')
        plt.legend(
            bbox_to_anchor = (1.05, 1), 
            loc = 'upper left')
        plt.title(f'{folds} Fold Peak Recall with {name}')
        plt.xlabel('% Peaks with J')
        plt.xlim([-0.1, 1.1])
        plt.ylabel('% Js in Peak')
        plt.ylim([-0.1,1.1])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def main():
    # Get argparse arguments. 
    arguments = setup()

    # Import modules. It's placed here since it takes like 20 seconds. 
    import_modules()

    total_start = utils.start_time()
    # Get rid of random tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Reading data. 
    start = utils.start_time('Reading Data')
    dataframe = pd.read_hdf(arguments.data_frame)
    vectors = np.load(arguments.input)
    with open(arguments.metadata) as infile:
        metadata = json.load(infile)
    window = len(metadata['columns'])
    utils.end_time(start) 

    # Training model.
    start = utils.start_time()
    vectors, dataframe, model, results = train_dataset(
        vectors = vectors,
        dataframe = dataframe,
        threshold = arguments.fold_change,
        n_examples = arguments.n_examples,
        holdout = arguments.holdout, 
        window = window,
        min_peak_length = arguments.min_peak_length,
        center = arguments.center,
        train_all = arguments.train_all)
    utils.end_time(start) 

    # Plotting performance. 
    print ('Plotting Performance')
    project_folder = utils.project_path()
    reports_folder = os.path.join(project_folder, 'reports')
    # Create filename.
    if arguments.prefix:
        filename = os.path.join(reports_folder, f'{arguments.prefix}_model_performance.pdf')
    else:
        filename = os.path.join(reports_folder, 'model_performance.pdf')
    plot(filename, **results)

    if not arguments.skip_final:
        start = utils.start_time('Training Final Model')
        models_folder = os.path.join(project_folder, 'models')
        if arguments.prefix:
            filename = os.path.join(models_folder, f'{arguments.prefix}_model.h5')
        else:
            filename = os.path.join(models_folder, 'model.h5')
        train_final(vectors, dataframe, model)
        model.save(filename)
        utils.end_time(start)

    total_time = utils.end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()