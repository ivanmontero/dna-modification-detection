import argparse

def import_modules():
    print ('Importing Modules')
    # Make the local imports global
        
    global progress_bars
    global plot_metrics
    global metrics
    global utils
    global keras
    global copy 
    global json
    global sys
    global tf
    global pd
    global np
    global os

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
    import plot_metrics

    # Import Everything Else
    from tensorflow import keras
    from sklearn import metrics
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import json
    import copy
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

    # Whether the final model is trained on only As and Ts
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

def simple_threshold(dataframe):
    top_ipd = dataframe['top_ipd'].to_numpy()
    bottom_ipd = dataframe['bottom_ipd'].to_numpy()
    scores = np.maximum(top_ipd, bottom_ipd)

    return scores

#      Top: +6              +2      +0    
# Position:  0   1   2   3   4   5   6   7   8   9   10  11  12
#   Bottom:                         +0      +2               +6
def complex_threshold(dataframe):
    top_ipd = dataframe['top_ipd'].to_numpy()
    bottom_ipd = dataframe['bottom_ipd'].to_numpy()

    top_zero = top_ipd[6:]
    top_two = top_ipd[4:-2]
    top_six = top_ipd[:-6]

    bottom_zero = bottom_ipd[:-6]
    bottom_two = bottom_ipd[2:-4]
    bottom_six = bottom_ipd[6:]

    top_sum = top_zero + top_two + top_six
    bottom_sum = bottom_zero + bottom_two + bottom_six
    minimum = np.amin(top_sum)
    scores = np.full(len(dataframe), minimum)

    scores[6:-6] = np.maximum(top_sum[:-6], bottom_sum[6:])

    return scores

def get_metrics(dataframe, scores):
    condition = ((dataframe['top_A'] == 1) | (dataframe['top_T'] == 1))
    dataframe = dataframe.loc[condition]

    if len(scores) != len(dataframe):
        scores = scores[condition]

    labels = dataframe['label'].to_numpy()
    peak_ids = dataframe['peak_id'].to_numpy()

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, scores)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    average_precision = metrics.average_precision_score(labels, scores)

    peaks_with_j, js_with_peak = peak_j_curve(peak_ids, scores)
    peak_auc = metrics.auc(peaks_with_j, js_with_peak)

    receiver_operator = {
        'x': false_positive_rate,
        'y': true_positive_rate,
        'area': roc_auc
    }

    # The recall and precision have to be reversed because of something to do
    # with the way average curves are calculated.  
    precision_recall = {
        'x': recall[::-1],
        'y': precision[::-1],
        'area': average_precision    
    }

    peak_curve = {
        'x': peaks_with_j,
        'y': js_with_peak,
        'area': peak_auc   
    }

    return receiver_operator, precision_recall, peak_curve

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

# TODO: Right now it trains on everything in the end. Maybe that's too much? 
# Maybe it won't generalize? Tough to say, we should probably check using the 
# cross validation step. 
# TODO: Split to have equal classes?
def train_final(vectors, dataframe, model, center = False, batch_size = 32):
    # Extract examples and labels.
    labels = dataframe['label'].to_numpy()

    # Shuffle the order of examples.
    index = np.random.permutation(len(vectors))
    vectors = vectors[index]
    labels = labels[index]

    # Filter out non-centers if specified
    if center:
        condition = ((dataframe['top_A'] == 1) | (dataframe['top_T'] == 1))
        vectors = vectors[condition]
        dataframe = dataframe.loc[condition]

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

def create_training_fold(vectors, labels, n_examples, batch_size = 32, train_all = False):
    # Filter on labels.
    positives = vectors[labels == 1]
    negatives = vectors[labels == 0]

    # Sample n examples.
    n_examples = int(n_examples/2)
    if not train_all: 
        positives = sample(positives, n_examples)
        negatives = sample(negatives, n_examples)

    # Aggregate training examples and labels.
    vectors = np.vstack([positives, negatives])
    labels = np.hstack([np.ones(len(positives)), np.zeros(len(negatives))])

    # Shuffle the order of examples.
    length = len(vectors)
    index = np.random.permutation(length)
    vectors = vectors[index]
    labels = labels[index]

    # Convert to tensorflow dataset.
    dataset = tf.data.Dataset.from_tensor_slices((vectors, labels))
    dataset = dataset.batch(batch_size)

    # Compute the number of batches.
    length = int(np.ceil(length/batch_size))

    return dataset, length

def create_validation_fold(vectors, labels, batch_size = 32):
    # Convert to tensorflow dataset.
    dataset = tf.data.Dataset.from_tensor_slices((vectors, labels))
    dataset = dataset.batch(batch_size)

    # Compute the number of batches.
    length = int(np.ceil(len(vectors)/batch_size))

    return dataset, length

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
        callbacks = [callback])

    epochs = range(1, len(history.history['accuracy']) + 1)
    training_history = {'x': epochs, 'y': history.history['accuracy'], 'area': None}
    validation_history = {'x': epochs, 'y': history.history['val_accuracy'], 'area': None}
    
    return training_history, validation_history

def validate_network(model, validation_dataset, length):
    callback = progress_bars.predict_progress(length)

    scores = model.predict(
        validation_dataset,
        callbacks = [callback])

    return scores.reshape(-1)

def train_fold(model, vectors, dataframe, n_examples, train_all):
    labels = dataframe['label'].to_numpy()
    dataset, length = create_training_fold(
        vectors = vectors,
        labels = labels,
        n_examples = n_examples,
        train_all = train_all)

    training_history, validation_history = train_network(model, dataset, length)

    return training_history, validation_history

def validate_fold(model, vectors, dataframe):
    labels = dataframe['label'].to_numpy()
    dataset, length = create_validation_fold(
        vectors = vectors,
        labels = labels)

    scores = validate_network(model, dataset, length)

    return  scores

def train_dataset(vectors, dataframe, threshold, n_examples, holdout, window, min_peak_length, train_all):
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
        chromosomes.remove(holdout)
        condition = (index != holdout)
        vectors = vectors[condition]
        dataframe = dataframe.loc[condition]

    # Remove the Maxi_A chromosome.
    if 'MaxiA' in chromosomes:
        index = dataframe.index.get_level_values('chromosome')
        chromosomes.remove('MaxiA')
        condition = (index != 'MaxiA')
        vectors = vectors[condition]
        dataframe = dataframe.loc[condition]

    # Percent of rows which are in peaks.
    peak_percent = dataframe['label'].sum()/len(dataframe)

    results = {
        'history': [],
        'receiver_operator': [],
        'precision_recall': [],
        'peak_curve': [],
        'peak_baseline': peak_percent
    }

    model = create_model(window)
    index = dataframe.index.get_level_values('chromosome')
    for holdout in chromosomes:
        # Training fold.
        training_vectors = vectors[index != holdout]
        training_dataframe = dataframe.loc[index != holdout]

        # Validation fold.
        validation_vectors = vectors[index == holdout]
        validation_dataframe = dataframe.loc[index == holdout]

        # Complex threshold baseline.
        scores = complex_threshold(validation_dataframe)

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

        # Only keep the T's. 
        condition = ((validation_dataframe['top_A'] == 1) | (validation_dataframe['top_T'] == 1))
        validation_vectors = validation_vectors[condition]
        validation_dataframe = validation_dataframe.loc[condition]

        # Simple threshold baseline.
        scores = simple_threshold(validation_dataframe)

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

        training_history, validation_history = train_fold(
            model = model,
            vectors = training_vectors,
            dataframe = training_dataframe,
            n_examples = n_examples,
            train_all = train_all)

        # Store history.
        results['history'].append(training_history)
        results['history'].append(validation_history)

        scores = validate_fold(
            model = model,
            vectors = validation_vectors,
            dataframe = validation_dataframe)
        model.reset_states()

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

        # Only keep the T's. 
        condition = ((training_dataframe['top_A'] == 1) | (training_dataframe['top_T'] == 1))
        training_vectors = training_vectors[condition]
        training_dataframe = training_dataframe.loc[condition]

        training_history, validation_history = train_fold(
            model = model,
            vectors = training_vectors,
            dataframe = training_dataframe,
            n_examples = n_examples,
            train_all = train_all)

        # Store history.
        results['history'].append(training_history)
        results['history'].append(validation_history)

        scores = validate_fold(
            model = model,
            vectors = validation_vectors,
            dataframe = validation_dataframe)
        model.reset_states()

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

    return results

def order_results(results):
    reorder = {
        'history': [],
        'receiver_operator': [],
        'precision_recall': [],
        'peak_curve': [],
    }

    total_curves = {
        'history': 4,
        'receiver_operator': 4,
        'precision_recall': 4,
        'peak_curve': 4
    }

    for item in reorder:
        curve = {
            'x': [],
            'y': [],
            'area': [],
            'label': None,
        }

        for i in range(total_curves[item]):
            reorder[item].append(copy.deepcopy(curve))

    history_labels = [
        'All Bases (Training)',
        'All Bases (Validation)',
        'Only T (Training)',
        'Only T (Validation)']

    other_labels = [
        'Complex Threshold',
        'Simple Threshold',
        'All Bases',
        'Only T'
    ]

    for i in range(len(history_labels)):
        reorder['history'][i]['label'] = history_labels[i]

    for metric in ['receiver_operator', 'precision_recall', 'peak_curve']:
        for i in range(len(other_labels)):
            reorder[metric][i]['label'] = other_labels[i]

    for metric in reorder:
        current_result = results[metric]
        current_metric = reorder[metric]

        curves = total_curves[metric]
        folds = int(len(current_result)/curves)

        n = 0
        for i in range(folds):
            for j in range(curves):
                current_fold = current_result[n]
                for item in current_fold:
                    current_metric[j][item].append(current_fold[item])
                n += 1

    reorder['peak_baseline'] = results['peak_baseline']
    return reorder

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
    print("Training Model")
    start = utils.start_time()
    results = train_dataset(
        vectors = vectors,
        dataframe = dataframe,
        threshold = arguments.fold_change,
        n_examples = arguments.n_examples,
        holdout = arguments.holdout, 
        window = window,
        min_peak_length = arguments.min_peak_length,
        train_all = arguments.train_all)
    utils.end_time(start) 

    results = order_results(results)

    # Plotting performance. 
    print ('Plotting Performance')
    project_folder = utils.project_path()
    reports_folder = os.path.join(project_folder, 'reports')
    # Create filename.
    if arguments.prefix:
        filename = os.path.join(reports_folder, f'{arguments.prefix}_model_performance.pdf')
    else:
        filename = os.path.join(reports_folder, 'model_performance.pdf')
    plot_metrics.plot_pdf(results, filename)

    if not arguments.skip_final:
        start = utils.start_time('Training Final Model')
        models_folder = os.path.join(project_folder, 'models')
        if arguments.prefix:
            filename = os.path.join(models_folder, f'{arguments.prefix}_model.h5')
        else:
            filename = os.path.join(models_folder, 'model.h5')
        train_final(vectors, dataframe, model, arguments.center)
        model.save(filename)
        utils.end_time(start)

    total_time = utils.end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()
