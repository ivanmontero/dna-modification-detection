import argparse

# Imports necessary modules
def import_modules():
    print ('Importing Modules')
    # Make the local imports global
        
    global progress_bars
    global tqdm
    global plot_metrics
    global datetime
    global metrics
    global utils
    global copy 
    global json
    global sys
    global torch
    global nn
    global F
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
    from sklearn import metrics
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pandas as pd
    import numpy as np
    import datetime
    import json
    import copy
    from tqdm import tqdm
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
        '--dataframe', 
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

    parser.add_argument(
        '-od', 
        '--outdir', 
        default = None,
        help = 'Output directory.')

    # Whether the final model is trained on only As and Ts
    parser.add_argument(
        '--only_t',
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

    # Minimum valid ChIP peak size. 
    parser.add_argument(
        '--min-peak-length',
        default = 20,
        type = int,
        help = argparse.SUPPRESS)

    # Turn progress bars off. 
    parser.add_argument(
        '--progress-off',
        default = False,
        action = 'store_true',
        help = argparse.SUPPRESS)

    # Save a description during hyperparameter search.
    parser.add_argument(
        '--description',
        default = False,
        help = argparse.SUPPRESS)

    # Number of neurons per layer. 
    parser.add_argument(
        '--layers',
        nargs = '+',
        type = int,
        default = [300, 150, 50],
        help = argparse.SUPPRESS)

    # Dropout rate. 
    parser.add_argument(
        '--dropout',
        default = 0.5,
        type = float,
        help = argparse.SUPPRESS)

    return parser.parse_args()

# Determines how many unique peaks that exist with at least min_peak_length length. A "peak"
# is considered a contiguous block of positions that have a fold change value above threshold.
# Each base is labeled in the given dataframe with the fold change peak it's associated with,
# and zero otherwise.
def label_peaks(dataframe, threshold, min_peak_length):
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

# Computes quantities relating to peaks 
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

# The simple threshold by simply looking at the current IPD vakue and comparing it to 
def simple_scores(dataframe):
    top_ipd = dataframe['top_ipd'].to_numpy()
    bottom_ipd = dataframe['bottom_ipd'].to_numpy()
    scores = np.maximum(top_ipd, bottom_ipd)

    return scores

# The complex scores are computed by considering the sum of the max IPD values at the
# 0, 2, and 6 position. A visualization of the IPD values taken into consideration are
# below:
#      Top: +6              +2      +0    
# Position:  0   1   2   3   4   5   6   7   8   9   10  11  12
#   Bottom:                         +0      +2               +6
def complex_scores(dataframe):
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
    minimum = np.min(top_sum)
    scores = np.full(len(dataframe), minimum)
    
    scores[:-6] = bottom_sum
    scores[6:] = np.maximum(scores[6:], top_sum)

    return scores

# Given scores, returns the receiver operator characteristic, precision recall, and peak
# curves
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

# From a (m, d) numpy array, returns n_examples rows sampled randomly w/o replacement.
def sample(vectors, n_examples):
    if len(vectors) <= n_examples:
        return vectors
    else:
        index = np.arange(len(vectors))
        selection = np.random.choice(index, n_examples, replace = False)
        return vectors[selection]

# We will define our model as a multi-layer densely connected neural network
# with dropout between the layers.
# TODO: Parameterize this
def create_model(input_dim, hidden_dims, dropout=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers).to(device)

def fit(model, vectors, labels, epochs=10, batch_size=32, valid_vectors=None, valid_labels=None):  # TODO: Return history
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = torch.optim.Adam(model.parameters())

    n = vectors.shape[0]
    dataset = torch.utils.data.TensorDataset(torch.tensor(vectors), torch.tensor(labels))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {stat: [] for stat in ["loss", "acc", "val_loss", "val_acc"]}

    for epoch in tqdm(range(epochs), desc="Training", position=0):
        correct = 0
        cum_loss = 0.0

        # Train a single epoch
        for bx, by in tqdm(dataloader, desc="Train Batch", position=1, leave=False):
            bx = bx.to(device).float()
            by = by.to(device).float()

            output = model(bx)
            loss = F.binary_cross_entropy(output.reshape(-1), by)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += ((output.reshape(-1) > 0.5).int() == by).float().sum().cpu().item()
            cum_loss += loss.cpu().item()

        history["loss"].append(cum_loss / n)
        history["acc"].append(correct / n)

        if valid_vectors is not None:
            val_loss, val_acc = validate(model, valid_vectors, valid_labels, batch_size)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

    return history

def validate(model, vectors, labels, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = vectors.shape[0]
    dataset = torch.utils.data.TensorDataset(torch.tensor(vectors), torch.tensor(labels))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    correct = 0
    cum_loss = 0.0
    with torch.no_grad():
        for bx, by in tqdm(dataloader, desc="Validation Batch", position=1, leave=False):
            bx = bx.to(device).float()
            by = by.to(device).float()

            output = model(bx)
            loss = F.binary_cross_entropy(output.reshape(-1), by)

            cum_loss += loss.cpu().item()
            correct += ((output.reshape(-1) > 0.5).int() == by).float().sum().cpu().item()
    
    return cum_loss / n, correct / n

def predict(model, vectors, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = vectors.shape[0]
    # dataset = torch.utils.data.TensorDataset(torch.tensor(vectors))
    dataloader = torch.utils.data.DataLoader(torch.tensor(vectors), batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for bx in dataloader:
            bx = bx.to(device).float()

            output = model(bx)
            predictions.append(output)
    
    return torch.cat(predictions, dim=0).cpu().numpy()

# Determines the most important bases in determining the model's confidence in the
# window classification. Goes to each base in the window, and runs the classifier
# with each changed to a different base. Returns the additive probability drop on
# each base.
def feature_importance(model, vectors, scores, metadata, progress_off, batch_size = 32):
    arguments = metadata["arguments"]
    window_size = arguments["window"]
    columns = arguments["columns"]
    center = window_size // 2
    length = len(vectors)

    # Find out where each base starts in the vector.
    start_indexing = [
        columns.index('top_A'),
        columns.index('top_T'),
        columns.index('top_C'),
        columns.index('top_G')]

    # Determine where center base is. 
    center_indexing = np.array(start_indexing) + center

    # Empty array with 3 times as many rows. 
    alternate_vectors = np.zeros((len(vectors) * 3, vectors.shape[1]))

    # Only change the center base of the vectors.
    current = vectors.copy()
    to_rotate = current[:,center_indexing]

    # Create vectors with all three other bases in the center.
    for i in range(3):
        current[:,center_indexing] = np.roll(to_rotate, i+1, axis = 1)
        alternate_vectors[i*length:(i+1)*length,:] = current.copy()

    new_predictions = validate_network(
        model,
        alternate_vectors,
        progress_off
    )

    # Get the aggregate new prediction and find the delta. 
    new_predictions = new_predictions.reshape(3,-1)
    drops = scores - np.min(new_predictions, axis = 0)

    return drops

# Trains a odel on the entire passed in dataframe.
def train_final(vectors, dataframe, n_examples, metadata, only_t, train_all, progress_off, layers, dropout):
    labels = dataframe['label'].to_numpy()

    # Only keep Ts.
    if only_t:
        condition = ((dataframe['top_A'] == 1) | (dataframe['top_T'] == 1))
        vectors = vectors[condition]
        labels = labels[condition]

    # Create the dataset. 
    vectors, labels = create_training_fold(
        vectors = vectors,
        labels = labels,
        n_examples = n_examples,
        train_all = train_all)

    window = len(metadata['columns'])
    model = create_model(window, layers, dropout=dropout)

    history = fit(model, vectors, labels)

    return model

# Prepares a training fold dataset.
def create_training_fold(vectors, labels, n_examples, train_all = False, batch_size = 32):
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

    # Compute the number of batches.
    length = int(np.ceil(length/batch_size))

    return vectors, labels

# Prepares a validation fold dataset.
def create_validation_fold(vectors, labels):
    return vectors, labels

# Begin training the neural network, with optional validation data, and model saving.
def train_network(model, vectors, labels, progress_off, validation_split = 0.1, epochs=10):
    n_validation = int(vectors.shape[0]*validation_split)
    # Save 10% of the training to see how validation history looks.
    idx = np.random.permutation(vectors.shape[0])
    valid_idx, train_idx = idx[:n_validation], idx[n_validation:]

    history = fit(
        model,
        vectors[train_idx],
        labels[train_idx],
        valid_vectors=vectors[valid_idx],
        valid_labels=labels[valid_idx],
        epochs=epochs)

    training_history = {'x': range(1, epochs + 1), 'y': history['acc'], 'area': None}
    validation_history = {'x': range(1, epochs + 1), 'y': history['val_acc'], 'area': None}
    
    return training_history, validation_history

# Predicts on a provided valiation dataset.
def validate_network(model, vectors, progress_off):
    return predict(model, vectors).reshape(-1)

# Trains on a fold of the dataset.
def train_fold(model, vectors, dataframe, n_examples, train_all, progress_off):
    labels = dataframe['label'].to_numpy()
    vectors, labels = create_training_fold(
        vectors = vectors,
        labels = labels,
        n_examples = n_examples,
        train_all = train_all)

    training_history, validation_history = train_network(model, vectors, labels, progress_off)

    return training_history, validation_history

# Validates on a fold of the dataset.
def validate_fold(model, vectors, dataframe, progress_off):
    labels = dataframe['label'].to_numpy()
    vectors, _ = create_validation_fold(
        vectors = vectors,
        labels = labels)

    scores = validate_network(model, vectors, progress_off)

    return scores

# Performs a k-fold experiment on the dataset, where k is equal to the number of chromosomes. On each fold, this method
# obtains metrics by training on all examples from all chromosomes except a holdout one, then obtains metrics on the one
# held out chromosome.
def train_dataset(vectors, dataframe, threshold, n_examples, holdout, metadata, min_peak_length, train_all, progress_off, layers, dropout):
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

    window = len(metadata['columns'])
    index = dataframe.index.get_level_values('chromosome')
    for holdout in chromosomes:
        # Training fold.
        training_vectors = vectors[index != holdout]
        training_dataframe = dataframe.loc[index != holdout]

        # Validation fold.
        validation_vectors = vectors[index == holdout]
        validation_dataframe = dataframe.loc[index == holdout]

        # Complex threshold baseline.
        scores = complex_scores(validation_dataframe)

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
        scores = simple_scores(validation_dataframe)

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

        model = create_model(window, layers, dropout)
        training_history, validation_history = train_fold(
            model = model,
            vectors = training_vectors,
            dataframe = training_dataframe,
            n_examples = n_examples,
            train_all = train_all,
            progress_off = progress_off)

        # Store history.
        results['history'].append(training_history)
        results['history'].append(validation_history)

        scores = validate_fold(
            model = model,
            vectors = validation_vectors,
            dataframe = validation_dataframe,
            progress_off = progress_off)
        # model.reset_states()

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

        # Delta T baseline.
        scores = feature_importance(
            model = model,
            vectors = validation_vectors,
            scores = scores,
            metadata = metadata,
            progress_off = progress_off)

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

        # Only keep the T's. 
        condition = ((training_dataframe['top_A'] == 1) | (training_dataframe['top_T'] == 1))
        training_vectors = training_vectors[condition]
        training_dataframe = training_dataframe.loc[condition]

        model = create_model(window, layers, dropout)
        training_history, validation_history = train_fold(
            model = model,
            vectors = training_vectors,
            dataframe = training_dataframe,
            n_examples = n_examples,
            train_all = train_all,
            progress_off = progress_off)

        # Store history.
        results['history'].append(training_history)
        results['history'].append(validation_history)

        scores = validate_fold(
            model = model,
            vectors = validation_vectors,
            dataframe = validation_dataframe,
            progress_off = progress_off)

        # Store metrics.
        roc, pr, peak = get_metrics(validation_dataframe, scores)
        results['receiver_operator'].append(roc)
        results['precision_recall'].append(pr)
        results['peak_curve'].append(peak)

    return vectors, dataframe, results

def order_results(results):
    reorder = {
        'history': [],
        'receiver_operator': [],
        'precision_recall': [],
        'peak_curve': [],
    }

    # Labels for things needing training.
    history_labels = [
        'All Bases (Training)',
        'All Bases (Validation)',
        'Only T (Training)',
        'Only T (Validation)']

    # Labels for everything evaluated.
    other_labels = [
        'Complex Threshold',
        'Simple Threshold',
        'All Bases',
        'Delta T',
        'Only T'
    ]

    # The number of curves for each metric.
    total_curves = {
        'history': len(history_labels),
        'receiver_operator': len(other_labels),
        'precision_recall': len(other_labels),
        'peak_curve': len(other_labels)
    }

    # Add the expected structure for each curve.
    for item in reorder:
        curve = {
            'x': [],
            'y': [],
            'area': [],
            'label': None,
        }

        for i in range(total_curves[item]):
            reorder[item].append(copy.deepcopy(curve))

    # Add the labels for the history metric.
    for i in range(len(history_labels)):
        reorder['history'][i]['label'] = history_labels[i]

    # Add the labels for everything else.
    for metric in ['receiver_operator', 'precision_recall', 'peak_curve']:
        for i in range(len(other_labels)):
            reorder[metric][i]['label'] = other_labels[i]

    # Reorder the results so folds are sequential instead of metrics.
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
    dataframe = pd.read_hdf(arguments.dataframe)
    with open(arguments.metadata) as infile:
        metadata = json.load(infile)
    vectors = np.memmap(arguments.input, dtype = 'float32', mode = 'r', shape = (metadata['rows'], len(metadata['columns'])))
    utils.end_time(start)

    print("Model architecture:")
    print(create_model(vectors.shape[1], [300, 150, 50]))

    # Training model.
    print("Training Model")
    start = utils.start_time()
    vectors, dataframe, results = train_dataset(
        vectors = vectors,
        dataframe = dataframe,
        threshold = arguments.fold_change,
        n_examples = arguments.n_examples,
        holdout = arguments.holdout, 
        metadata = metadata,
        min_peak_length = arguments.min_peak_length,
        train_all = arguments.train_all,
        progress_off = arguments.progress_off,
        layers = arguments.layers,
        dropout = arguments.dropout)
    utils.end_time(start) 

    results = order_results(results)

    # Plotting performance. 
    print ('Plotting Performance')
    project_folder = utils.project_path(arguments.outdir)
    reports_folder = os.path.join(project_folder, 'reports')
    training_folder = os.path.join(reports_folder, 'training')
    os.makedirs(reports_folder, exist_ok = True)
    os.makedirs(training_folder, exist_ok = True)
    # Create filename.
    if arguments.prefix:
        filename = os.path.join(training_folder, f'{arguments.prefix}_model_performance.pdf')
    else:
        filename = os.path.join(training_folder, 'model_performance.pdf')

    metrics = plot_metrics.plot_pdf(results, filename, arguments.description)
    if arguments.description:
        model = create_model(len(metadata['columns']), arguments.layers, arguments.dropout)
        parameters = len(model.parameters())

        date = datetime.date.today()
        time = datetime.datetime.now().time()

        string = ''
        for item in metrics:
            string += f'\t{item}'

        line = f'{date}\t{time}\t{arguments.description}\t{vars(arguments)}{string}\t{parameters}\t'

        hyperparameter_folder = os.path.join(reports_folder, 'hyperparameter')
        metrics_file = os.path.join(hyperparameter_folder, 'metrics.txt')
        with open(metrics_file, 'a+') as outfile:
            outfile.write(line)

    if not arguments.skip_final:
        start = utils.start_time('Training Final Model')
        models_folder = os.path.join(project_folder, 'models')
        os.makedirs(models_folder, exist_ok = True)
        if arguments.prefix:
            filename = os.path.join(models_folder, f'{arguments.prefix}_model.h5')
        else:
            filename = os.path.join(models_folder, 'model.h5')

        model = train_final(
            vectors = vectors, 
            dataframe = dataframe, 
            n_examples = arguments.n_examples, 
            metadata = metadata, 
            only_t = arguments.only_t,
            train_all = arguments.train_all,
            progress_off = arguments.progress_off,
            layers = arguments.layers,
            dropout = arguments.dropout)

        # model.save(filename)
        torch.save(model, filename)
        utils.end_time(start)

    total_time = utils.end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()
