from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

# Import local helper functions.
import os
import sys
current_path = os.path.dirname(__file__)
utils_path = os.path.join(current_path, '..', 'utils')
sys.path.append(utils_path)
import utils

# Import local.
import progress_bars

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Make predictions for Base J locations.',
        formatter_class = argparse.RawTextHelpFormatter)

    parser.version = 0.1

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
        '--model',
        required = True,
        help = 'The file containing the model.')

    parser.add_argument(
        '-c',
        '--chromosome',
        required = True,
        help = 'Which chromosome to visualize.')

    parser.add_argument(
        '-s',
        '--start',
        type = int,
        required = True,
        help = 'Start coordinate.')

    parser.add_argument(
        '-e',
        '--end',
        type = int,
        required = True,
        help = 'End coordinate.')

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    # Turn progress bars off. 
    parser.add_argument(
        '--progress-off',
        default = False,
        action = 'store_true',
        help = argparse.SUPPRESS)

    parser.add_argument(
        '-od', 
        '--outdir', 
        default = None,
        help = 'Output directory.')

    return parser.parse_args()

def predict(model, vectors, progress_off, batch_size = 32):
    # Convert to tensorflow dataset.
    dataset = tf.data.Dataset.from_tensor_slices(vectors)
    dataset = dataset.batch(batch_size)

    # Compute the number of batches.
    length = int(np.ceil(len(vectors)/batch_size))

    # Create our custom TQDM progress bar for validation.
    if progress_off:
        callback = progress_bars.no_progress()
    else:
        callback = progress_bars.predict_progress(length)
    
    scores = model.predict(
        dataset,
        callbacks = [callback])

    return scores.reshape(-1)

def plot(dataframe, filename):

    dataframe.fillna(0, inplace = True)

    if 'fold_change' in dataframe.columns:
        num_plots = 3
        index = 1

        fold_change = dataframe['fold_change'].to_numpy()
        plt.figure(figsize = (8,9), dpi = 100)
        plt.subplot(num_plots, 1, 1)
        plt.plot(fold_change)
        plt.ylim([0, np.max(fold_change) * 1.1])
        plt.ylabel('Fold Change')
        plt.title('Fold Change')
    else:
        num_plots = 2
        index = 0
        plt.figure(figsize = (8,6), dpi = 100)

    top_ipd = dataframe['top_ipd'].to_numpy()
    bottom_ipd = dataframe['bottom_ipd'].to_numpy()

    min_top = np.min(top_ipd)
    min_bottom = np.min(bottom_ipd)

    # Renormalize the IPD data so there are no negative numbers. 
    top_ipd = top_ipd - min_top
    bottom_ipd = -bottom_ipd + min_bottom
    position =  np.arange(len(top_ipd))

    index += 1
    plt.subplot(num_plots, 1, index)
    plt.bar(position, top_ipd)
    plt.bar(position, bottom_ipd)
    plt.ylabel('IPD')
    plt.title('IPD Ratios')

    scores = dataframe['score'].to_numpy()
    position = np.arange(len(scores))

    index += 1
    plt.subplot(num_plots, 1, index)
    plt.bar(position, scores)
    plt.ylim([0,1])
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('Predictions')
    plt.tight_layout()
    plt.savefig(filename)

def main():
    # Get argparse arguments. 
    arguments = setup()

    total_start = utils.start_time()
    # Get rid of random tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Reading data. 
    start = utils.start_time('Reading Data')
    dataframe = pd.read_hdf(arguments.dataframe)
    vectors = np.load(arguments.input)
    utils.end_time(start) 

    # Only use requested section.
    index = dataframe.index.get_level_values('chromosome')
    position = dataframe.index.get_level_values('position')

    condition = ((index == arguments.chromosome) & (position > arguments.start) & (position < arguments.end))
    vectors = vectors[condition]
    dataframe = dataframe[condition]

    # Only predict on Ts. 
    condition = ((dataframe['top_A'] == 1) | (dataframe['top_T'] == 1))
    vectors = vectors[condition]

    model = keras.models.load_model(arguments.model)
    scores = predict(
        model = model, 
        vectors = vectors,
        progress_off = arguments.progress_off)
    dataframe.loc[condition, 'score'] = scores

    project_folder = utils.project_path(arguments.outdir)
    reports_folder = os.path.join(project_folder, 'reports')
    predict_folder = os.path.join(reports_folder, 'predict')
    os.makedirs(predict_folder, exist_ok=True)
    if arguments.prefix:
        filename = os.path.join(predict_folder, f'{arguments.prefix}_predict.png')
    else:
        filename = os.path.join(predict_folder, 'predict.png')
    plot(dataframe, filename)


if __name__ == '__main__':
    main()

