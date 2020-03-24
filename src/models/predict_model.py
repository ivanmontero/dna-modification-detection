from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

pd.set_option('mode.chained_assignment', 'raise')

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
        '-l',
        '--location',
        required = True,
        nargs = '+',
        help = 'Which locations to visualize.\n\
                In the format chromosome:start:end.\n \
                Example: 25L_PLASMID_corrected:4200:4300')

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

def get_sequence(dataframe):
    top_encoding = dataframe[[
        'top_A',
        'top_T',
        'top_C',
        'top_G']].to_numpy(dtype = bool).T
    top_sequence = np.array(['N'] * len(top_encoding[0]))

    bases = ['A', 'T', 'C', 'G']
    for i in range(len(bases)):
        top_sequence[top_encoding[i]] = bases[i]

    bottom_encoding = dataframe[[
        'bottom_A',
        'bottom_T',
        'bottom_C',
        'bottom_G']].to_numpy(dtype = bool).T
    bottom_sequence = np.array(['N'] * len(bottom_encoding[0]))

    for i in range(len(bases)):
        bottom_sequence[bottom_encoding[i]] = bases[i]

    return top_sequence, bottom_sequence

def plot(dataframe, chromosome, start, end, pdf):

    dataframe.fillna(0, inplace = True)

    if 'fold_change' in dataframe.columns:
        extra = True
        fold_change = dataframe['fold_change'].to_numpy()

        # Make the figure template. 
        figure = plt.figure(figsize = (8,10), dpi = 100)
        grid = figure.add_gridspec(6, 1)

        plt.gcf().add_subplot(grid[0:2, :])
        plt.plot(fold_change)
        plt.ylim([0, np.max(fold_change) * 1.1])
        plt.ylabel('Fold Change')
        plt.title(f'{chromosome} - {start}:{end}')
    else:
        # Make the figure template. 
        figure = plt.figure(figsize = (8,8), dpi = 100)
        grid = figure.add_gridspec(4, 1)
        extra = False

    top_ipd = dataframe['top_ipd'].to_numpy()
    bottom_ipd = dataframe['bottom_ipd'].to_numpy()

    min_top = np.min(top_ipd)
    min_bottom = np.min(bottom_ipd)

    # Renormalize the IPD data so there are no negative numbers. 
    top_ipd = top_ipd - min_top
    bottom_ipd = -bottom_ipd + min_bottom
    position =  np.arange(len(top_ipd))

    if extra:
        grid_start = 2
        grid_end = 4
    else:
        grid_start = 0
        grid_end = 2

    plt.gcf().add_subplot(grid[grid_start:grid_end, :])
    plt.bar(position, top_ipd)
    plt.bar(position, bottom_ipd)
    plt.ylabel('IPD')
    if not extra:
        plt.title(f'{chromosome} - {start}:{end}')

    top_sequence, bottom_sequence = get_sequence(dataframe)

    if extra:
        grid_position = 4
    else:
        grid_position = 2

    plt.gcf().add_subplot(grid[grid_position, :])
    # Plot top sequence. 
    for i in range(len(top_sequence)):
        letter = top_sequence[i]
        if letter == 'T':
            color = 'red'
        else:
            color = 'black'

        plt.text(
            x = i,
            y = 0.5,
            s = letter, 
            color = color,
            fontname = 'monospace',
            fontsize = 8,
            ha = 'left')

    for i in range(len(bottom_sequence)):
        letter = bottom_sequence[i]
        if letter == 'T':
            color = 'red'
        else:
            color = 'black'

        plt.text(
            x = i,
            y = -0.5,
            s = letter,
            color = color,
            fontname = 'monospace',
            fontsize = 8,
            ha = 'left')

    lower = 0 - (len(top_sequence) * 0.05)
    upper = len(top_sequence) * 1.05
    plt.xlim([lower, upper])
    plt.ylim([-2, 2])
    plt.xticks([],[])
    plt.yticks([],[])

    scores = dataframe['score'].to_numpy()
    position = np.arange(len(scores))

    if extra:
        grid_position = 5
    else:
        grid_position = 3

    plt.gcf().add_subplot(grid[grid_position, :])
    plt.bar(position, scores)
    plt.ylim([0,1])
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def plot_locations(model, vectors, dataframe, locations, filename, progress_off):
    # Only use requested section.
    index = dataframe.index.get_level_values('chromosome')
    position = dataframe.index.get_level_values('position')

    with PdfPages(filename) as pdf:
        for location in locations:
            split = location.split(':')
            chromosome = split[0]
            start = int(split[1])
            end = int(split[2])

            condition = ((index == chromosome) & (position >= start) & (position <= end))
            current_vectors = vectors[condition]
            current_dataframe = dataframe.loc[condition]

            # Only predict on Ts. 
            condition = ((current_dataframe['top_A'] == 1) | (current_dataframe['top_T'] == 1))
            current_vectors = current_vectors[condition]
            
            scores = predict(
                model = model, 
                vectors = current_vectors,
                progress_off = progress_off)

            # Assign scores.
            current_scores = np.zeros(len(current_dataframe))
            current_scores[condition] = scores
            current_dataframe = current_dataframe.assign(score = current_scores)

            plot(current_dataframe, chromosome, start, end, pdf)

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

    # Load model. 
    model = keras.models.load_model(arguments.model)

    # Determine filename.
    project_folder = utils.project_path()
    reports_folder = os.path.join(project_folder, 'reports')
    predict_folder = os.path.join(reports_folder, 'predict')
    if arguments.prefix:
        filename = os.path.join(predict_folder, f'{arguments.prefix}_predict.pdf')
    else:
        filename = os.path.join(predict_folder, 'predict.pdf')

    # Plot locations.
    plot_locations(
        model = model,
        vectors = vectors,
        dataframe = dataframe,
        locations = arguments.location,
        filename = filename, 
        progress_off = arguments.progress_off)

if __name__ == '__main__':
    main()

