from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import json
import torch

# Import local helper functions.
import sys
import os
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
        '--metadata',
        required = True,
        help = 'Metadata for input table.')
    
    parser.add_argument(
        '-md',
        '--model',
        required = True,
        help = 'The file containing the model.')

    parser.add_argument(
        '-l',
        '--location',
        required = False,
        nargs = '+',
        help = 'Which locations to visualize.\n\
                In the format chromosome:start:end.\n \
                Example: 25L_PLASMID_corrected:4200:4300')

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

    return parser.parse_args()

def predict(model, vectors, progress_off, batch_size=32):
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

    j_count = []
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

            # Plot data.
            plot(current_dataframe, chromosome, start, end, pdf)

            # Count number of Js called. 
            j_count.append(np.sum(scores > 0.9))

    return j_count

def main():
    # Get argparse arguments. 
    arguments = setup()

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

    # Load model. 
    # model = keras.models.load_model(arguments.model)
    model = torch.load(arguments.model)

    # Determine filename.
    project_folder = utils.project_path(arguments.outdir)
    reports_folder = os.path.join(project_folder, 'reports')
    predict_folder = os.path.join(reports_folder, 'predict')
    os.makedirs(predict_folder, exist_ok=True)
    if arguments.prefix:
        filename = os.path.join(predict_folder, f'{arguments.prefix}_predict.pdf')
    else:
        filename = os.path.join(predict_folder, 'predict.pdf')

    j_count = plot_locations(
        model = model,
        vectors = vectors,
        dataframe = dataframe,
        locations = arguments.location,
        filename = filename, 
        progress_off = arguments.progress_off)

    if arguments.description:
        line = f'{j_count[0]}\t{j_count[1]}\n'

        hyperparameter_folder = os.path.join(reports_folder, 'hyperparameter')
        metrics_file = os.path.join(hyperparameter_folder, 'metrics.txt')
        with open(metrics_file, 'a+') as outfile:
            outfile.write(line)

if __name__ == '__main__':
    main()

