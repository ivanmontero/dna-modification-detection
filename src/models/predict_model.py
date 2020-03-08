from matplotlib import pyplot as plt
from tensorflow import keras
from matplotlib.backends.backend_pdf import PdfPages

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import utils

import numpy as np
import argparse
import json
import time
import tqdm
import pandas as pd

PEAKS_TO_VISUALIZE = 10
WINDOW_AROUND_PEAK = 250

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Train a model on the features and save it.')

    parser.version = 0.1

    parser.add_argument(
        '-i', 
        '--input', 
        required = True,
        help = 'Input extracted feature vectors json file to predict on.')

    parser.add_argument(
        '-m', 
        '--metadata',
        required = True,
        help = 'Metadata for input table.')
    
    parser.add_argument(
        '-mf',
        '--model_file',
        required = True,
        help = 'The file containing the model'
    )

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    parser.add_argument(
        '-c',
        '--center',
        default=False,
        action='store_true',
        help = 'Whether to only predict on the centers'
    )

    return parser.parse_args()

def predict(model, vectors, data):
    data["prediction"] = model.predict(vectors, verbose=1)

# Determines the most important bases in determining the model's confidence in the
# window classification. Goes to each base in the window, and runs the classifier
# with each changed to a different base. Returns the additive probability drop on
# each base.
def feature_importance(model, vectors, data, metadata):
    arguments = metadata["arguments"]
    window_size = arguments["window"]
    columns = arguments["columns"]
    predictions = data["prediction"]
    center = window_size//2
    n = vectors.shape[0]

    start_indexing = [columns.index('top_A'), columns.index('top_T'), columns.index('top_C'), columns.index('top_G')]
    center_indexing = [i + center for i in start_indexing]

    rolled = np.zeros((n*3, vectors.shape[1]))
    current = vectors.copy()
    to_rotate = current[:,center_indexing]
    for j in range(0, 3):
        current[:,center_indexing] = np.roll(to_rotate, j+1, axis=1)
        rolled[j*n:(j+1)*n,:] = current.copy()

    roll_pred = model.predict(rolled, verbose=1)
    
    drops = predictions - np.min(
                np.concatenate(
                    (roll_pred[:n], roll_pred[n:2*n], roll_pred[2*n:]), axis=-1), axis=-1)
        
    drops[drops < 0] = 0
    data["drop"] = drops

def main():
    # Get rid of random tensorflow warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    arguments = setup()
    print ('Reading Data')
    data = pd.read_pickle(arguments.input)
    with open(arguments.metadata) as infile:
        metadata = json.load(infile)

    data = data.dropna()
    vectors = np.array(list(data['vectors'].to_numpy()))
    model = keras.models.load_model(arguments.model_file)
    window = len(metadata['columns'])

    print ('Making Predictions')
    predict(model, vectors, data)

    print ('Running Feature Importance')
    feature_importance(model, vectors, data, metadata)

    project_folder = utils.project_path()
    data_folder = os.path.join(project_folder, 'data')
    processed_folder = os.path.join(data_folder, 'processed')
    reports_folder = os.path.join(project_folder, 'reports')
    if arguments.prefix:
        reports_filename = os.path.join(reports_folder, f'{arguments.prefix}_feature_importance.pdf')
        predictions_filename = os.path.join(processed_folder, f'{arguments.prefix}_predictions.csv')
    else:
        reports_filename = os.path.join(reports_folder, 'feature_importance.pdf')
        predictions_filename = os.path.join(processed_folder, 'predictions.csv')

    print ('Saving Feature Importance Values')
    data['drop'].to_csv(predictions_filename)

    print ('Saving Feature Importance Plots')
    with PdfPages(reports_filename) as pdf: 
        # plot_region(pdf, data, "25L_PLASMID_corrected", 4212, 4270)
        for c in set(data.index.get_level_values('chromosome')):
            c_data = data.loc[c]
            largest_rows = c_data.nlargest(PEAKS_TO_VISUALIZE, 'drop')
            for index, row in largest_rows.iterrows():
                iloc = c_data.index.get_loc(row.name)
                window = c_data.iloc[iloc - WINDOW_AROUND_PEAK: iloc + WINDOW_AROUND_PEAK + 1]
                plot_window(pdf, window)

def plot_region(pdf, data, chromosome, start_position, end_position):
    window = data.loc[chromosome, start_position:end_position, :]
    plot_window(pdf, window)

def plot_window(pdf, window):
    plot_row_num = 3 if "fold_change" in window.columns else 2
    c_pos = window.index.get_level_values('position')
    # TODO: Show the actual IPD values
    plt.subplot(plot_row_num, 1, 1)
    min_top, min_bottom = window["top_ipd"].min(), window["bottom_ipd"].min()
    plt.plot(c_pos, window["top_ipd"] - min_top, label="top_ipd", linewidth=1)
    plt.plot(c_pos, -window["bottom_ipd"] + min_bottom, label="bottom_ipd", linewidth=1)
    plt.legend(
        bbox_to_anchor = (1.05, 1), 
        loc = 'upper left')
    plt.subplot(plot_row_num, 1, 2)
    plt.plot(c_pos, window["prediction"], label="prediction", linewidth=1)
    plt.plot(c_pos, window["drop"], label="drop", linewidth=1)
    plt.legend(
        bbox_to_anchor = (1.05, 1), 
        loc = 'upper left')
    plt.tight_layout()
    if "fold_change" in window.columns:
        plt.subplot(plot_row_num, 1, 3)
        plt.plot(c_pos, window["fold_change"], label="fold_change", linewidth=1)
        plt.legend(
            bbox_to_anchor = (1.05, 1), 
            loc = 'upper left')
    pdf.savefig()
    plt.close()

if __name__ == '__main__':
    main()

