from matplotlib import pyplot as plt
from tensorflow import keras
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import argparse
import json
import time
import tqdm
import pandas as pd

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
        '--model', 
        required = True,
        help = 'Filename for model.')

    parser.add_argument(
        '-of',
        '--original_file',
        required = True,
        help = 'Input preprocessed h5 original data file.'
    )

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    return parser.parse_args()

# Start the timer. 
def start_time(string = None):
    if string:
        print (string)
    return time.time()

# End the timer. 
def end_time(start, stop = False):
    seconds = time.time() - start
    hours, seconds =  seconds // 3600, seconds % 3600
    minutes, seconds = seconds // 60, seconds % 60
    string = f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}'
    if stop:
        return string
    print (f'{string} elapsed.')

# Return path to project level. 
def project_path():
    script_path = os.path.abspath(__file__)
    script_folder = os.path.dirname(script_path)
    src_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(src_folder)
    
    return project_folder

# Loads the extracted features file, and relevant data.
def load(feature_file, original_file):
    with open(feature_file) as f:
        contents = json.load(f)
        data = contents['vectors']
        positions = contents['positions']
        chromosomes = contents['chromosomes']
        feature_args = contents['arguments']

    original = pd.read_hdf(original_file).loc[list(set(chromosomes))]

    return np.array(data), np.array(positions), chromosomes, feature_args, original

# Determines the most important bases in determining the model's confidence in the
# window classification. Goes to each base in the window, and runs the classifier
# with each changed to a different base. Returns the additive probability drop on
# each base.
def feature_importance(vector, prediction, model, feature_args):
    features = np.zeros(feature_args['window'])
    alternate = []

    a_start = feature_args['columns'].index('top_A') * feature_args['window']
    t_start = feature_args['columns'].index('top_T') * feature_args['window']
    c_start = feature_args['columns'].index('top_C') * feature_args['window']
    g_start = feature_args['columns'].index('top_G') * feature_args['window']
    for i in range(feature_args['window']): # Assumes a window size, and that the bases appear at the beginning
        a = i + a_start
        t = i + t_start
        c = i + c_start
        g = i + g_start

        current = vector.copy()
        embedding = vector[[a,t,c,g]]
        new = 0
        for j in range(1,4):
            current[[a,t,c,g]] = np.roll(embedding, j)
            alternate.append(current.copy())

    alternate = np.array(alternate)
    new_predictions = model.predict(alternate, batch_size = len(alternate))
    for i in range(feature_args['window']):
        features[i] = max(0, prediction - min(new_predictions[i*3:i*3+3]))

    return features

def main():
    arguments = setup()
    model = keras.models.load_model(arguments.model)
    data, positions, chromosomes, feature_args, original = load(arguments.input, arguments.original_file)

    original["drop"] = 0

    # index = [(c, p) for c in set(chromosomes) for p in range(positions[np.array(chromosomes) == c].max()+1)]

    # importance = pd.DataFrame(index=pd.MultiIndex.from_tuples(index, names=['chromosome', 'position']), columns=["drop"], dtype=float)
    # print(importance)
    # print(importance.loc["LtaP_01", 0])


    predictions = model.predict(data)
    # plasmid = np.zeros(positions.max() + 1) # Assumes only one chromosome...
    # TODO: Open merged data, and map between that and the predictions

    for i in tqdm.tqdm(range(len(data))):
        window = positions[i]
        vector = data[i] 
        chromosome = chromosomes[i]
        prediction = predictions[i]
        drops = feature_importance(vector, prediction, model, feature_args)
        for i, p in enumerate(window):
            original.loc[(chromosome, p), 'drop'] += drops[i]

    project_folder = project_path()
    data_folder = os.path.join(project_folder, 'data')
    processed_folder = os.path.join(data_folder, 'processed')
    reports_folder = os.path.join(project_folder, 'reports')
    if arguments.prefix:
        reports_filename = os.path.join(reports_folder, f'{arguments.prefix}_feature_importance.pdf')
        predictions_filename = os.path.join(processed_folder, f'{arguments.prefix}_predictions.csv')
    else:
        reports_filename = os.path.join(reports_folder, 'feature_importance.pdf')
        predictions_filename = os.path.join(processed_folder, 'predictions.csv')

    original['drop'].to_csv(predictions_filename)

    with PdfPages(reports_filename) as pdf: 
        for c in original.index.get_level_values('chromosome'):
            pass
        # # TODO: Visualize with ipd values, fold change values (if applicable), etc.
        # plt.figure(figsize = (8,4), dpi = 100)
        # plt.plot(plasmid[4000:4500])
        # plt.savefig('test.png')


if __name__ == '__main__':
    main()

