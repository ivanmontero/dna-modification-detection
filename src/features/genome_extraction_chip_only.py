import data_extraction
import pandas as pd
import numpy as np
import json
import os

# Genome data extraction from solely the ChIP information.
# TODO: Populate JSON file with the arguments
def main():
    arguments = data_extraction.setup()

    start = data_extraction.start_time('Reading data.')
    data = pd.read_hdf(arguments.infile, columns = ['fold_change'] + arguments.columns)
    data_extraction.end_time(start)

    print ('Filtering data.')
    fold_change = arguments.fold_change
    positive = data[data['fold_change'] > fold_change]
    negative = data[data['fold_change'] < fold_change]

    print ('Sampling data.')
    positive = data_extraction.sample(positive, arguments.examples)
    negative = data_extraction.sample(negative, arguments.examples)

    start = data_extraction.start_time('Extracting windows.')
    data = pd.read_hdf(arguments.infile)
    positive_features, positive_positions = data_extraction.windows(positive, data, arguments.window, arguments.columns)
    negative_features, negative_positions = data_extraction.windows(negative, data, arguments.window, arguments.columns)
    data_extraction.end_time(start)

    print ('Creating labels.')
    features = np.vstack([positive_features,negative_features])
    positions = np.vstack([positive_positions,negative_positions])
    labels = np.hstack([np.ones(len(positive_features)), np.zeros(len(negative_features))])

    index = np.random.permutation(len(features))
    features = features[index]
    positions = positions[index]
    labels = labels[index]

    column_labels = []
    for column in arguments.columns:
        column_labels += [column] * arguments.window

    print ('Writing output.')
    data = {'columns': column_labels,
            'vectors': features.tolist(), 
            'positions': positions.tolist(), 
            'labels': labels.tolist()}

    project_folder = data_extraction.project_path()
    data_folder = os.path.join(project_folder, 'data')
    processed_folder = os.path.join(data_folder, 'processed')
    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_data.json')
    else:
        filename = os.path.join(processed_folder, 'data.json')

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

if __name__ == '__main__':
    main()
