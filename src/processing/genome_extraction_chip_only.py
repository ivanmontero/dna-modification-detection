import data_extraction
import pandas as pd
import numpy as np
import json
import time
import os

def main():
    total_start = time.time()
    arguments = data_extraction.setup()

    print ('Reading data.')
    start = time.time()
    data = pd.read_hdf(arguments.infile)
    data = data[['fold_change'] + arguments.columns]
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Filtering data.')
    fold_change = arguments.fold_change
    positive = data[data['fold_change'] > fold_change]
    negative = data[data['fold_change'] < fold_change]

    print ('Sampling data.')
    positive = data_extraction.sample(positive, arguments.examples)
    negative = data_extraction.sample(negative, arguments.examples)

    print ('Extracting windows.')
    start = time.time()
    positive_features, positive_positions = data_extraction.windows(positive, data, arguments.window, arguments.columns)
    negative_features, negative_positions = data_extraction.windows(negative, data, arguments.window, arguments.columns)
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

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

    directory = os.path.dirname(arguments.infile)
    if arguments.output:
        filename = os.path.join(directory, f'{arguments.output}.json')
    else:
        filename = os.path.join(directory, 'data.json')

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

    elapsed = time.time() - total_start
    print (f'{elapsed:.0f} seconds elapsed in total.')

if __name__ == '__main__':
    main()
