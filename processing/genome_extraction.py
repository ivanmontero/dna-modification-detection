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
    ipd = arguments.ipd
    fold_change = arguments.fold_change

    true_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] > fold_change)]
    false_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] > fold_change)]
    true_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] < fold_change)]
    false_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] < fold_change)]

    print ('Sampling data.')
    true_positive = data_extraction.sample(true_positive, arguments.examples)
    false_positive = data_extraction.sample(false_positive, arguments.examples)
    true_negative = data_extraction.sample(true_negative, arguments.examples)
    false_negative = data_extraction.sample(false_negative, arguments.examples)

    print ('Extracting windows.')
    start = time.time()
    true_positive_features, true_positive_positions = data_extraction.windows(true_positive, data, arguments.window, arguments.columns)
    false_negative_features, false_negative_positions = data_extraction.windows(false_negative, data, arguments.window, arguments.columns)
    true_negative_features, true_negative_positions = data_extraction.windows(true_negative, data, arguments.window, arguments.columns)
    false_positive_features, false_positive_positions = data_extraction.windows(false_positive, data, arguments.window, arguments.columns)
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Creating labels.')
    features = np.vstack([true_positive_features,
                          false_negative_features, 
                          true_negative_features, 
                          false_positive_features])

    positions = np.vstack([true_positive_positions,
                           false_negative_positions,
                           true_negative_positions,
                           false_positive_positions])
    
    labels = np.hstack([np.ones(len(true_positive_features)), 
                        np.ones(len(false_negative_features)),
                        np.zeros(len(true_negative_features)),
                        np.zeros(len(false_positive_features))])

    if arguments.save_classes:
        classes = ['tp'] * len(true_positive_features) + \
                  ['fn'] * len(false_negative_features) + \
                  ['tn'] * len(true_negative_features) + \
                  ['fp'] * len(false_positive_features)

        data = {'vectors': features.tolist(), 
                'positions': positions.tolist(), 
                'classes': classes} 

        if arguments.output:
            filename = f'{arguments.output}_classes.json'
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)
        else:
            directory = os.path.dirname(arguments.infile)
            filename = os.path.join(directory, 'classes.json')
            with open(filename, 'w') as outfile:
                json.dump(data, outfile)

    index = np.random.permutation(len(features))
    features = features[index]
    positions = positions[index]
    labels = labels[index]

    print ('Writing output.')
    data = {'vectors': features.tolist(), 
            'positions': positions.tolist(), 
            'labels': labels.tolist()}

    if arguments.output:
        with open(arguments.output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        directory = os.path.dirname(arguments.infile)
        filename = os.path.join(directory, 'data.json')
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    elapsed = time.time() - total_start
    print (f'{elapsed:.0f} seconds elapsed in total.')

if __name__ == '__main__':
    main()






