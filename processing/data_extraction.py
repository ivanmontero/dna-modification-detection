import pandas as pd
import numpy as np 
import argparse
import time
import os

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Create a numpy objects with a set of feature vectors.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.version = 0.2

    parser.add_argument(
        '-i', 
        '--infile', 
        required = True,
        help = 'Input file.')

    parser.add_argument(
        '-w', 
        '--window', 
        default = 50,
        help = 'The size of the window used for predictions.')

    parser.add_argument(
        '-c',
        '--columns',
        default = ['top_A',
                   'top_T',
                   'top_C',
                   'top_G', 
                   'top_ipd', 
                   'bottom_ipd'],
        help = 'List of columns to include as features.')

    parser.add_argument(
        '-p',
        '--ipd',
        default = 2,
        help = 'IPD threshold value.')

    parser.add_argument(
        '-f',
        '--fold-change',
        default = 10,
        help = 'Fold change threshold value.')

    parser.add_argument(
        '-e',
        '--examples',
        default = 50000,
        help = 'Max number of examples from each class.')

    parser.add_argument(
        '-o', 
        '--outfile', 
        default = None,
        help = 'Output file.')
    
    return parser.parse_args()

def sample(data, examples):
    if len(data) <= examples:
        return data.index.values
    else:
        return data.sample(examples).index.values

def windows(index, data, window, columns):
    radius = int(window/2)
    features = []

    k = 0
    for i in range(len(index)):
        chromosome = index[i][0]
        position = index[i][1]
        lower_bound = position - radius
        upper_bound = position + radius + 1

        try:
            vector = {}
            for column in columns:
                vector[column] = []

            for j in range(lower_bound, upper_bound):
                selection =  data.loc[chromosome, j]
                for column in columns:
                    vector[column].append(selection[column])

            concatenation = []
            for column in columns:
                concatenation += vector[column]
            features.append(concatenation)

        except TypeError:
            k += 1
            
        if (i % 10000 == 0) and (i > 0):
            print (f'{i} examples created.')

    print (f'Skipped {k} examples because of missing values.')
    return features

def main():
    total_start = time.time()
    arguments = setup()

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
    false_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] < fold_change)]
    true_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] < fold_change)]
    false_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] > fold_change)]

    print ('Sampling data.')
    true_positive = sample(true_positive, arguments.examples)
    false_positive = sample(false_positive, arguments.examples)
    true_negative = sample(true_negative, arguments.examples)
    false_negative = sample(false_negative, arguments.examples)

    print ('Extracting windows.')
    start = time.time()
    true_positive_features = windows(true_positive, data, arguments.window, arguments.columns)
    false_positive_features = windows(false_positive, data, arguments.window, arguments.columns)
    true_negative_features = windows(true_negative, data, arguments.window, arguments.columns)
    false_negative_features = windows(false_negative, data, arguments.window, arguments.columns)
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Creating labels.')
    positive_data = true_positive_features + false_negative_features
    positive_labels = list(np.ones(len(positive_data)))
    negative_data = true_negative_features + false_positive_features
    negative_labels = list(np.zeros(len(negative_data)))

    data = np.array(positive_data + negative_data)
	labels = np.array(positive_labels + negative_labels)

	index = np.arange(len(data))
    np.random.shuffle(index)

    data = data[index]
    labels = labels[index]

    if arguments.outfile:
        filename = arguments.outfile
        np.save(f'{filename}_data.npy', data)
        np.save(f'{filename}_labels.npy', labels)
    else:
        directory = os.path.dirname(arguments.infile)
        np.save(os.path.join(directory, 'data.npy'), data)
        np.save(os.path.join(directory, 'labels.npy'), labels)

    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    elapsed = time.time() - total_start
    print (f'{elapsed:.0f} seconds elapsed in total.')

if __name__ == '__main__':
    main()






