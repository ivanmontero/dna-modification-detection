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

    directory = os.path.dirname(arguments.infile)
    if arguments.save_classes:
        classes = ['tp'] * len(true_positive_features) + \
                  ['fn'] * len(false_negative_features) + \
                  ['tn'] * len(true_negative_features) + \
                  ['fp'] * len(false_positive_features)

        data = {'vectors': features.tolist(), 
                'positions': positions.tolist(), 
                'classes': classes}

        if arguments.output:
            filename = os.path.join(directory, f'{arguments.output}_classes.json')
        else:
            filename = os.path.join(directory, 'classes.json')

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent = 4)

        tp_seqeunces = data_extraction.create_fasta(true_positive_features, arguments.window)
        fn_seqeunces = data_extraction.create_fasta(false_negative_features, arguments.window)
        tn_seqeunces = data_extraction.create_fasta(true_negative_features, arguments.window)
        fp_seqeunces = data_extraction.create_fasta(false_positive_features, arguments.window)

        if arguments.output:        
            filename = os.path.join(directory, f'{arguments.output}_tp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tp_seqeunces)

            filename = os.path.join(directory, f'{arguments.output}_fn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fn_seqeunces)

            filename = os.path.join(directory, f'{arguments.output}_tn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tn_seqeunces)

            filename = os.path.join(directory, f'{arguments.output}_fp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fp_seqeunces)

        else:
            filename = os.path.join(directory, 'tp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tp_seqeunces)

            filename = os.path.join(directory, 'fn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fn_seqeunces)

            filename = os.path.join(directory, 'tn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tn_seqeunces)

            filename = os.path.join(directory, 'fp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fp_seqeunces)

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






