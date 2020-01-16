import data_extraction
import pandas as pd
import numpy as np
import json
import os

# Data extraction from the genome file.
def main():
    total_start = data_extraction.start_time()
    arguments = data_extraction.setup()

    start = data_extraction.start_time('Reading data.')
    data = pd.read_hdf(arguments.infile, columns = ['fold_change'] + arguments.columns)
    data_extraction.end_time(start)

    print ('Filtering data.')
    ipd = arguments.ipd
    fold_change = arguments.fold_change

    data = data[~data['chromosome'].isin(arguments.exclude)]

    true_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] > fold_change)]
    false_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] > fold_change)]
    true_negative = data[(data['top_ipd'] < ipd) & (data['bottom_ipd'] < ipd) & (data['fold_change'] < fold_change)]
    false_positive = data[((data['top_ipd'] > ipd) | (data['bottom_ipd'] > ipd)) & (data['fold_change'] < fold_change)]

    print ('Sampling data.')
    true_positive = data_extraction.sample(true_positive, arguments.examples)
    false_positive = data_extraction.sample(false_positive, arguments.examples)
    true_negative = data_extraction.sample(true_negative, arguments.examples)
    false_negative = data_extraction.sample(false_negative, arguments.examples)

    start = data_extraction.start_time('Extracting windows.')
    true_positive_features, true_positive_positions = data_extraction.windows(true_positive, data, arguments.window, arguments.columns)
    false_negative_features, false_negative_positions = data_extraction.windows(false_negative, data, arguments.window, arguments.columns)
    true_negative_features, true_negative_positions = data_extraction.windows(true_negative, data, arguments.window, arguments.columns)
    false_positive_features, false_positive_positions = data_extraction.windows(false_positive, data, arguments.window, arguments.columns)
    data_extraction.end_time(start)

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

    project_folder = data_extraction.project_path()
    data_folder = os.path.join(project_folder, 'data')
    processed_folder = os.path.join(data_folder, 'processed')
    interm_folder = os.path.join(data_folder, 'interm')

    column_labels = []
    for column in arguments.columns:
        column_labels += [column] * arguments.window

    if arguments.save_classes:
        print ('Saving Classes')
        classes = [0] * len(true_positive_features) + \
                  [1] * len(false_negative_features) + \
                  [2] * len(true_negative_features) + \
                  [3] * len(false_positive_features)

        data = {'classes': {0: 'True Positive', 
                            1: 'False Negative',
                            2: 'True Negative',
                            3: 'False Positive'},
                'columns': column_labels,
                'vectors': features.tolist(), 
                'positions': positions.tolist(), 
                'labels': classes}

        if arguments.prefix:
            filename = os.path.join(interm_folder, f'{arguments.prefix}_classes.json')
        else:
            filename = os.path.join(interm_folder, 'classes.json')

        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent = 4)

        tp_seqeunces = data_extraction.create_fasta(true_positive_features, arguments.window)
        fn_seqeunces = data_extraction.create_fasta(false_negative_features, arguments.window)
        tn_seqeunces = data_extraction.create_fasta(true_negative_features, arguments.window)
        fp_seqeunces = data_extraction.create_fasta(false_positive_features, arguments.window)

        if arguments.prefix:        
            filename = os.path.join(interm_folder, f'{arguments.prefix}_tp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tp_seqeunces)

            filename = os.path.join(interm_folder, f'{arguments.prefix}_fn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fn_seqeunces)

            filename = os.path.join(interm_folder, f'{arguments.prefix}_tn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tn_seqeunces)

            filename = os.path.join(interm_folder, f'{arguments.prefix}_fp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fp_seqeunces)

        else:
            filename = os.path.join(interm_folder, 'tp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tp_seqeunces)

            filename = os.path.join(interm_folder, 'fn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fn_seqeunces)

            filename = os.path.join(interm_folder, 'tn.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(tn_seqeunces)

            filename = os.path.join(interm_folder, 'fp.fasta')
            with open(filename, 'w') as outfile:
                outfile.write(fp_seqeunces)

    index = np.random.permutation(len(features))
    features = features[index]
    positions = positions[index]
    labels = labels[index]

    print ('Writing output.')
    data = {'columns': column_labels,
            'vectors': features.tolist(), 
            'positions': positions.tolist(), 
            'labels': labels.tolist(),
            'arguments': vars(arguments)}

    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_data.json')
    else:
        filename = os.path.join(processed_folder, 'data.json')

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

    total_time = data_extraction.end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()






