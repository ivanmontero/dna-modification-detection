import argparse

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
        default = 1000,
        help = 'Max number of examples from each class.')

    parser.add_argument(
        '--save-classes',
        action='store_true',
        default = False,
        help = argparse.SUPPRESS)

    parser.add_argument(
        '-o', 
        '--outfile', 
        default = None,
        help = 'Output file. Default auto-generates name.')
    
    return parser.parse_args()

def sample(data, examples):
    if len(data) <= examples:
        return data.index.values
    else:
        return data.sample(examples).index.values

def windows(index, data, window, columns):
    radius = int(window/2)
    features = []
    positions = []

    k = 0
    for i in range(len(index)):
        chromosome = index[i][0]
        position = index[i][1]
        lower_bound = position - radius
        upper_bound = position + radius 

        try:
            feature_vector = {}
            for column in columns:
                feature_vector[column] = []

            coordinates = range(lower_bound, upper_bound)
            for j in coordinates:
                selection =  data.loc[chromosome, j]
                for column in columns:
                    feature_vector[column].append(selection[column])

            concatenation = []
            for column in columns:
                concatenation += feature_vector[column]
            features.append(concatenation)

            positions.append(coordinates) 

        except TypeError:
            k += 1
            
        if (i % 10000 == 0) and (i > 0):
            print (f'{i} examples created.')

    print (f'Skipped {k} examples because of missing values.')
    return features, positions





