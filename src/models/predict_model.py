from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import argparse
import json
import time

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Train a model on the features and save it.')

    parser.version = 0.1

    parser.add_argument(
        '-i', 
        '--input', 
        required = True,
        help = 'Input filename for predictions.')

    parser.add_argument(
        '-m', 
        '--model', 
        required = True,
        help = 'Filename for model.')

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


def load(filename):
    with open(filename) as infile:
        contents = json.load(infile)
        data = contents['vectors']
        positions = contents['positions']
        feature_args = contents['arguments']

    return np.array(data), np.array(positions), feature_args

def feature_importance(vector, prediction, model, feature_args):
    features = np.zeros(feature_args['window']) # TODO: Make generic, so that window size is an option
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
    for i in range(50):
        features[i] += max(0, prediction - min(new_predictions[i*3:i*3+3]))

    return features

def main():
    arguments = setup()
    model = keras.models.load_model(arguments.model)
    data, positions, feature_args = load(arguments.input)

    predictions = model.predict(data)
    plasmid = np.zeros(positions.max() + 1)

    for i in range(len(data)):
        print (i)
        window = positions[i]
        vector = data[i] 
        prediction = predictions[i]
        plasmid[window] -= feature_importance(vector, prediction, model, feature_args)

    plt.figure(figsize = (8,4), dpi = 100)
    plt.plot(plasmid[4000:4500])
    plt.savefig('test.png')

if __name__ == '__main__':
    main()

