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

    return np.array(data), np.array(positions)

def feature_importance(vector, prediction, model):
    features = np.zeros(50) # TODO: Make generic, so that window size is an option
    alternate = []

    for i in range(50): # Assumes a window size, and that the bases appear at the beginning
        a = i
        t = i + 50
        c = i + 100
        g = i + 150

        current = vector.copy()
        embedding = vector[[a,t,c,g]]
        new = 0
        for j in range(1,4):
            current[[a,t,c,g]] = np.roll(embedding, j)
            alternate.append(current.copy())

    alternate = np.array(alternate)
    new_predictions = model.predict(alternate, batch_size = len(alternate))
    for i in range(50):
        features[i] += min(0, prediction - np.mean(new_predictions[i*3:i*3+3]))

    return features

def main():
    arguments = setup()
    model = keras.models.load_model(arguments.model)
    data, positions = load(arguments.input)

    predictions = model.predict(data)
    plasmid = np.zeros(positions.max() + 1)

    for i in range(len(data)):
        print (i)
        window = positions[i]
        vector = data[i] 
        prediction = predictions[i]
        plasmid[window] -= feature_importance(vector, prediction, model)

    plt.figure(figsize = (8,4), dpi = 100)
    plt.plot(plasmid[4000:4500])
    plt.savefig('test.png')

if __name__ == '__main__':
    main()

