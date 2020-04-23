import numpy as np
import subprocess

def training(prefix, description, layers, dropout):
	string_layers = []
	for item in layers:
		string_layers.append(str(item))

	command = ['python','src/models/train_model.py',
		'--input', 'data/processed/new_data.npy',
		'--dataframe', 'data/interm/new_data.h5',
		'--metadata', 'data/processed/new_metadata.json',
		'--holdout', 'LtaP_36',
		'--prefix', prefix,
		'--description', description,
		'--n-examples', '100000',
		'--progress-off',
		'--layers'] + string_layers + \
		['--dropout', str(dropout)]

	print (command)

	return command

def predict(prefix, description):
	command = ['python', 'src/models/predict_model.py',
		'--input', 'data/processed/new_data.npy',
		'--dataframe', 'data/interm/new_data.h5',
		'--model', f'models/{prefix}_model.h5',
		'--location', 'LtaP_25:414070:414170', 'LtaP_26:1073700:1073777',
		'--prefix', prefix,
		'--metadata', 'data/processed/new_metadata.json',
		'--description', description,
		'--progress-off']

	return command

def run_iteration(prefix, description, layers = [300, 150, 50], dropout = 0.25):
	# Training 
	subprocess.run(
		training(
			prefix = prefix, 
			description = description,
			layers = layers,
			dropout = dropout))

	# Predict
	subprocess.run(
		predict(
			prefix = prefix, 
			description = description))   

def main():
	first_layer = np.linspace(0, 1000, 6, dtype = int)
	for neurons in first_layer:
		prefix = f'single_layer_{neurons}'
		description = f'Single Layer: {neurons} Neurons'
		layers = [neurons]

		print (description)
		run_iteration(
			prefix = prefix,
			description = description,
			layers = layers)

if __name__ == '__main__':
	main()

