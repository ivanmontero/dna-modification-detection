## SRC format
classification/
* Contains the initial exploration code to try out different methods of classification on the genome information
data/
* Contains scripts necessary for processing the data into useful formats for feature extraction.
features/
* Contains scripts to process the data into feature formats that can be easily be processed.
models/
* Contains scripts to train and evaluate models on the features extracted from the data.


### Steps

1) Run the preprocessing. This will merge the data from the PacBio run with the data from the ChIP experiment. See test data for expected format. 
`python src/data/preprocessing.py -i data/test/test_new_ipd.csv -f data/test/test_fold_change.csv -p test`

2) Generate the feature vectors from windows of genomic information.  
`python src/features/extraction.py -i data/interm/test_data.h5 -p test`

3) Train the model on the data. 
`python src/models/train_model.py -i data/processed/test_data.npy -d data/interm/test_data.h5 -m data/processed/test_metadata.json -p test`

4) Preprocess the plasmid data to test the predictions on. 
`python src/data/preprocessing.py -i data/raw/1.csv -p plasmid_1`

5) Generate the feature vectors for some plasmid data. 
`python src/features/extraction.py -i data/interm/plasmid_1_data.h5 -p plasmid_1`

6) Create predictions on the plamid data using the trained network. 
`python src/models/predict_model.py -i data/processed/plasmid_1_data.npy -d data/interm/plasmid_1_data.h5 -m models/test_model.h5 -c 25L_PLASMID_corrected -s 4212 -e 4270`
