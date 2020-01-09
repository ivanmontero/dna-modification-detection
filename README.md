# Recognizing Base J from SMRT Sequencing
Base J is a glycosylated nucleotide found in Trypanosomatids, a family of single celled parasites responsible for Sleeping Sickness, Chagas Disease, and Leishmaniasis. This modified base is thought to play an important role in transcriptional termination for these organisms. Current methods of analyzing Base J rely solely on anti-J immunoprecipitation experiments, which provide low resolution information pertaining to Base J positions. SMRT-seq produces auxiliary information during sequencing that has shown to reveal information related to the positioning of Base J.

We explore analytical approaches such as dimensionality reduction, machine learning, and signal processing to determine which patterns of interpulse duration (IPD) correspond to evidence for Base J across the entire Leishmania tarentolae genome.  We extract the IPD values and base sequences that appear near IPD peaks and train classifiers to predict whether they will appear near anti-J immunoprecipitation (ChIP) peaks.

## Repo format
data/
- interim/
    * Intermediate data created from the raw data
- legacy/
    * Processed data that older code may use
- processed/
    * Processed raw data that other programs use
- raw/
    * The raw data that is used to produce processed versions

reports/
- Contains figures and PDFs of results obtained through our experiments.

src/
- classification/
    * Contains all the code related to classification
- data/
    * Contains all the code related to data manipulation/processing
- features/
    * Contains all the code related to visualization.
- models/
    * Contains all the code related to training and predicting with a model.

## Requirements
* numpy
* pandas
* tensorflow
* scikit-learn
* matplotlib
* xgboost
* keras
* scipy