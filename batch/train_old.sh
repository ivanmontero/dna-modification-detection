#!/bin/bash

#PBS -N train_old
#PBS -A pmyler
#PBS -l nodes=1:ppn=28,mem=500gb
#PBS -M aakash.sur@seattlechildrens.org
#PBS -q workq

cd /active/myler_p/People/Sur/software/dna-modification-detection

python src/data/preprocessing.py -i data/raw/old_ipd.csv -f data/raw/fold_change.csv -p old

python src/features/extraction.py -i data/interm/old_data.h5 -p old

python src/models/train_model.py -i data/processed/old_data.npy -d data/interm/old_data.h5 -m data/processed/old_metadata.json -p old_100000 --progress-off --holdout LtaP_36
