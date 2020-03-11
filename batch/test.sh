#!/bin/bash

#PBS -N test
#PBS -A pmyler
#PBS -l nodes=1:ppn=28,mem=500gb
#PBS -M aakash.sur@seattlechildrens.org

cd /active/myler_p/People/Sur/software/dna-modification-detection
python src/models/train_model.py -i data/processed/new_data.npy -d data/interm/new_data.h5 -m data/processed/new_metadata.json -p new --skip-final --holdout LtaP_36
