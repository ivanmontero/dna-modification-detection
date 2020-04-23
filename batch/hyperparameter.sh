#!/bin/bash

#PBS -N hyperparameter
#PBS -A pmyler
#PBS -l nodes=1:ppn=14,mem=200gb
#PBS -M aakash.sur@seattlechildrens.org
#PBS -q workq

cd /active/myler_p/People/Sur/software/dna-modification-detection

python src/models/hyperparameter_search.py