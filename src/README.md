# Instructions
All default values should work.
## l_tarentolae processing
Creates all of the genome data necessary for exploration.
* run data/create_tsv.py to concat all the genome information into a single .tsv file (l_tarentolae.tsv)
* run data/generate_sequences.py to create two files: one containing the center-of-sequence information (centers.csv) and another containing the actual sequence information for each column (sequences.npy)
    - Creates sequences centered around ipd values that are greater than a specified ipd ratio on both strands.
## plasmid processing
Creates all of the plasmid data necessary for exploration.
* run data/create_tsv_plasmid.py to concat all the plasmid data into a single .tsv file (plasmid.tsv)
* run data/fix_js.py to process the ground truth J information and combine it with plasmid.tsv to create a file containing both information (with consistent namimg) (written to plasmid_and_j.csv)
* run data/generate_sequences_plasmid.py to create four files: two containing the center-of-sequence information for both the bottom (plasmid_bottom_centers.csv) and the top (plasmid_top_centers.csv), and two containing the actual sequence for each column corresponding to those centers (plasmid_bottom_sequences.npy and plasmid_top_sequences.npy respectively)
    - Creates sequences for the entire plasmid

## k-folds cross validation classification - genome
Evaluates classifier performances on classifying sequences in the genome
* run classification/classify_kfolds.py to generate ROC plots for specified classifiers to evaluate their performance on determining if a center appears under high anti-j immunoprecipitation values

## k-folds cross validation classification - plasmid
Evaluates classifier performances on classifying sequences in the plasmids
* run classification/classify_kfolds_plasmid.py to generate ROC plots for specified classifiers to evaluate their performance on determining if a center is a base J.

## evaluating a classifier trained on the genome on the plasmid data
Runs a classfier trained on the genome on the plasmid data, then, for each area predicted with high confidence, we alter the Ts in the region and see which causes the largest drop in confidence.
* run classification/eval_on_plasmids.py to create a tsv file that contains the original plasmid data as well as the predicted confidences.
* run classification/viz_plasmids.py to visualize the ipd values, classifier confidences, drops in classfier confidences at specific Ts, and the ground truth Js.

## TODO roadmap
- [x] update data/create_tsv.py
- [x] update data/create_tsv_plasmid.py
- [x] update data/generate_sequences.py
- [x] update data/generate_sequences_plasmid.py
- [x] update data/fix_js.py
- [x] update classification/classify_kfolds.py
- [x] update classification/classify_kfolds_plasmid.py
- [x] update classification/eval_on_plasmids.py
- [x] update viz_plasmids.py
- [ ] update and finish classification/eval_on_plasmids_dual.py
- [ ] update classification code to use new repo format
- [ ] create txt file that specifies what scripts create what files