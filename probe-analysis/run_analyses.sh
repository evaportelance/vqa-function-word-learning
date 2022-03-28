#!/bin/bash



# Experiment 3

#for seed in {0..4}
#do
#python analysis_and_or.py --res_dir "../../data/results/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"  --pred_dir "../../data/preds/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"

#python analysis_same.py --res_dir "../../data/results/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"  --pred_dir "../../data/preds/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"

#python analysis_more_less.py --res_dir "../../data/results/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"  --pred_dir "../../data/preds/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"

#python analysis_behind_front.py --res_dir "../../data/results/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"  --pred_dir "../../data/preds/2022-02-28_experiment3_e25_4mac_newSAME_seed$seed"
#done

# New and2 or2 probe results
#Experiment1
#for seed in {0..4}
#do
#python analysis_and_or.py --res_dir "../../data/results/2022-03-06_experiment1_e25_4mac_newANDORonly_seed$seed"  --pred_dir "../../data/preds/2022-03-06_experiment1_e25_4mac_newANDORonly_seed$seed"
#done

#Experiment3

#for seed in {0..4}
#do
#python analysis_and_or.py --res_dir "../../data/results/2022-03-11_experiment3_e25_4mac_newANDORonly_seed$seed"  --pred_dir "../../data/preds/2022-03-11_experiment3_e25_4mac_newANDORonly_seed$seed"
#done

#for seed in {0..4}
#do
#python analysis_and_or.py --res_dir "../../data/results/2022-03-16_experiment2_e25_4mac_newANDORonly_seed$seed"  --pred_dir "../../data/preds/2022-03-16_experiment2_e25_4mac_newANDORonly_seed$seed"
#done

#for seed in {0..4}
#do
#python analysis_and_or.py --res_dir "../../data/results/2022-03-20_experiment3noor_e25_4mac_seed$seed"  --pred_dir "../../data/preds/2022-03-20_experiment3noor_e25_4mac_seed$seed"
#done

python analysis_and_or.py --res_dir "../../data/results/2022-03-20_experiment3noor_e25_4mac_seed4"  --pred_dir "../../data/preds/2022-03-20_experiment3noor_e25_4mac_seed4"
