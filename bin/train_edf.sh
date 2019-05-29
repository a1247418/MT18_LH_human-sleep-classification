#!/cluster/apps/sfos/bin/bash

n_folds=1
dataset="sleepedf"
home_path="/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification/"
data_path="${home_path}data/${dataset}/"
cfg_path="${home_path}cfg/${dataset}/"
train_csv="${cfg_path}debug_train.csv"
val_csv="${cfg_path}debug_val.csv"
test_csv="${cfg_path}debug_test.csv"
out_dir="${home_path}reports/results/sleepedf/train/attention/all/"

eval_file="${home_path}logs/97/model_best.pth.tar"

let max_fold_cnt=${n_folds}-1
for i in {0..${max_fold_cnt}}
do
    bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_EOG_2D singlechanexp save_model=True ds.fold=${i}&
    bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_Fpz_2D singlechanexp save_model=True ds.fold=${i}&
    bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_Pz_2D singlechanexp save_model=True ds.fold=${i}&
done
wait

for i in {0..${max_fold_cnt}}
do
    bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_all_2D AttentionNet_RS160_edf save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv}&
done
wait

for i in {0..${max_fold_cnt}}
do
    bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python validate.py --model=${eval_file} --data_dir=${data_dir} --subject_csv=${test_csv} --output_dir=${out_dir}&
done
wait

echo "Finished all jobs."