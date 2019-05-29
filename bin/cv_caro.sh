#!/cluster/apps/sfos/bin/bash

dataset="caro"
home_path="/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification/"
data_path="${home_path}data/${dataset}/"
cfg_path="${home_path}cfg/${dataset}/"
train_csv="${cfg_path}debug_train.csv"
val_csv="${cfg_path}debug_val.csv"
test_csv="${cfg_path}debug_test.csv"

bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_EOG_2D singlechanexp ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv}&
bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_Fpz_2D singlechanexp ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv}&
bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_Pz_2D singlechanexp ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv}&
bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with caro_all_2D AttentionNet_RS160 ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv}&

wait

echo "Finished all jobs."