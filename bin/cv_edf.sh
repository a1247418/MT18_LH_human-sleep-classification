#!/cluster/apps/sfos/bin/bash

dataset="sleepedf"
home_path="/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification"
data_path="${home_path}data/${dataset}/"
cfg_path="${home_path}data/${dataset}/"

bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_EOG_2D singlechanexp&
bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_Fpz_2D singlechanexp&
bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_Pz_2D singlechanexp&
bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with sleepedf_all_2D AttentionNet_RS160

wait

echo "Finished all jobs."