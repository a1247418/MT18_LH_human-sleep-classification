#!/cluster/apps/sfos/bin/bash

n_folds=16
dataset="caro_new"
train_experts=false
train=false
evaluate=false
cv=true

home_path="/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification/"
data_path="/cluster/scratch/llorenz/data/${dataset}/"
cfg_path="${home_path}cfg/${dataset}/"
train_csv="${cfg_path}cv_train.csv"
val_csv="${cfg_path}cv_val.csv"
test_csv="${cfg_path}cv_test.csv"
out_dir="${home_path}reports/results/${dataset}/attention/all/"
log_dir="${home_path}logs/cv_ready/${dataset}/"

# Train experts
if ${train_experts} ; then
    for (( i=0; i<${n_folds}; i++ ))
    do
        bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EEG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EEG&
        bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EOGL_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EOGL&
        bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EOGR_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EOGR&
        bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EMG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EMG&
    done
    wait
fi

# Train attention model
if ${train} ; then
    for (( i=0; i<${n_folds}; i++ ))
     do
        bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_all_2D AttentionNet_RS160_caro save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}att&
    done
    wait
fi

# Evaluate (cross validation)
if ${cv} ; then
    bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python cv.py with caro_all_2D AttentionNet_RS160_caro save_model=True ds.nfolds=${n_folds} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${test_csv} log_dir=${log_dir}att&
    wait
fi

# Evaluate
if ${evaluate} ; then
    eval_file="${log_dir}att/model_best.pth.tar"
    for (( i=0; i<${n_folds}; i++ ))
    do
        bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python validate.py --model=${eval_file} --data_dir=${data_dir} --subject_csv=${test_csv} --output_dir=${out_dir}&
    done
    wait
fi

echo "Finished all jobs."