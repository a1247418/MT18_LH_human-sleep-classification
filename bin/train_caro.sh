#!/cluster/apps/sfos/bin/bash

setup="dssm_supervised"
dataset="caro"

train=true # experts need to be trained beforehand, if needed
evaluate=false # evaluate with train/test split
cv=true # evaluate with cross validation

if [ "${dataset}" = "caro_new" ]; then
    n_folds=16
fi
if [ "${dataset}" = "caro" ]; then
    n_folds=7
fi

unsupervized=true
if [ "${setup}" = "dssm_supervised" ]; then
    data_config=caro_all_2D_onesided_no_sweat
    model=DSSM_caro
    train_experts=false
    unsupervized=false
fi
if [ "${setup}" = "dssm" ]; then
    data_config=caro_all_2D_onesided_no_sweat
    model=Sleep_Classifier_caro
    train_experts=true
fi
if [ "${setup}" = "att" ]; then
    data_config=caro_all_2D_no_sweat
    model=AttentionNet_RS160_caro
    train_experts=true
fi

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
        if [ "${setup}" = "att" ]; then
            bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EEG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EEG&
            bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EOGL_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EOGL&
            bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EOGR_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EOGR&
            bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EMG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EMG&
        fi
        if [ "${setup}" = "dssm" ]; then
           bsub -n 5 -W 10:00 -R "rusage[mem=18000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} DSSM_caro save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} unsupervized=True log_dir=${log_dir}${setup}&
        fi
    done
    wait
fi

# Train attention model
if ${train} ; then
    for (( i=0; i<${n_folds}; i++ ))
     do
        bsub -n 5 -W 10:00 -R "rusage[mem=18000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} ${model} save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}${setup}&
    done
    wait
fi

# Evaluate (cross validation)
if ${cv} ; then
    bsub -n 5 -W 10:00 -R "rusage[mem=18000,scratch=60000,ngpus_excl_p=1]" -K python cv.py with ${data_config} ${model} save_model=True unsupervized=${unsupervized} ds.nfolds=${n_folds} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${test_csv} log_dir=${log_dir}${setup}&
    wait
fi

# Evaluate
if ${evaluate} ; then
    eval_file="${log_dir}{setup}/model_best.pth.tar"
    for (( i=0; i<${n_folds}; i++ ))
    do
        bsub -n 5 -W 4:00 -R "rusage[mem=18000,scratch=60000,ngpus_excl_p=1]" -K python validate.py --model=${eval_file} --data_dir=${data_dir} --subject_csv=${test_csv} --output_dir=${out_dir}&
    done
    wait
fi

echo "Finished all jobs."