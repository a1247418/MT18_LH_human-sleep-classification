#!/cluster/apps/sfos/bin/bash
# 1st argument: model
# 2nd argument: data-set
# 3rd argument: extra model parameters, e.g. "ms.hidden_size=32"
# 4th argument: extra data parameters, e.g. "ds.nbrs=3"


if [ "$1" != "" ]; then
    setup="$1"
else
    setup="dssm_supervised"
fi

if [ "$2" != "" ]; then
    dataset="$2"
else
    dataset="caro_new"
fi

if [ "$3" != "" ]; then
    extra_ms="$3"
else
    extra_ms=""
fi
if [ "$4" != "" ]; then
    extra_ds="$4"
else
    extra_ds=""
fi

train=true # experts need to be trained beforehand, if needed
evaluate=false # evaluate with train/test split
cv=true # evaluate with cross validation

if [ "${dataset}" = "caro_new" ]; then
    n_folds=20
fi
if [ "${dataset}" = "caro" ]; then
    n_folds=7
fi
if [ "${setup}" = "dssm_supervised" ]; then
    data_config=caro_all_2D_onesided_no_sweat
    model=DSSM_caro
    train_experts=false
    unsupervized=False
fi
if [ "${setup}" = "ensembler" ]; then
    data_config=caro_all_2D_onesided_no_sweat
    model=Ensembler_caro
    train_experts=true
    train=true
    unsupervized=False
fi
if [ "${setup}" = "dssm" ]; then
    data_config=caro_all_2D_onesided_no_sweat
    model=Sleep_Classifier_caro
    train_experts=true
    unsupervized=True
fi
if [ "${setup}" = "att" ]; then
    data_config=caro_all_2D_no_sweat
    model=AttentionNet_RS160_caro
    train_experts=true
    unsupervized=False
fi
if [ "${dataset}" = "sleepedf" ]; then
    n_folds=20
    if [ "${setup}" = "att" ]; then
        model=AttentionNet_RS160_edf
    fi
    data_config=sleepedf_all_2D
fi

home_path="/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification/"
data_path="/cluster/scratch/llorenz/data/${dataset}/"
cfg_path="${home_path}cfg/${dataset}/"
train_csv="${cfg_path}cv_train.csv"
val_csv="${cfg_path}cv_val.csv"
test_csv="${cfg_path}cv_test.csv"
out_dir="${home_path}reports/results/${dataset}/${setup}/all/"
log_dir="/cluster/scratch/llorenz/logs/"
model_dir="/cluster/scratch/llorenz/models/"

# Train experts
if ${train_experts} ; then
    for (( i=1; i<${n_folds}; i++ ))
    do
        if [ "${setup}" = "att" ]; then
            if [ "${dataset}" = "sleepedf" ]; then
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_Pz_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${model_dir}debug_edf_1 ${extra_ds}&
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_Fpz_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${model_dir}debug_edf_2 ${extra_ds}&
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with sleepedf_EOG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${model_dir}debug_edf_3 ${extra_ds}&
            else
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EEG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EEG ${extra_ds}&
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EOGL_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EOGL ${extra_ds}&
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EOGR_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EOGR ${extra_ds}&
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EMG_2D singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EMG ${extra_ds}&
                bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=65000,ngpus_excl_p=1]" -K python train.py with caro_EEG_2D_no_sweat singlechanexp save_model=True ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}EEG_raw  ${extra_ds}&
            fi
        fi
        if [ "${setup}" = "dssm" ]; then
           bsub -n 5 -W 10:00 -R volta -R "rusage[mem=8000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} DSSM_caro save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} unsupervized=True log_dir=${log_dir}dssm_usv  ${extra_ms} ${extra_ds}&
        fi
        if [ "${setup}" = "ensembler" ]; then
           bsub -n 5 -W 10:00 -R volta -R "rusage[mem=12000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} DSSM_caro save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} seed=1 log_dir=${log_dir}dssm1 ${extra_ms} ${extra_ds}&
           bsub -n 5 -W 10:00 -R volta -R "rusage[mem=12000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} DSSM_caro save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} seed=2 log_dir=${log_dir}dssm2 ${extra_ms} ${extra_ds}&
           bsub -n 5 -W 10:00 -R volta -R "rusage[mem=12000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} DSSM_caro save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} seed=3 log_dir=${log_dir}dssm3 ${extra_ms} ${extra_ds}&
        fi
    done
    wait
fi

# Train attention model
if ${train} ; then
    for (( i=1; i<${n_folds}; i++ ))
     do
        bsub -n 5 -W 10:00 -R volta -R "rusage[mem=16000,scratch=60000,ngpus_excl_p=1]" -K python train.py with ${data_config} ${model} save_model=True unsupervized=${unsupervized} ds.fold=${i} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} log_dir=${log_dir}tr_${setup} ${extra_ms} ${extra_ds}&
    done
    wait
fi

# Evaluate (cross validation)
if ${cv} ; then
    bsub -n 5 -W 4:00 -R "rusage[mem=16000,scratch=60000,ngpus_excl_p=1]" -K python cv.py with ${data_config} ${model} save_model=True unsupervized=${unsupervized} ds.nfolds=${n_folds} ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${test_csv} log_dir=${log_dir}tr_${setup} ${extra_ds}&
    wait
fi

# Evaluate
if ${evaluate} ; then
    eval_file="${log_dir}{setup}/model_best.pth.tar"
    for (( i=1; i<${n_folds}; i++ ))
    do
        bsub -n 5 -W 4:00 -R "rusage[mem=16000,scratch=60000,ngpus_excl_p=1]" -K python validate.py --model=${eval_file} --data_dir=${data_dir} --subject_csv=${test_csv} --output_dir=${out_dir} ${extra_ms} ${extra_ds}&
    done
    wait
fi

echo "Finished all jobs."