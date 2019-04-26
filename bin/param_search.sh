#!/cluster/apps/sfos/bin/bash

dataset="caro"

home_path="/cluster/home/llorenz/sleep/MT18_LH_human-sleep-classification/"
data_path="/cluster/scratch/llorenz/data/${dataset}/"
cfg_path="${home_path}cfg/${dataset}/"
train_csv="${cfg_path}debug_train_med.csv"
val_csv="${cfg_path}debug_val_med.csv"
out_dir="${home_path}reports/results/${dataset}/attention/all/"
log_dir="${home_path}logs/cv_ready/${dataset}/"


declare -a hidden=(16 32 64 128)
hiddenl=${#hidden[@]}

declare -a filter=(32 64 128)
filterl=${#filter[@]}

declare -a sep=(0)
sepl=${#sep[@]}

declare -a sweat=(1)
sweatl=${#sweat[@]}

declare -a laball=(1)
laballl=${#laball[@]}

declare -a bs=(16 32)
bsl=${#bs[@]}

declare -a lr=("adam,lr=0.001" "adam,lr=0.01" "adam,lr=0.001" "adam,lr=0.0001")
lrl=${#lr[@]}

for (( i=0; i<${hiddenl}; i++ ));
do
for (( j=0; j<${filterl}; j++ ));
do
for (( k=0; k<${sepl}; k++ ));
do
for (( l=0; l<${sweatl}; l++ ));
do
for (( m=0; m<${laballl}; m++ ));
do
for (( n=0; n<${lrl}; n++ ));
do
for (( o=0; o<${bsl}; o++ ));
do
    dat="caro_all_2D_onesided"
    if [ ${sweat[$l]} -eq 1 ]
    then
        dat="caro_all_2D_onesided_no_sweat"
    fi

    bsub -n 5 -W 4:00 -R "rusage[mem=8000,scratch=65000,ngpus_excl_p=1]" -oo "_exp_h"${hidden[$i]}"_f"${filter[$j]}"_se"${sep[$k]}"_sw"${sweat[$l]}"_la"${laball[$m]}"_lr"${lr[$n]}"_bs"${bs[$o]} -K python train.py with ${dat} DSSM_caro save_model=False ds.fold=0 ds.data_dir=${data_path} ds.train_csv=${train_csv} ds.val_csv=${val_csv} ms.hidden_size=${hidden[$i]} ms.filter_size=${filter[$j]} ms.sep_channels=${sep[$k]} ms.label_nbrs=${laball[$m]} ms.optim=${lr[$n]} ds.batch_size_train=${bs[$o]}&
done
done
done
done
done
wait
done
done
