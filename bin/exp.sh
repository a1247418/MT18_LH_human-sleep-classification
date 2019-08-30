#!/cluster/apps/sfos/bin/bash
# first argument: flags
# second argument: folder
# 3rd argument: data

if [ "$3" != "" ]; then
    data=$3
else
    data="caro_all_2D_onesided_no_sweat"
fi

for (( i=10; i<20; i++ ))
do
    bsub -n 5 -W 8:00 -R volta -R "rusage[mem=12000,scratch=60000,ngpus_excl_p=1]" -oo ../logs/experiments_fin/$2${i}/out.lsf python train.py with $data DSSM_caro save_model=False unsupervized=False ds.fold=0 ds.data_dir=/cluster/scratch/llorenz/data/caro_new/ ds.train_csv=../cfg/caro/cv_train.csv ds.val_csv=../cfg/caro/cv_test.csv log_dir=../logs/experiments_fin/$2${i} $1 seed=${i}&
    #bsub -n 5 -W 4:00 -R volta -R "rusage[mem=10000,scratch=60000,ngpus_excl_p=1]" -oo ../logs/experiments_fin/$2${i}/out.lsf python train.py with $data DSSM_caro save_model=False unsupervized=False ds.fold=0 ds.data_dir=/cluster/scratch/llorenz/data/caro_new/ ds.train_csv=../cfg/caro/cv_train.csv ds.val_csv=../cfg/caro_new/cv_test_mod.csv log_dir=../logs/experiments_fin/$2${i} $1 seed=${i}&
    #sleep 7
done