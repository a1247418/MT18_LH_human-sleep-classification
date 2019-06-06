# Sleep Learning


## Getting started
 * Download [anaconda](https://docs.anaconda.com/anaconda/install/) for your system
 
    `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`
 * Run the installer
 `bash Miniconda3-latest-Linux-x86_64.sh `
 * Create an environment
`conda env create -n sl python=3.6.4 -f environment.yml`
 * Activate the created environment `source activate sl`

## Usage

IMPORTANT: 
* When training/prediciting the processed data is by default stored in 
<data_dir>/tmp/. To store the data at different location you must set the 
TMPDIR environment variable accordingly. This variable is set automatically 
on leonahrd when requesting scratch space. It is recommended to store the 
data in /cluster/scratch/<username> and request enough scratch space on the 
executing compute nodes. 
* Be aware the required space for the processed data scales with the number 
of neighbors.
* Before submitting any any task to the scheduling system, load the appropriate python module:
`module load python_gpu/3.6.4`

#### Single Channel Expert Model
Training the single channel expert model local on the EEG channel:
`bsub -n 5 -W 4:00 -R "rusage[mem=6000,scratch=25000,ngpus_excl_p=1]" python train.py with caro_EEG_2D singlechanexp save_model=True ds.fold=0 ds.data_dir=[DATA DIRECTORY] ds.train_csv=../cfg/caro/train.csv ds.val_csv=../cfg/caro/val.csv`

#### Attention Model (Requires trained Expert Models)
The attention model requires trained experts models. Training the attention model local on all channels:
`bsub -n 5 -W 5:00 -R "rusage[mem=6000,scratch=25000,ngpus_excl_p=1]" python train.py with caro_all_2D_no_sweat AttentionNet_RS160_caro save_model=True ds.fold=0 ds.data_dir=[DATA DIRECTORY] ds.train_csv=../cfg/caro/train.csv ds.val_csv=../cfg/caro/val.csv`

#### DSSM
Training an unsupervized DSSM requires trained expert models. By changing the "unsupervized" flag, you can switch between the supervized and unsupervized DSSM:
`bsub -n 5 -W 10:00 -R "rusage[mem=18000,scratch=60000,ngpus_excl_p=1]" python train.py with caro_all_2D_no_sweat DSSM_caro save_model=True unsupervized=[True/False] ds.fold=0 ds.data_dir=[DATA DIRECTORY] ds.train_csv=../cfg/caro/train.csv ds.val_csv=../cfg/caro/val.csv`

#### Sleep Classifier
This model is responsible for translating unsupervized DSSM states into sleep stage predictions. It requires a trained unsupervized DSSM.
`bsub -n 5 -W 5:00 -R "rusage[mem=6000,scratch=25000,ngpus_excl_p=1]" python train.py with caro_all_2D_no_sweat Sleep_Classifier_caro save_model=True ds.fold=0 ds.data_dir=[DATA DIRECTORY] ds.train_csv=../cfg/caro/train.csv ds.val_csv=../cfg/caro/val.csv`


### Evaluation Via Cross-Validation

To train and evaluate a model via cross-validation, you can run the script train_caro.sh. Make sure to set the correct paths in the script before, and to specify the dataset (e.g. caro or caro_new) at the top of the script. With the variabes providied at the beginning of the script you can also skip training or cross-validation.

Cross-validation will create a .npz file for each subject in the subject_csv and save in 
the output_dir. It is recommended to save the results in 

`reports/results/<dataset>/<experiment>/<model arch>/<channel>`

A jupyter notebook can then be used analyze overall accuracy, the confusion 
matrix, etc. An example can be found [here](https://github.com/hlinus/SleepLearning/blob/master/reports/Evaluation-SleepEDF.ipynb). 



