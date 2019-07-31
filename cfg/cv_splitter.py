import os
import csv
import argparse
import re
import random

import pdb


def split(data_dir, csv_dir, file_id_re, subj_id_re, n_folds, n_val_files=2):
    # Gather how many subjects and nights there are
    files = [f for f in os.listdir(data_dir) if
             os.path.isfile(os.path.join(data_dir, f))]
    n_nights = 0
    subj2file_name = {}
    subject_ids = []

    for file in files:
        file_id = re.findall(file_id_re, file)
        if file_id:
            assert len(file_id) == 1  # Only one night per file
            if type(file_id[0])==tuple:
                file_id = [f for f in file_id[0] if f] # Sort out multiple matches
            subject_id = int(re.findall(subj_id_re, file_id[0])[0])
            subject_ids.append(subject_id)
            if subject_id in subj2file_name:
                subj2file_name[subject_id] += file_id
            else:
                subj2file_name[subject_id] = file_id
            n_nights += 1
    subject_ids = sorted(list(set(subject_ids)))
    print(subject_ids)
    n_subjects = len(subject_ids)
    print(f"Found {n_subjects} subjects.")

    # Generate a random permutation of fold assignments
    fold_assignments = []
    for i in range(n_subjects):
        fold_assignments.append(i % n_folds)
    # random.shuffle(fold_assignments)
    subj2fold = {subject_ids[i]: fold_assignments[i] for i in range(n_subjects)}
    subj_per_fold = n_subjects//n_folds

    # Save to csv
    with open(csv_dir + 'cv_train.csv', 'w+', newline="") as tr_file,\
         open(csv_dir + 'cv_val.csv', 'w+', newline="") as val_file,\
         open(csv_dir + 'cv_test.csv', 'w+', newline="") as te_file,\
         open(csv_dir + 'cv_tune.csv', 'w+', newline="") as tu_file,\
         open(csv_dir + 'cv_tune_train.csv', 'w+', newline="") as tutr_file,\
         open(csv_dir + 'cv_tune_val.csv', 'w+', newline="") as tuval_file,\
         open(csv_dir + 'cv_tune_test.csv', 'w+', newline="") as tute_file:
        tr_writer = csv.writer(tr_file)
        val_writer = csv.writer(val_file)
        te_writer = csv.writer(te_file)
        tu_writer = csv.writer(tu_file) # tune file
        tutr_writer = csv.writer(tutr_file) # tune train file
        tuval_writer = csv.writer(tuval_file) # tune train file
        tute_writer = csv.writer(tute_file) # tune test file

        # in the first step, split in train and test subjects
        tr_names_all = []
        te_names_all = []
        for fold in range(n_folds):
            # Randomize order in which the subjects are listed
            subject_order = [x for x in subject_ids]
            random.shuffle(subject_order)
            tr_names = []
            te_names = []
            for subj in subject_order:
                # Test samples
                if subj2fold[subj] == fold:
                    te_names += subj2file_name[subj]
                # Train samples
                else:
                    tr_names += subj2file_name[subj]
            tr_names_all.append(tr_names)
            te_names_all.append(te_names)

        # split off the validation and tuning set
        val_names_all = [tr[-n_val_files:] for tr in tr_names_all]
        tr_names_all = [tr[:-n_val_files] for tr in tr_names_all]
        tu_names_all = [[te[0]] for te in te_names_all]
        tutr_names_all = [tr[:-n_val_files] for tr in tr_names_all]
        tuval_names_all = [tr[-n_val_files:] for tr in tr_names_all]
        tute_names_all = [te[1:] for te in te_names_all]
        keep_idxs = []
        for i_te, te in enumerate(tute_names_all):
            if type(te) != list:
                tute_names_all[i_te] = [tute_names_all[i_te]]
            if len(tute_names_all[i_te]) > 0:
                keep_idxs.append(i_te)
        tute_names_all = [val for i_val, val in enumerate(tute_names_all) if i_val in keep_idxs]
        tutr_names_all = [val for i_val, val in enumerate(tutr_names_all) if i_val in keep_idxs]
        tuval_names_all = [val for i_val, val in enumerate(tuval_names_all) if i_val in keep_idxs]
        tu_names_all = [val for i_val, val in enumerate(tu_names_all) if i_val in keep_idxs]
        print("Nr. tune folds:", len(keep_idxs))

        # Pad rows to equal length with commas, s.t. pandas can read them
        max_tr_len = max([len(n) for n in tr_names_all])
        max_te_len = max([len(n) for n in te_names_all])
        max_val_len = max([len(n) for n in val_names_all])
        max_tutr_len = max([len(n) for n in tutr_names_all])
        max_tuval_len = max([len(n) for n in tuval_names_all])
        max_tute_len = max([len(n) for n in tute_names_all])
        for fold in range(n_folds):
            tr_names_all[fold] += [""] * (max_tr_len - len(tr_names_all[fold]))
            val_names_all[fold] += [""] * (max_val_len - len(val_names_all[fold]))
            te_names_all[fold] += [""] * (max_te_len - len(te_names_all[fold]))
            if fold < len(keep_idxs):
                tu_names_all[fold] += [""] * (1 - len(tu_names_all[fold]))
                tutr_names_all[fold] += [""] * (max_tutr_len - len(tutr_names_all[fold]))
                tuval_names_all[fold] += [""] * (max_tuval_len - len(tuval_names_all[fold]))
                tute_names_all[fold] += [""] * (max_tute_len - len(tute_names_all[fold]))


        # Write each fold as a column
        for col in range(max_tr_len):
            tr_writer.writerow([f[col] for f in tr_names_all])
        for col in range(max_val_len):
            val_writer.writerow([f[col] for f in val_names_all])
        for col in range(max_te_len):
            te_writer.writerow([f[col] for f in te_names_all])

        tu_writer.writerow([f[0] for f in tu_names_all])
        for col in range(max_tutr_len):
            tutr_writer.writerow([f[col] for f in tutr_names_all])
        for col in range(max_tuval_len):
            tuval_writer.writerow([f[col] for f in tuval_names_all])
        for col in range(max_tute_len):
            tute_writer.writerow([f[col] for f in tute_names_all])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Cross validation splitter')
    parser.add_argument("--dataset", dest="dataset", default="sleepedf")
    parser.add_argument("--n_folds", dest="k", type=int, default=20)
    args = parser.parse_args()

    if args.dataset == "sleepedf":
        data_dir = "../data/sleepedf/"
        csv_dir = "sleepedf/"
        name_re = re.compile(r'SC4\d{3}E.-PSG')
        subj_re = re.compile(r'\d{4}')
    elif args.dataset == "caro":
        data_dir = "/cluster/scratch/llorenz/data/caro/"
        csv_dir = "../cfg/caro/"
        name_re = re.compile(r'(WESA_S\d{3}_N\d+S_MLready)|(WESA_S\d{3}_N\d+V_MLready)|(S\d{3}_msco\d_MLready)')
        subj_re = re.compile(r'\d{3}')
    elif args.dataset == "caro_new":
        data_dir = "/cluster/scratch/llorenz/data/caro_new/"
        csv_dir = "../cfg/caro_new/"
        name_re = re.compile(r'WESA_S\d{3}_N\d+S_MLready')
        subj_re = re.compile(r'\d{3}')
    split(data_dir, csv_dir, name_re, subj_re, args.k)

    print("Success")