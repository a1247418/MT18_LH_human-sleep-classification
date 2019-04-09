from os import listdir
from os.path import isfile, join
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
parser = argparse.ArgumentParser(prog='Sleep Learning cross validation')
parser.add_argument("--data_dir", dest="data_dir", default="/cluster/scratch/llorenz/data/caro_new/")


def print_distribtions(data_path):
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    distributions = {}
    totals = {}
    for file in files:
        name = file
        print(name)
        mat = loadmat(join(data_path, name), squeeze_me=True, struct_as_record=False)

        for expert in ["E1","E2","E3","E4","E5","CL"]:
            if f'sleepStage_score_{expert}' in mat:
                hypnogram = mat[f'sleepStage_score_{expert}']
                total = len(hypnogram)
                distrib = [0]*7
                for i in hypnogram:
                    distrib[i] += 1
                totals[name] = total
                distributions[name] = distrib
                break
        del mat

    labels = ["W","N1","N2","N3","N4","R","A"]
    for n in totals.keys():
        print(n, "\tTotal:", totals[n])
        print(", ".join([f"{labels[i]}:{distributions[n][i]/totals[n]:.2f}" for i in range(len(labels))]))

    return totals, distributions

if __name__ == '__main__':
    args = parser.parse_args()
    print_distribtions(args.data_dir)