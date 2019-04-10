import os
import sys
import argparse
root_dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, root_dir)
import shutil
import pandas as pd
from sacred.stflow import LogFileWriter
from sklearn.metrics import accuracy_score, confusion_matrix
from sleeplearning.lib.base import Base
from sleeplearning.lib.logger import Logger
from cfg.config import *
import numpy as np
parser = argparse.ArgumentParser(prog='Sleep Learning cross validation')
parser.add_argument("--data_dir", dest="fn", default=os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '..')) + '/logs/ ',
                        help="filepath of cv folds")

def get_basename_(path):
    return os.path.basename(os.path.normpath(path))

@ex.main
def cv(ds, arch, ms, cuda, log_dir, seed, _run):
    cuda = torch.cuda.is_available()

    output_dir = os.path.join(log_dir, get_basename_(ds['val_csv']))
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if _run._id is not None:
        # Capture TensorBoard logs with sacred
        with LogFileWriter(ex):
            logger = Logger(log_dir, _run)
    else:
        logger = Logger(log_dir, None)

    targets = np.array([], dtype=int)
    predictions = np.array([], dtype=int)
    clf = Base(logger=logger, cuda=cuda, verbose=True)

    print(ds['nfolds'])
    for i in range(ds['nfolds']):
        fold_accs = np.array([])
        checkpoint_path = os.path.join(log_dir, 'fold{}'.format(i),
                                       'checkpoint.pth.tar')
        clf.restore(checkpoint_dir=checkpoint_path)
        # reset fold id since we evaluate each subject independently
        clf.ds['fold'] = 0
        subjects = pd.read_csv(ds['val_csv'], header=None)[i].dropna().tolist()
        print(subjects)
        for subject in subjects:
            acc, probs, true = clf.score(os.path.join(ds['data_dir'], subject),
                              probs=True)
            pred = np.argmax(probs, 1)
            savedict = dict(subject=subject, acc=acc, probs=probs, y_pred=pred,
                            y_true=true)
            print("subject {}: {}".format(subject, acc))
            fold_accs = np.append(fold_accs, acc)
            np.savez(os.path.join(output_dir, subject), **savedict)
            targets = np.append(targets, true)
            predictions = np.append(predictions, pred)
        print("fold {}: {}".format(i, np.mean(fold_accs)))
    cm = confusion_matrix(targets, predictions)
    overall_acc = accuracy_score(targets, predictions)*100
    print("\n CONFUSION MATRIX ")
    print(cm)
    print("\n OVERALL ACCURACY ")
    print("{0:.2f}%".format(overall_acc))

    return overall_acc


if __name__ == '__main__':
    args = sys.argv
    args += ['--name', args[-1]]
    ex.run_commandline(argv=args)

