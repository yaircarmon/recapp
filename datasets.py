"""
Wrapper for the libsvmdata package, which downloads and reads LIBSVM data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
"""
import libsvmdata
from pathlib import Path
from sklearn.preprocessing import normalize
import os
import sys


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_data(dataset_name, n=-1, root=None):
    if root is not None:
        libsvmdata.datasets.DATA_HOME = Path(root) / 'libsvm'

    with SuppressPrint():
        X, y = libsvmdata.fetch_libsvm(dataset_name)

    # note: this is the preprocesing done in the Catalyst paper for covertype. However, the features for covertype
    # appear to be slightly different in the libsvm repository compared to files in https://github.com/hongzhoulin89/Catalyst-QNing/tree/master/catalyst_v1/data
    # It is possible to get the exact feartures from that repo using using sklearn.datasets.fetch_covtype.

    X = normalize(X, axis=1)

    if n >= 0:
        X, y = X[:n], y[:n]

    return X, y

