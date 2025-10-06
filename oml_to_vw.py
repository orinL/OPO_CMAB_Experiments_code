#!/usr/bin/env python3
"""
OpenML to VowpalWabbit Dataset Converter - Final Release Version

This script downloads and converts specific OpenML datasets to VowpalWabbit format.
Download the 18  datasets used in the experiments.

Usage:
    python oml_to_vw.py <min_did> <max_did>
    
Example:
    python oml_to_vw.py 0 9999  # Downloads all 18 datasets
    python oml_to_vw.py 1006 1007  # Downloads only dataset 1006
"""

import argparse
# from config import OML_API_KEY
import gzip
import openml
import os
import scipy.sparse as sp
import numpy as np

VW_DS_DIR = 'datasets/'

def save_vw_dataset(X, y, did, ds_dir):
    n_classes = y.max() + 1
    fname = 'ds_{}_{}.vw.gz'.format(did, n_classes)
    with gzip.open(os.path.join(ds_dir, fname), 'w') as f:
        if sp.isspmatrix_csr(X):
            for i in range(X.shape[0]):
                f.write('{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(X[i].indices, X[i].data))).encode())
        else:
            for i in range(X.shape[0]):
                f.write('{} | {}\n'.format(y[i] + 1, ' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in enumerate(X[i]) if val != 0)).encode())
 

def shuffle(X, y):
    n = X.shape[0]
    perm = np.random.permutation(n)
    X_shuf = X[perm, :]
    y_shuf = y[perm]
    return X_shuf, y_shuf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openML to vw converter')
    parser.add_argument('min_did', type=int, default=0, help='minimum dataset id to process')
    parser.add_argument('max_did', type=int, default=None, help='maximum dataset id to process')
    args = parser.parse_args()
    print(args.min_did, ' to ', args.max_did)

    # openml.config.apikey = OML_API_KEY
    # Set cache directory for newer OpenML versions
    if hasattr(openml.config, 'set_cache_directory'):
        openml.config.set_cache_directory('omlcache/')
    else:
        openml.config.cache_directory = 'omlcache/'

    print('loaded openML')

    if not os.path.exists(VW_DS_DIR):
        os.makedirs(VW_DS_DIR)

    # Specific datasets for final release
    # Extracted from: ['ds_1006_2', 'ds_1012_2', 'ds_1015_2', 'ds_1062_2.0', 'ds_1073_2.0', 'ds_1084_3', 
    #                  'ds_339_3', 'ds_346_2', 'ds_457_4', 'ds_476_2', 'ds_729_2', 'ds_785_2', 
    #                  'ds_835_2', 'ds_848_2', 'ds_874_2', 'ds_905_2', 'ds_928_2', 'ds_964_2']
    final_release_dids = [339, 346, 457, 476, 729, 785, 835, 848, 874, 905, 928, 964, 1006, 1012, 1015, 1062, 1073, 1084]

    dids = final_release_dids

    for did in sorted(dids):
        if did < args.min_did:
            continue
        if args.max_did is not None and did >= args.max_did:
            break
        print('processing did', did)
        try:
            ds = openml.datasets.get_dataset(did)
            # 
            # 
            classes = ds.qualities['NumberOfClasses']
            print("Number of classes: ", classes)
            if classes == 0.0:
                print("Error: Dataset {} is not a classification dataset.".format(did))
                continue
            (X, y, _, _) = ds.get_data(target=ds.default_target_attribute,dataset_format='array')
            print(X.shape)
            print(y.shape)
            X, y = shuffle(X, y)
            # 
        except Exception as e:
            print(e)
            continue
        save_vw_dataset(X, y, did, VW_DS_DIR)
