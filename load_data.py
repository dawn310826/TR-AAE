#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from scipy import sparse
import numpy as np

def load_pro_data(unique_id_path):
    unique_id = list()
    with open(unique_id_path, 'r') as f:
        for line in f:
            unique_id.append(line.strip())
    return unique_id


def load_train_data(csv_file, n_tags):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, item_cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, item_cols)), 
							dtype='float64',shape=(n_users, n_tags))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te, n_tags):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), 
							dtype='float64', shape=(end_idx - start_idx + 1, n_tags))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), 
							dtype='float64', shape=(end_idx - start_idx + 1, n_tags))
    return data_tr, data_te 
