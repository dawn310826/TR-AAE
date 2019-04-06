#!/usr/bin/env python
# coding=utf-8
import numpy as np
import bottleneck as bn
#Evaluate function: Normalized discounted cumulative gain (NDCG@k) and Recall@k
def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)]) + 0.0001
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / (np.minimum(k, X_true_binary.sum(axis=1)) + 0.0001)
    return recall

def Precision_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    precision = tmp / (np.minimum(k, X_pred_binary.sum(axis=1)) + 0.0001)
    return precision

def MAP_at_k_batch(X_pred, heldout_batch, k=10):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = np.zeros_like(batch_users, dtype=float)
    
    for i in range(1, k+1):
        rel = np.zeros_like(batch_users, dtype=int)
        for user in range(batch_users):
            if X_true_binary[user , idx[user, k-1]]:
                rel[user] = 1
        print(rel.shape)
        r = Precision_at_k_batch(X_pred, heldout_batch, i) * rel
        tmp = tmp + r
    
    Map = tmp / (np.minimum(k, X_true_binary.sum(axis=1)) + 0.0001)
    return Map