#!/usr/bin/env python
# coding=utf-8
import os
import shutil
import argparse
import numpy as np

from AAE import AAE
import tensorflow as tf

from metrics import Recall_at_k_batch
from metrics import NDCG_binary_at_k_batch
from metrics import Precision_at_k_batch
from metrics import MAP_at_k_batch

from load_data import load_pro_data
from load_data import load_train_data
from load_data import load_tr_te_data

####################################################
##Load args
####################################################
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", type=int, default=100, help="train epochs")
parser.add_argument("-d", "--dropout", type=float, default=0.5, help="dropout")
parser.add_argument("-t", "--train_or_predict", type=bool, default=True, help="train_or_predict")
parser.add_argument("-l", "--layer1", type=int, default=1000)
parser.add_argument("-ll", "--layer2", type=int, default=200)
args = parser.parse_args()

####################################################
##Load data and set up training hyperparameters
####################################################
DATA_DIR = './data/'
pro_dir = os.path.join(DATA_DIR, 'pro_sg_tag')
unique_sid = load_pro_data(os.path.join(pro_dir, 'unique_sid.txt'))
n_tags = len(unique_sid)
train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_tags)
vad_data_tr, vad_data_te = load_tr_te_data(
		os.path.join(pro_dir, 'validation_tr.csv'),
		os.path.join(pro_dir, 'validation_te.csv'), n_tags)
test_data_tr, test_data_te = load_tr_te_data(
		os.path.join(pro_dir, 'test_tr.csv'),
		os.path.join(pro_dir, 'test_te.csv'), n_tags)

N = train_data.shape[0]
N_vad = vad_data_tr.shape[0]
N_test = test_data_tr.shape[0]

idxlist = list(range(N))
idxlist_vad = range(N_vad)
idxlist_test = range(N_test)

batch_size = 500
batch_size_vad = 2000
batch_size_test = 2000
batches_per_epoch = int(np.ceil(float(N) / batch_size))

####################################################
##Train a AAE
####################################################

dims = [n_tags, args.layer1, args.layer2]
tf.reset_default_graph()
aae = AAE(dims, random_seed=98765)
saver, reconstruct_x, loss_vars, train_op_vars, merged_var = aae.build_graph()
autoencoder_loss, dc_loss, generator_loss = loss_vars
autoencoder_op, discriminator_op, generator_op = train_op_vars

recall_var = tf.Variable(0.0)
recall_summary = tf.summary.scalar('recall_at_k_validation', recall_var)

####################################################
##Set up logging and checkpoint directory
####################################################
arch_str = "I-%s-I" % ('-'.join([str(d) for d in aae.dims[1:-1]]))
log_dir = './log/AAE_{}'.format(arch_str)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
chkpt_dir = './chkpt/AAE_{}'.format(arch_str)
if not os.path.isdir(chkpt_dir):
    os.makedirs(chkpt_dir) 
print("chkpt directory: %s" % chkpt_dir)

def train():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        best_recall = -np.inf
        update_count = 0.0
        
        for epoch in range(args.epoch):
            print("------------------Epoch {}/{}------------------".format(epoch, args.epoch))
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                X = train_data[idxlist[st_idx:end_idx]]
                X = X.toarray().astype('float32')    
                
                z_real_dist = np.random.randn(batch_size, dims[-1]) * 1.

                feed_dict = {aae.x_input: X, 
                             aae.keep_prob_ph: args.dropout, 
                             aae.real_distribution: z_real_dist}
                sess.run(autoencoder_op, feed_dict=feed_dict)
                sess.run(discriminator_op, feed_dict=feed_dict)
                sess.run(generator_op, feed_dict=feed_dict)
                
                if bnum % 100 == 0:
                    a_loss, d_loss, g_loss, summary = sess.run(
                            [autoencoder_loss, dc_loss, generator_loss, merged_var], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, global_step=epoch * batches_per_epoch + bnum)
                update_count += 1

            recall_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]
                X = X.toarray().astype('float32')  
            
                pred_val = sess.run(reconstruct_x, feed_dict={aae.x_input: X})
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                recall_dist.append(Recall_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))
            
            recall_dist = np.concatenate(recall_dist)
            recall_ = recall_dist.mean()
            merged_valid_val = sess.run(recall_summary, feed_dict={recall_var: recall_})
            summary_writer.add_summary(merged_valid_val, epoch)
            
            print(recall_)
            # update the best model (if necessary)
            if recall_ > best_recall:
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_recall = recall_
    ##Compute test metrics
    predict()
    
def predict():
    ####################################################
    ##Compute test metrics
    ####################################################
    tf.reset_default_graph()
    aae = AAE(dims)
    saver, reconstruct_x, _, _, _ = aae.build_graph()  
   
    k_list = range(1, 21)
    r_list = [[] for _ in range(len(k_list))]
    n_list = [[] for _ in range(len(k_list))]
    p_list = [[] for _ in range(len(k_list))]
    m_list = [[] for _ in range(len(k_list))]
    
    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))
    
        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]
            X = X.toarray().astype('float32')
    
            pred_val = sess.run(reconstruct_x, feed_dict={aae.x_input: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            for i in range(len(k_list)):
                r_list[i].append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=k_list[i]))
                n_list[i].append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=k_list[i]))
                p_list[i].append(Precision_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=k_list[i]))
                m_list[i].append(MAP_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=k_list[i]))

    for i in range(len(k_list)):
        r = np.concatenate(r_list[i]).mean()
        print("Test Recall@%d=%.5f" % (k_list[i], r))
        
    for i in range(len(k_list)):
        n = np.concatenate(n_list[i]).mean()
        print("Test NDCG@%d=%.5f" % (k_list[i], n))
        
    for i in range(len(k_list)):
        p = np.concatenate(p_list[i]).mean()
        print("Test Precision@%d=%.5f" % (k_list[i], p))
        
    for i in range(len(k_list)):
        m = np.concatenate(m_list[i]).mean()
        print("Test MAP@%d=%.5f" % (k_list[i], m))
        
if __name__ == '__main__':
    if args.train_or_predict:
        train()
    else:
        predict()