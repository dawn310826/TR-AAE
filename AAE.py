#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
#from tensorflow.contrib.layers import apply_regularization, l2_regularizer

class AAE(object):
    def __init__(self, encoder_dims, beta1=0.9, lr=1e-3, lam=0.01, random_seed=None):
        self.encoder_dims = encoder_dims
        self.decoder_dims = encoder_dims[::-1]
        self.dims = self.encoder_dims + self.decoder_dims[1:]
        self.lr = lr
        self.lam = lam
        self.beta1 = beta1 
        self.random_seed = random_seed
        self.construct_placeholders()

    def construct_placeholders(self):
        # Placeholders for input data and the targets
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.encoder_dims[0]], name='Input')
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, self.encoder_dims[-1]], name='Real_distribution')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def dense(self, x, n1, n2, name):
        with tf.variable_scope(name, reuse=None):
            weights = tf.get_variable("weights", shape=[n1, n2],
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
                                      #initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))
            bias = tf.get_variable("bias", shape=[n2],
                                   initializer=tf.truncated_normal_initializer(stddev=0.001, seed=self.random_seed))
            
            out = tf.add(tf.matmul(x, weights), bias, name='matmul')
            return out

    def corrupt(self, x):
        x *= 1.0 / self.keep_prob_ph
        x = tf.nn.dropout(x, self.keep_prob_ph)
        return x

	# The autoencoder network
    def encoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Encoder'):
            x = tf.nn.l2_normalize(x, 1)
            x = self.corrupt(x)

            for i in range(len(self.encoder_dims) - 2):
                x = tf.nn.relu(self.dense(x, self.encoder_dims[i], self.encoder_dims[i+1], 'e_dense_'+str(i + 1)))	
            latent_variable = self.dense(x, self.encoder_dims[-2], self.encoder_dims[-1], 'e_latent_variable')
            return latent_variable

    def decoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            for i in range(len(self.decoder_dims) - 2):
                x = tf.nn.relu(self.dense(x, self.decoder_dims[i], self.decoder_dims[i+1], 'd_dense_'+str(i + 1)))
            output = self.dense(x, self.decoder_dims[-2], self.decoder_dims[-1], 'd_output')
            return output

    def discriminator(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Discriminator'):
            dc_den1 = tf.nn.relu(self.dense(x, self.encoder_dims[-1], 200, name='dc_den1'))
            dc_den2 = tf.nn.relu(self.dense(dc_den1, 200, 100, name='dc_den2'))
            dc_den3 = tf.nn.relu(self.dense(dc_den2, 100, 20, name='dc_den3'))
            output = tf.nn.softmax(self.dense(dc_den3, 20, 1, name='dc_output'))
            return output
        
    def build_graph(self):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            encoder_output = self.encoder(self.x_input)
            decoder_output = self.decoder(encoder_output)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            d_real = self.discriminator(self.real_distribution)
            d_fake = self.discriminator(encoder_output, reuse=True)
        
        saver = tf.train.Saver()
        all_variables = tf.trainable_variables()    
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]
        #de_var = [var for var in all_variables if 'de' in var.name]

        # Autoencoder loss
        #autoencoder_loss = tf.reduce_mean(tf.square(self.x_input - decoder_output))
        log_softmax_var = tf.nn.log_softmax(decoder_output)
        autoencoder_loss = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * self.x_input, axis=-1))
        
        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        dc_loss = dc_loss_fake + dc_loss_real
        
        # Generator loss
        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        # Optimizers
        autoencoder_op= tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(autoencoder_loss)
        discriminator_op = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.beta1).minimize(dc_loss, var_list=dc_var)
        generator_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(generator_loss, var_list=en_var)
        
        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
        tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
        tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
        merged = tf.summary.merge_all()

        return saver, decoder_output, [autoencoder_loss, dc_loss, generator_loss], [autoencoder_op, discriminator_op, generator_op], merged
