#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : mlp
# @Date : 05/19/2018 14:20:36
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
import time


class MLP(object):
    def __init__(self, feature_dim, label_dim, num_hidden, learning_rate, dropout, weight_decay):

        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.loss = 0.
        self.current_step = 1

        self.build()

        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss))

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()


    def build(self):

        with tf.name_scope('input'):
            self.plh_x = tf.placeholder(dtype=np.float32, shape=(None, self.feature_dim), name='input_x')
            self.plh_y = tf.placeholder(dtype=np.float32, shape=(None, self.label_dim), name='input_y')
            self.plh_dropout = tf.placeholder_with_default(0., shape=(), name='dropout')

        self.weights = []
        with tf.name_scope('mlp'):
            name = 'hidden_layer'
            hidden_output = self.build_hidden_layer(name, self.plh_x, self.feature_dim, self.num_hidden)

            name = 'output_layer'
            self.logits = self.build_hidden_layer(name, hidden_output, self.num_hidden, self.label_dim, lambda x: x)

            self.pred_y = tf.sigmoid(self.logits, name='pred_y')

            with tf.name_scope('loss'):
                for weight in self.weights:
                    self.loss += self.weight_decay * tf.nn.l2_loss(weight)

                self.loss += tf.nn.sigmoid_cross_entropy_with_logits(labels=self.plh_y, logits=self.logits)

        with tf.name_scope('metrics'):
            self.f1 = self.metric_f1(self.plh_y, self.pred_y)
            self.iou = self.metric_iou(self.plh_y, self.pred_y)

    def build_hidden_layer(self, name, input_tensor, input_dim, output_dim, act=tf.nn.relu):

        with tf.name_scope(name):

            dropout_input = tf.nn.dropout(input_tensor, 1-self.plh_dropout)

            initial = tf.truncated_normal((input_dim, output_dim), stddev=0.1)
            weight = tf.Variable(initial, name='weight')
            self.weights.append(weight)

            initial = tf.constant(0.1, shape=(output_dim,))
            bias = tf.Variable(initial, name='bias')

            output = tf.matmul(dropout_input, weight) + bias

        return act(output)


    def metric_f1(self, labels, pred_labels):
        with tf.name_scope('f1_score'):
            threshold = tf.ones_like(labels) * 0.5
            bin_lbl = tf.cast(tf.greater(labels, threshold), tf.float32)
            bin_pred_lbl = tf.cast(tf.greater(pred_labels, threshold), tf.float32)
            intersection = tf.reduce_sum(tf.minimum(bin_lbl, bin_pred_lbl), axis=1)
            union = tf.maximum(tf.reduce_sum(tf.maximum(bin_lbl, bin_pred_lbl), axis=1), 10 ** -8)

            f1 = tf.divide(2 * intersection, intersection + union)

        return f1

    def metric_iou(self, labels, pred_labels):
        with tf.name_scope('iou_score'):

            threshold = tf.ones_like(labels) * 0.5
            bin_lbl = tf.cast(tf.greater(labels, threshold), tf.float32)
            bin_pred_lbl = tf.cast(tf.greater(pred_labels, threshold), tf.float32)
            intersection = tf.reduce_sum(tf.minimum(bin_lbl, bin_pred_lbl), axis=1)
            union = tf.maximum(tf.reduce_sum(tf.maximum(bin_lbl, bin_pred_lbl), axis=1), 10**-8)

            iou = tf.divide(intersection, union)

        return iou

    def reduce_learning_rate(self, factor=0.5):

        self.learning_rate *= factor
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss))


    def train(self, train_features, train_labels, val_features, val_labels, batch_size, epochs):


        data_size = train_features.shape[0]

        for epoch in range(epochs):
            for batch_id in range(0, data_size, batch_size):

                t = time.time()

                batch_features = train_features[batch_id:min(batch_id+batch_size, data_size), :]
                batch_labels = train_labels[batch_id:min(batch_id+batch_size, data_size), :]

                train_loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                    self.plh_x: batch_features,
                    self.plh_y:
                    batch_labels,
                    self.plh_dropout: self.dropout
                })

                # validation
                val_loss, f1, iou = self.sess.run([self.loss, self.f1, self.iou], feed_dict={
                    self.plh_x: val_features,
                    self.plh_y: val_labels
                })

                print('Epoch %d, step %d, train loss: %.5f, val loss: %.5f, val f1: %.5f, val iou: %.5f, running time: %.5f.' % (epoch, self.current_step, np.mean(train_loss), np.mean(val_loss), np.mean(f1), np.mean(iou), time.time() - t))
                self.current_step += 1

    def predict(self, test_features, test_labels):

        pred_lbls, f1, iou = self.sess.run([self.pred_y, self.f1, self.iou], feed_dict={
            self.plh_x: test_features,
            self.plh_y: test_labels
        })

        # np.set_printoptions(threshold=np.nan)
        # for i in range(pred_lbls.shape[0]):
        #     print(pred_lbls[i])
        #     print(test_labels[i])
        #     print(f1[i])
        #     print(iou[i])

        print('Predict mean f1 %.5f, mean iou: %.5f' % (np.mean(f1), np.mean(iou)))


def load_data(path, num_train, num_val, num_test):

    adj_fn = 'adj.npy'
    features_fn = 'features.npy'
    labels_fn = 'labels.npy'

    adj = np.load(os.path.join(path, adj_fn))
    features = np.load(os.path.join(path, features_fn))
    labels = np.load(os.path.join(path, labels_fn))


    train_features = features[:num_train, :]
    train_labels = labels[:num_train, :]

    val_features = features[num_train:num_train+num_val, :]
    val_labels = labels[num_train:num_train+num_val, :]

    test_features = features[num_train+num_val:num_train+num_val+num_test, :]
    test_labels = labels[num_train+num_val:num_train+num_val+num_test, :]

    return adj, train_features, train_labels, val_features, val_labels, test_features, test_labels


if __name__ == '__main__':

    path = '/network/rit/lab/ceashpc/fjie/tmp/data/sup1'

    num_train = 10000
    num_val = 100
    num_test = 100

    adj, train_features, train_labels, val_features, val_labels, test_features, test_labels = load_data(path, num_train, num_val, num_test)

    feature_dim = 10
    label_dim = 10
    num_hidden = 16
    learning_rate = 0.01
    dropout = 0.3
    weight_decay = 5e-4
    batch_size = 100
    epochs = 1000


    model = MLP(feature_dim=feature_dim, label_dim=label_dim, num_hidden=num_hidden, learning_rate=learning_rate, dropout=dropout, weight_decay=weight_decay)

    model.train(train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels, batch_size=batch_size, epochs=epochs)

    model.predict(test_features=test_features, test_labels=test_labels)