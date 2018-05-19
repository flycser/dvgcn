#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : gcn
# @Date : 05/19/2018 14:20:44
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
import time


class GCN(object):

    def __init__(self, num_nodes, feature_dim, label_dim, num_hidden, learning_rate, dropout, weight_decay):

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.loss = 0.
        self.current_step = 1

        self.build()

        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()


    def build(self):

        with tf.name_scope('input'):
            self.plh_x = tf.placeholder(shape=(self.num_nodes, self.feature_dim), dtype=tf.float32)
            self.plh_y = tf.placeholder(shape=(self.num_nodes, self.label_dim), dtype=tf.float32)
            self.plh_adj = tf.placeholder(dtype=tf.float32)
            self.plh_dropout = tf.placeholder_with_default(0., shape=())

        self.weights = []
        with tf.name_scope('gcn'):
            name = 'gcn_hidden_layer'
            hidden_output = self.build_gcn_layer(name, self.plh_x, self.feature_dim, self.num_hidden, act=tf.nn.relu)

            name = 'gcn_output_layer'
            self.logits = self.build_gcn_layer(name, hidden_output, self.num_hidden, self.label_dim, act=lambda x: x)

            self.pred_y = tf.sigmoid(self.logits, name='pred_y')

            with tf.name_scope('loss'):
                # regularization
                for weight in self.weights:
                    self.loss += self.weight_decay * tf.nn.l2_loss(weight)

                self.loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.plh_y)

        with tf.name_scope('metrics'):
            self.f1 = self.metric_f1(labels=self.plh_y, pred_labels=self.pred_y)
            self.iou = self.metric_iou(labels=self.plh_y, pred_labels=self.pred_y)


    def build_gcn_layer(self, name, input_tensor, input_dim, output_dim, act=tf.nn.relu):

        with tf.name_scope(name):

            dropout_input = tf.nn.dropout(input_tensor, 1 - self.plh_dropout)

            weight = self.glorot(shape=(input_dim, output_dim), name='weight')
            self.weights.append(weight)

            if name == 'gcn_hidden_layer':
                bias = self.zeros(shape=(output_dim), name='bias')
            else:
                bias = self.zeros(shape=(num_nodes, 1), name='bias')

            gcn_output = tf.matmul(tf.matmul(self.plh_adj, dropout_input), weight) + bias
            # gcn_output = tf.matmul(tf.matmul(self.plh_adj, dropout_input), weight)

        return act(gcn_output)

    def metric_f1(self, labels, pred_labels):
        with tf.name_scope('f1_score'):
            threshold = tf.ones_like(labels) * 0.5
            bin_lbl = tf.cast(tf.greater(labels, threshold), tf.float32)
            bin_pred_lbl = tf.cast(tf.greater(pred_labels, threshold), tf.float32)
            intersection = tf.reduce_sum(tf.minimum(bin_lbl, bin_pred_lbl), axis=0)
            union = tf.maximum(tf.reduce_sum(tf.maximum(bin_lbl, bin_pred_lbl), axis=0), 10 ** -8)
            f1 = tf.divide(2 * intersection, intersection + union)

        return f1

    def metric_iou(self, labels, pred_labels):
        with tf.name_scope('iou_score'):

            threshold = tf.ones_like(labels) * 0.5
            bin_lbl = tf.cast(tf.greater(labels, threshold), tf.float32)
            bin_pred_lbl = tf.cast(tf.greater(pred_labels, threshold), tf.float32)
            intersection = tf.reduce_sum(tf.minimum(bin_lbl, bin_pred_lbl), axis=0)
            union = tf.maximum(tf.reduce_sum(tf.maximum(bin_lbl, bin_pred_lbl), axis=0), 10**-8)

            iou = tf.divide(intersection, union)

        return iou

    def reduce_learning_rate(self, factor=0.5):
        self.learning_rate *= factor
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, adj, num_nodes, train_features, train_labels, val_features, val_labels, epochs):

        # try full setting firstly
        for epoch in range(epochs):
            t = time.time()

            features = train_features[epoch, :].reshape(num_nodes, -1)
            labels = train_labels[epoch, :].reshape(num_nodes, -1)

            pred_y, train_loss, _ = self.sess.run([self.pred_y, self.loss, self.train_step], feed_dict={
                self.plh_x: features,
                self.plh_y: labels,
                self.plh_adj: adj,
                self.plh_dropout: self.dropout
            })

            # print(pred_y.reshape(1, -1))
            # print(train_labels[epoch])

            # validation
            val_loss = []
            val_f1 = []
            val_iou = []
            for i in range(val_labels.shape[0]):
                features = val_features[i, :].reshape(num_nodes, -1)
                labels = val_labels[i, :].reshape(num_nodes, -1)

                loss, f1, iou = self.sess.run([self.loss, self.f1, self.iou], feed_dict={
                    self.plh_x: features,
                    self.plh_y: labels,
                    self.plh_adj: adj,
                })

                # x = self.sess.run(self.weights[1])
                # print(x)

                val_loss.append(np.mean(loss))
                val_f1.append(np.mean(f1))
                val_iou.append(np.mean(iou))

            print('Step %d, train loss: %.5f, val loss: %.5f, val f1: %.5f, val iou: %.5f, running time: %.5f.' % (self.current_step, np.mean(train_loss), np.mean(val_loss), np.mean(val_f1), np.mean(val_iou), time.time() - t))

            self.current_step += 1

    def predict(self, adj, test_features, test_labels):

        test_f1 = []
        test_iou = []

        for i in range(test_features.shape[0]):
            features = test_features[i, :].reshape(num_nodes, -1)
            labels = test_labels[i, :].reshape(num_nodes, -1)

            pred_lbl, f1, iou = self.sess.run([self.pred_y, self.f1, self.iou], feed_dict={
                self.plh_x: features,
                self.plh_y: labels,
                self.plh_adj: adj
            })
            test_f1.append(f1)
            test_iou.append(iou)

            print(features.reshape(1, -1))
            print(labels.reshape(1, -1))
            print(pred_lbl.reshape(1, -1))

        print('Predict mean f1 %.5f, mean iou: %.5f' % (np.mean(test_f1), np.mean(test_iou)))


    def glorot(self, shape, name=None):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        # initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        initial = tf.truncated_normal(shape, stddev=0.1)

        return tf.Variable(initial, name=None)

    def zeros(self, shape, name=None):
        """All zeros."""
        initial = tf.zeros(shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)


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

def preprocess_adj(adj):

    adj = adj + np.identity(adj.shape[0])
    # print('original adjacent matrix', adj)
    degree = np.sum(adj, axis=1)
    # print('degree vector', degree)
    d_inv_sqrt = np.divide(1., np.sqrt(degree))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # print('inv sqrt degree vector', d_inv_sqrt)
    d_inv_sqrt_mat = np.diag(d_inv_sqrt)

    return np.matmul(np.matmul(d_inv_sqrt_mat, adj), d_inv_sqrt_mat)

def preprocess_features(features):

    # row_sum = np.sum(features, axis=1)
    # r_inv = np.power(row_sum, -1)
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = np.diag(r_inv)
    # features = r_mat_inv.dot(features)

    # print(features[0, :])

    return features

if __name__ == '__main__':

    path = '/network/rit/lab/ceashpc/fjie/tmp/data/sup1/'

    num_train = 10000
    num_val = 100
    num_test = 100

    num_nodes = 10
    feature_dim = 1
    label_dim = 1
    num_hidden = 16
    learning_rate = 0.01
    dropout = 0.3
    weight_decay = 5e-4
    epochs = 10000

    adj, train_features, train_labels, val_features, val_labels, test_features, test_labels = load_data(path, num_train, num_val, num_test)

    # I did not preprocess features here !!!
    adj = preprocess_adj(adj)
    train_features = preprocess_features(train_features)
    val_features = preprocess_features(val_features)
    test_features = preprocess_features(test_features)


    # print(adj)
    # print(train_features[23])
    # print(train_labels[23])


    model = GCN(num_nodes=num_nodes, feature_dim=feature_dim, label_dim=label_dim, num_hidden=num_nodes, learning_rate=learning_rate, dropout=dropout, weight_decay=weight_decay)

    for i in range(3):
        model.train(adj=adj, num_nodes=num_nodes,train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels, epochs=epochs)

        # model.reduce_learning_rate(factor=0.9)

    # model.train(adj=adj, num_nodes=num_nodes, train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels, epochs=epochs)

    model.predict(adj=adj, test_features=test_features, test_labels=test_labels)


