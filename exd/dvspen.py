#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : dvspen
# @Date : 05/19/2018 14:21:04
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score

import os
import time

class DVSPEN(object):
    def __init__(self, feature_dim, label_dim, num_hidden, num_pairwise, learning_rate, inf_lr, dropout, weight_decay):
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.num_hidden = num_hidden
        self.num_pairwise = num_pairwise
        self.learning_rate = learning_rate
        self.inf_lr = inf_lr
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.loss = 0.
        self.rmse = 0.
        self.current_step = 1

        self.build()

        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss))

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def build(self):
        with tf.name_scope('input'):
            self.plh_x = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='input_x') # 10x1
            self.plh_y = tf.placeholder(dtype=tf.float32, shape=[None, self.label_dim], name='input_y') # 1x10
            self.plh_gt_score = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input_gt_score') # 1x2
            # self.plh_dropout = tf.placeholder_with_default(0., shape=(), name='dropout')

        self.weights = []
        self.raw_pred_score = 0.
        with tf.name_scope('dvn'):
            with tf.name_scope('local_value'):
                with tf.name_scope('mlp_layers'):
                    name = 'mlp_hidden_layer'
                    mlp_hidden_output = self.build_mlp_layer(name, self.plh_x, self.feature_dim, self.num_hidden, act=tf.nn.softplus) # nx10, 10xh

                    name = 'mlp_output_layer'
                    self.mlp_output = self.build_mlp_layer(name, mlp_hidden_output, self.num_hidden, self.label_dim, act=lambda x: x) # nxh, hx10

                self.local_score = tf.reduce_sum(self.plh_y * self.mlp_output, axis=1) # nx10, nx10 -> n,
                self.raw_pred_score += self.local_score

            with tf.name_scope('label_value'):
                label_weight = tf.Variable(tf.random_normal(shape=[self.label_dim, self.num_pairwise], stddev=np.sqrt(2.0 / self.label_dim)), name='label_weight') # 10xh
                self.weights.append(label_weight)
                self.label_score = tf.nn.softplus(tf.matmul(self.plh_y, label_weight)) # nx10, 10xh -> nxh

                label_weight_2 = tf.Variable(tf.random_normal(shape=[self.num_pairwise, 1], stddev=np.sqrt(2.0 / self.num_pairwise)), name='label_weight_2') # hx1
                self.weights.append(label_weight_2)

                self.label_score = tf.squeeze(tf.matmul(self.label_score, label_weight_2)) # nxh, hx1 -> n,

                self.raw_pred_score += self.label_score

                self.pred_score = tf.sigmoid(self.raw_pred_score)

        with tf.name_scope('loss'):
            # regularization
            for weight in self.weights:
                self.loss += self.weight_decay * tf.nn.l2_loss(weight)

            self.loss += tf.nn.sigmoid_cross_entropy_with_logits(labels=self.plh_gt_score[:, 1], logits=self.raw_pred_score)

        with tf.name_scope('metrics'):
            with tf.name_scope('rmse'):
                square_error = tf.square(self.pred_score - self.plh_gt_score[:, 1])
                self.rmse = tf.sqrt(tf.reduce_mean(square_error))

        # self.adv_gradient = tf.gradients(self.loss, self.plh_y)[0] # why [0]?
        self.gradient = tf.gradients(self.pred_score, self.plh_y)[0]


    def build_mlp_layer(self, name, input_tensor, input_dim, output_dim, act=tf.nn.relu):
        with tf.name_scope(name):
            weight = self.glorot(shape=[input_dim, output_dim], name='weight')
            self.weights.append(weight)
            bias = tf.Variable(tf.zeros([output_dim]), name='bias')

            output = tf.matmul(input_tensor, weight) + bias

        return act(output)

    def train(self,dropout, train_features, train_labels, train_gt_scores, val_features, val_labels, val_gt_scores, batch_size=100, epochs=100):

        train_features = np.array(train_features, np.float32)
        self.mean = np.mean(train_features, axis=0).reshape((1, -1))
        self.std = np.std(train_features, axis=0).reshape((1, -1)) + 10 ** -6
        train_features -= self.mean
        train_features /= self.std

        for epoch in range(epochs):
            data_size = train_features.shape[0]
            for idx in range(0, data_size, batch_size):
                t = time.time()
                feature_batch = train_features[idx:min(idx + batch_size, data_size), :]
                label_batch = train_labels[idx:min(idx + batch_size, data_size), :]
                score_batch = train_gt_scores[idx:min(idx + batch_size, data_size), :]
                _, weights, loss = self.sess.run([self.train_step, self.weights, self.loss], feed_dict={
                    self.plh_x: feature_batch,
                    self.plh_y: label_batch,
                    self.plh_gt_score: score_batch
                    # self.plh_dropout: dropout
                })
                # print(weights[0])
                # print(weights[1])
                # print(weights[2])
                # print(weights[3])

                print('Step: {:6d}, traning loss: {:.5f}, running time: {:.5f}'.format(self.current_step, np.mean(loss), time.time() - t))

                self.current_step += 1

            # if epoch % 10 == 0 and epoch > 0:
            #     val_loss, val_rmse = self.sess.run([self.loss, self.rmse], feed_dict={
            #         self.plh_x: val_features,
            #         self.plh_y: val_labels,
            #         self.plh_gt_score: val_gt_scores,
            #     })
            #     pred_labels = self.inference(val_features, self.inf_lr, num_iteration=50)
            #
            #     val_f1 = self.metric_f1(val_labels, pred_labels)
            #     val_iou = self.metric_iou(val_labels, pred_labels)
            #
            #     print('Validation loss: {:.5f}, mean rmse: {:.5f}, mean f1: {:.5f}, mean iou: {:.5f}'.format(val_loss, np.mean(val_rmse), np.mean(val_f1), np.mean(val_iou)))



    def reduce_learning_rate(self, factor=0.5):
        self.learning_rate *= factor
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss))

    def inference(self, features, learning_rate=0., num_iteration=1):

        pred_labels = np.zeros((features.shape[0], self.label_dim))
        for iter in range(num_iteration):
            gradient = self.sess.run(self.gradient, feed_dict={
                self.plh_x: features,
                self.plh_y: pred_labels,
            })

            pred_labels += learning_rate * gradient

            # projection
            pred_labels[pred_labels < 0] = 0
            pred_labels[pred_labels > 1] = 1

        return pred_labels


    def predict(self, gt_lbl, pred_lbl):

        pass


    def metric_f1(self, gt_lbl, pred_lbl):
        gt_lbl = np.array(gt_lbl > 0.5, np.float32)
        pred_lbl = np.array(pred_lbl > 0.5, np.float32)

        intersection = np.sum(np.minimum(gt_lbl, pred_lbl), axis=1)
        union = np.maximum(np.sum(np.maximum(gt_lbl, pred_lbl), axis=1), 10e-8)

        return 2. * intersection / (intersection + union)

    def metric_iou(self, gt_lbl, pred_lbl):
        gt_lbl = np.array(gt_lbl > 0.5, np.float32)
        pred_lbl = np.array(pred_lbl > 0.5, np.float32)

        intersection = np.sum(np.minimum(gt_lbl, pred_lbl), axis=1)
        union = np.maximum(np.sum(np.maximum(gt_lbl, pred_lbl), axis=1), 10e-8)

        return intersection / union

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

def load_data(path, num_train, num_val, num_test, measurement='F1'):
    features_fn = 'features.npy'
    labels_fn = 'labels.npy'
    scores_fn = 'scores.npy'

    features = np.load(os.path.join(path, features_fn))
    labels = np.load(os.path.join(path, labels_fn))
    scores = np.load(os.path.join(path, scores_fn))

    train_features = features[:num_train, :]
    train_labels = labels[:num_train, :]

    val_features = features[num_train:num_train + num_val, :]
    val_labels = labels[num_train:num_train + num_val, :]

    test_features = features[num_train + num_val:num_train + num_val + num_test, :]
    test_labels = labels[num_train + num_val:num_train + num_val + num_test, :]

    if measurement == 'F1':
        train_scores = scores[:num_train, 0]
        train_scores = preprocess_score(train_scores)
        val_scores = scores[num_train:num_train + num_val, 0]
        val_scores = preprocess_score(val_scores)
        test_scores = scores[num_train + num_val:num_train + num_val + num_test, 0]
        test_scores = preprocess_score(test_scores)
    else:
        train_scores = scores[:num_train, 1]
        train_scores = preprocess_score(train_scores)
        val_scores = scores[num_train:num_train + num_val, 1]
        val_scores = preprocess_score(val_scores)
        test_scores = scores[num_train + num_val:num_train + num_val + num_test, 1]
        test_scores = preprocess_score(test_scores)

    return train_features, train_labels, train_scores, val_features, val_labels, val_scores, test_features, test_labels, test_scores

def preprocess_feature(features):
    pass

def preprocess_score(scores):
    exd_scores = np.zeros(shape=(scores.shape[0], 2))
    exd_scores[:, 1] = scores
    exd_scores[:, 0] = 1. - scores

    return exd_scores

if __name__ == '__main__':

    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
    flags.DEFINE_float('inf_lr', 0.5, 'Inference learning rate.')
    flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
    flags.DEFINE_integer('batch_size', 20, 'Batch size.')
    flags.DEFINE_integer('num_hidden', 16, 'Number of units in hidden layer')
    flags.DEFINE_integer('num_pairwise', 16, 'Number of units in pairwise layer')
    flags.DEFINE_integer('feature_dim', 10, 'Dimension of features.')
    flags.DEFINE_integer('label_dim', 10, 'Dimension of labels.')
    flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

    path = '/network/rit/lab/ceashpc/fjie/tmp/data/dvgcn'
    num_train = 10000
    num_val = 100
    num_test = 100

    train_features, train_labels, train_scores, val_features, val_labels, val_scores, test_features, test_labels, test_scores = load_data(path, num_train, num_val, num_test, measurement='F1')

    print(train_features[4289])
    print(train_labels[4289])
    print(train_scores[4289])

    model = DVSPEN(feature_dim=FLAGS.feature_dim, label_dim=FLAGS.label_dim, num_hidden=FLAGS.num_hidden, num_pairwise=FLAGS.num_pairwise, learning_rate=FLAGS.learning_rate, inf_lr=FLAGS.inf_lr, dropout=FLAGS.dropout, weight_decay=FLAGS.weight_decay)

    model.train(FLAGS.dropout, train_features, train_labels, train_scores, val_features, val_labels, val_scores, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)

    model.reduce_learning_rate(factor=0.1)

    model.train(FLAGS.dropout, train_features, train_labels, train_scores, val_features, val_labels, val_scores, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)

    model.reduce_learning_rate(factor=0.1)

    model.train(FLAGS.dropout, train_features, train_labels, train_scores, val_features, val_labels, val_scores, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
