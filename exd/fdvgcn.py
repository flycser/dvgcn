#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : dvgcn
# @Date : 05/19/2018 14:20:52
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score

import os
import time

class DVGCN(object):
    def __init__(self, num_nodes, feature_dim, label_dim, num_hidden, num_pairwise, learning_rate, inf_lr, dropout, weight_decay):
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.num_hidden = num_hidden
        self.num_pariwise = num_pairwise
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.inf_lr = inf_lr
        self.learning_rate = learning_rate

        self.loss = 0.
        self.rmse = 0.
        self.current_step = 1

        self.build()

        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def build(self):
        with tf.name_scope('input'):
            self.plh_x = tf.placeholder(dtype=tf.float32, shape=(self.num_nodes, self.feature_dim), name='input_x') # 10x1
            self.plh_y = tf.placeholder(dtype=tf.float32, shape=(1, self.num_nodes), name='input_y') # 1x10
            self.plh_gt_score = tf.placeholder(dtype=tf.float32, shape=(1, 2), name='input_gt_score') # 1x2
            self.plh_adj = tf.placeholder(dtype=tf.float32, shape=(self.num_nodes, self.num_nodes), name='adj') # 10x10
            self.plh_dropout = tf.placeholder_with_default(0., shape=(), name='dropout')

        self.weights = []
        self.raw_pred_score = 0.
        with tf.name_scope('dvn'):
            with tf.name_scope('local_energy'):
                with tf.name_scope('gcn_layers'):
                    name = 'gcn_hidden_layer'
                    gcn_hidden_output = self.build_gcn_layer(name, self.plh_x, self.feature_dim, self.num_hidden) # 10x10, 10x1, 1xh

                    name = 'gcn_output_layer'
                    self.gcn_output = self.build_gcn_layer(name, gcn_hidden_output, self.num_hidden, self.label_dim, act=lambda x: x) # 10x10, 10xh, hx1

                local_score = tf.matmul(self.plh_y, self.gcn_output) # 1x10, 10x1
                self.raw_pred_score += local_score

            with tf.name_scope('label_energy'):
                label_weight = tf.Variable(tf.random_normal((self.num_nodes, self.num_pariwise), stddev=np.sqrt(2.0/self.num_nodes)), name='label_weight') # 10xh
                self.weights.append(label_weight)
                self.label_score = tf.matmul(self.plh_y, label_weight) # 1x10, 10xh

                label_weight_2 = tf.Variable(tf.random_normal((self.num_pariwise, 1), stddev=np.sqrt(2.0/self.num_pariwise)), name='label_weight_2') # hx1
                self.weights.append(label_weight_2)

                self.label_score = tf.squeeze(tf.matmul(tf.nn.softplus(self.label_score), label_weight_2)) # 1xh, hx1

                self.raw_pred_score += self.label_score

            with tf.name_scope('energy'):
                self.pred_score = tf.sigmoid(self.raw_pred_score)

        with tf.name_scope('loss'):
            # regularization
            # for weight in self.weights:
            #     self.loss += self.weight_decay * tf.nn.l2_loss(weight)

            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.plh_gt_score[:, 1], shape=(1, 1)), logits=self.pred_score))

        with tf.name_scope('metrics'):
            with tf.name_scope('rmse'):
                square_error = tf.nn.l2_loss(self.pred_score - self.plh_gt_score)
                self.rmse = tf.sqrt(tf.reduce_mean(square_error))

        # self.adv_gradient = tf.gradients(self.loss, self.plh_y)[0] # why [0]?
        self.gradient = tf.gradients(self.pred_score, self.plh_y)[0]


    def build_gcn_layer(self, name, input_tensor, input_dim, output_dim, act=tf.nn.relu):
        with tf.name_scope(name):
            weight = self.glorot(shape=(input_dim, output_dim), name='weight')
            self.weights.append(weight)

            output = tf.matmul(tf.matmul(self.plh_adj, input_tensor), weight, name='AXW')

        return act(output)

    def train(self, adj, dropout, train_features, train_labels, train_gt_scores, val_features, val_labels, val_gt_scores, epochs):

        for epoch in range(epochs):
            t = time.time()


            features = train_features[epoch, :].reshape((-1, 1))
            labels = train_labels[epoch, :].reshape((1, -1))
            score = train_gt_scores[epoch].reshape((1, 2))
            _, loss = self.sess.run([self.train_step, self.loss], feed_dict={
                self.plh_x: features,
                self.plh_y: labels,
                self.plh_gt_score: score,
                self.plh_adj: adj,
                self.plh_dropout: dropout
            })

            print('Epoch: {:6d}, traning loss: {:.5f}, running time: {:.5f}'.format(self.current_step, np.mean(loss), time.time() - t))

            self.current_step += 1

            if epoch % 1000 == 0 and epoch > 0:
                val_rmse = []
                val_f1 = []
                val_iou = []
                for i in range(val_features.shape[0]):
                    features = val_features[i, :].reshape((-1, 1))
                    labels = val_labels[i, :].reshape((1, -1))
                    score = val_gt_scores[i].reshape((1, 2))
                    rmse = self.sess.run(self.rmse, feed_dict={
                        self.plh_x: features,
                        self.plh_y: labels,
                        self.plh_gt_score: score,
                        self.plh_adj: adj
                    })
                    # pred_labels = self.inference(adj, features, self.inf_lr, num_iteration=50)

                    # f1 = self.metric_f1(labels, pred_labels)
                    # iou = self.metric_iou(labels, pred_labels)
                    # val_f1.append(f1)
                    # val_iou.append(iou)

                    val_rmse.append(rmse)

                print('Validation mean rmse: {:.5f}, mean f1: {:.5f}, mean iou: {:.5f}'.format(np.mean(val_rmse), np.mean(val_f1), np.mean(val_iou)))



    def reduce_learning_rate(self, factor=0.5):
        self.learning_rate *= factor
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def inference(self, adj, features, learning_rate=0., num_iteration=1):

        pred_labels = np.zeros((1, self.num_nodes))
        for iter in range(num_iteration):
            gradient = self.sess.run(self.gradient, feed_dict={
                self.plh_x: features,
                self.plh_y: pred_labels,
                self.plh_adj: adj
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

        intersection = np.sum(np.minimum(gt_lbl, pred_lbl))
        union = max(np.sum(np.maximum(gt_lbl, pred_lbl)), 10e-8)

        return 2. * intersection / (intersection + union)

    def metric_iou(self, gt_lbl, pred_lbl):
        gt_lbl = np.array(gt_lbl > 0.5, np.float32)
        pred_lbl = np.array(pred_lbl > 0.5, np.float32)

        intersection = np.sum(np.minimum(gt_lbl, pred_lbl))
        union = max(np.sum(np.maximum(gt_lbl, pred_lbl)), 10e-8)

        return intersection / union

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

def load_data(path, num_train, num_val, num_test, measurement='F1'):
    adj_fn = 'adj.npy'
    features_fn = 'features.npy'
    labels_fn = 'labels.npy'
    scores_fn = 'scores.npy'

    adj = np.load(os.path.join(path, adj_fn))
    adj = preprocess_adj(adj)
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

    return adj, train_features, train_labels, train_scores, val_features, val_labels, val_scores, test_features, test_labels, test_scores

def preprocess_adj(adj):
    adj = adj + np.identity(adj.shape[0])
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.divide(1., np.sqrt(degree))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_mat = np.diag(d_inv_sqrt)

    return np.matmul(np.matmul(d_inv_sqrt_mat, adj), d_inv_sqrt_mat)

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
    # flags.DEFINE_string('model_dir', tmp_dir + 'log2', 'Directory for model')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_float('inf_lr', 2.5, 'Inference learning rate.')
    flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
    flags.DEFINE_integer('num_hidden', 16, 'Number of units in hidden layer')
    flags.DEFINE_integer('num_pairwise', 16, 'Number of units in pairwise layer')
    flags.DEFINE_integer('num_nodes', 10, 'Number of nodes')
    flags.DEFINE_integer('feature_dim', 1, 'Dimension of features.')
    flags.DEFINE_integer('label_dim', 1, 'Dimension of labels.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')

    path = '/network/rit/lab/ceashpc/fjie/tmp/data/dvgcn/'
    num_train = 10000
    num_val = 100
    num_test = 100
    adj, train_x, train_y, train_v, val_x, val_y, val_v, test_x, test_y, test_v = load_data(path, num_train, num_val, num_test, measurement='F1')

    # print(adj)
    # print(train_x[4289])
    # print(train_y[4289])
    # print(train_v[4289])

    model = DVGCN(num_nodes=FLAGS.num_nodes, feature_dim=FLAGS.feature_dim, label_dim=FLAGS.label_dim, num_hidden=FLAGS.num_hidden, num_pairwise=FLAGS.num_pairwise, learning_rate=FLAGS.learning_rate, inf_lr=FLAGS.inf_lr, dropout=FLAGS.dropout, weight_decay=FLAGS.weight_decay)

    model.train(adj, FLAGS.dropout, train_x, train_y, train_v, val_x, val_y, val_v, FLAGS.epochs)

    model.reduce_learning_rate(factor=0.5)

    model.train(adj, FLAGS.dropout, train_x, train_y, train_v, val_x, val_y, val_v, FLAGS.epochs)

    model.reduce_learning_rate(factor=0.5)

    model.train(adj, FLAGS.dropout, train_x, train_y, train_v, val_x, val_y, val_v, FLAGS.epochs)


