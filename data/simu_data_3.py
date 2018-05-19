#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : simu_data_3
# @Date : 05/19/2018 14:20:15
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

import numpy as np
import networkx as nx

import os

def generate_data(path):
    np.random.seed(123)

    num_nodes = 10
    mean_0 = 1.  # not subgraph
    std_0 = 1.
    mean_1 = 5.  # subgraph
    std_1 = 1.
    size = 50000
    ratio = 0.5
    num_change = int(num_nodes * ratio)

    adj_fn = 'adj'  # adjacent matrix file name
    feature_fn = 'features'  # features file name
    label_fn = 'labels'  # labels file name
    score_fn = 'scores'

    subgraph = [5, 6, 7, 8, 9]

    graph = nx.Graph()
    graph.add_nodes_from([node for node in range(num_nodes)])
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 3), (3, 4), (3, 7), (5, 6), (5, 8), (6, 7), (6, 8), (6, 9), (7, 9),
         (8, 9)])

    gt_lbl = np.zeros(num_nodes)
    gt_lbl[subgraph] = 1

    adj_mat = nx.adjacency_matrix(graph).todense()
    # print nx.adjacency_matrix(graph)
    feature_mat = np.zeros((size, num_nodes), dtype=np.float32)  # one feature per node
    label_mat = np.zeros((size, num_nodes))  # binary classes per node
    score_mat = np.zeros((size, 2))

    # How to generate subgraph?
    for idx in range(size):
        print('Starting to create %d example...' % idx)

        labels = np.zeros(shape=(num_nodes))
        labels[subgraph] = 1.
        candidates_change = np.random.choice(graph.nodes, num_change, replace=False)

        features = [np.random.normal(mean_1, std_1) if node in subgraph else np.random.normal(mean_0, std_0) for node in graph.nodes]
        labels[candidates_change] = 1 - labels[candidates_change]
        feature_mat[idx, :] = features
        label_mat[idx, :] = labels
        f1 = metric_f1(gt_lbl, labels)
        iou = metric_iou(gt_lbl, labels)
        score_mat[idx, 0] = f1
        score_mat[idx, 1] = iou

        print(features)
        print(labels)
        print(f1, iou)
        print('Example %d was generated!' % idx)


    # save data
    print('Starting to save generated data')
    np.save(os.path.join(path, adj_fn), adj_mat)
    np.save(os.path.join(path, feature_fn), feature_mat)
    np.save(os.path.join(path, label_fn), label_mat)
    np.save(os.path.join(path, score_fn), score_mat)
    print('Saving ends ...')


def metric_f1(gt_lbl, pred_lbl):
    gt_lbl =  np.array(gt_lbl > 0.5, dtype=np.float32)
    pred_lbl = np.array(pred_lbl > 0.5, dtype=np.float32)

    intersection = np.sum(np.minimum(gt_lbl, pred_lbl))
    union = max(np.sum(np.maximum(gt_lbl, pred_lbl)), 10e-8)

    return 2. * intersection / (intersection + union)


def metric_iou(gt_lbl, pred_lbl):
    gt_lbl = np.array(gt_lbl > 0.5, dtype=np.float32)
    pred_lbl = np.array(pred_lbl > 0.5, dtype=np.float32)

    intersection = np.sum(np.minimum(gt_lbl, pred_lbl))
    union = max(np.sum(np.maximum(gt_lbl, pred_lbl)), 10e-8)

    return intersection / union



if __name__ == '__main__':

    path = '/network/rit/lab/ceashpc/fjie/tmp/data/dvgcn/'
    generate_data(path)