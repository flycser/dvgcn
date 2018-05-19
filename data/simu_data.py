#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : simu_data
# @Date : 05/19/2018 14:19:43
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx

import os


def generate_data(path, one_hot=False):
    np.random.seed(123)

    num_nodes = 10
    mean_0 = 1.  # not subgraph
    std_0 = 1.
    mean_1 = 5.  # subgraph
    std_1 = 1.
    num_train = 50000
    num_val = 100
    num_test = 100

    if one_hot:
        adj_fn = 'adj_onehot'  # adjacent matrix file name
        feature_fn = 'features_onehot'  # features file name
        label_fn = 'labels_onehot'
    else:
        adj_fn = 'adj'  # adjacent matrix file name
        feature_fn = 'features'  # features file name
        label_fn = 'labels'  # labels file name

    left_subgraph = [5, 6, 7, 8, 9]
    right_subgraph = [0, 1, 2, 3, 4]

    graph = nx.Graph()
    graph.add_nodes_from([node for node in range(num_nodes)])
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 3), (3, 4), (3, 7), (5, 6), (5, 8), (6, 7), (6, 8), (6, 9), (7, 9),
         (8, 9)])


    adj_mat = nx.adjacency_matrix(graph).todense()
    # print nx.adjacency_matrix(graph)

    num = num_train + num_test + num_val
    feature_mat = np.zeros((num, num_nodes), dtype=np.float32) # one feature per node
    if one_hot:
        label_mat = np.zeros((num, num_nodes, 2)) # one hot setting
    else:
        label_mat = np.zeros((num, num_nodes)) # binary classes per node


    # How to generate subgraph?
    subgraph_candidates = [left_subgraph, right_subgraph]
    # triple_cliques = [(1, 3, 4), (0, 1, 3), (0, 2, 3), (6, 7, 9), (6, 8, 9), (5, 6, 8)]

    for idx in range(num):
        print('Starting to create %d example...' % idx)
        subgraph_selection = np.random.choice(len(subgraph_candidates))
        subgraph = subgraph_candidates[subgraph_selection]
        features = [np.random.normal(mean_1, std_1)if node in subgraph else np.random.normal(mean_0, std_0) for node in graph.nodes]
        feature_mat[idx] = features

        labels = np.zeros(num_nodes)
        labels[subgraph] = 1.
        if one_hot:
            label_mat[idx, :, 0] = labels
            label_mat[idx, :, 1] = 1 - labels
        else:
            label_mat[idx, :] = labels

        print('Example %d was generate!' % idx)


    # save data
    print('Starting to save generated data')
    np.save(os.path.join(path, adj_fn), adj_mat)
    np.save(os.path.join(path, feature_fn), feature_mat)
    np.save(os.path.join(path, label_fn), label_mat)
    print('Saving ending...')


if __name__ == '__main__':

    path='/network/rit/lab/ceashpc/fjie/tmp/data/sup1'
    generate_data(path=path, one_hot=True)