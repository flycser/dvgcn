#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : simu_data_2
# @Date : 05/19/2018 14:20:05
# @Poject : ceashpc-dvgcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

from __future__ import division
from __future__ import print_function

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

    feature_mat = np.zeros((size, num_nodes), dtype=np.float32) # one feature per node
    label_mat = np.zeros((size, num_nodes)) # binary classes per node

    # How to generate subgraph?
    for idx in range(size):
        print('Starting to create %d example...' % idx)

        if np.random.random() >= 0.5:
            subgraph = get_subgraph(graph, left_subgraph)

            features = [np.random.normal(mean_1, std_1) if node in subgraph else np.random.normal(mean_0, std_0) for node in graph.nodes]
            labels = [1 if node in subgraph else 0 for node in graph.nodes]
            feature_mat[idx, :] = features
            label_mat[idx, :] = labels
        else:
            subgraph = get_subgraph(graph, right_subgraph)

            features = [np.random.normal(mean_1, std_1) if node in subgraph else np.random.normal(mean_0, std_0) for
                        node in graph.nodes]
            labels = [1 if node in subgraph else 0 for node in graph.nodes]
            feature_mat[idx, :] = features
            label_mat[idx, :] = labels

        print(feature_mat[idx, :])
        print(label_mat[idx, :])

        print('Example %d was generated!' % idx)


    # save data
    print('Starting to save generated data')
    np.save(os.path.join(path, adj_fn), adj_mat)
    np.save(os.path.join(path, feature_fn), feature_mat)
    np.save(os.path.join(path, label_fn), label_mat)
    print('Saving ends ...')

def get_subgraph(graph, candidates):

    subgraph = []
    num = np.random.randint(1, len(candidates))
    start = np.random.choice(candidates)
    subgraph.append(start)
    for i in range(num):
        neighbors = [node for x in subgraph for node in nx.neighbors(graph, x) if node in candidates and not node in subgraph]
        next = np.random.choice(neighbors)
        subgraph.append(next)

    return subgraph


if __name__ == '__main__':

    path='/network/rit/lab/ceashpc/fjie/tmp/data/sup2'
    generate_data(path=path)