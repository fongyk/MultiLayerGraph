from __future__ import print_function

import torch

import os
import shutil

from collections import defaultdict

import numpy as np

def getNeigh(node_num, feature_map, knn):
    similarity = np.dot(feature_map, feature_map.T)
    sort_id = np.argsort(-similarity, axis=1)
    adj_sets = defaultdict(set)
    for n in range(node_num):
        for k in range(1, knn+1):
            adj_sets[n].add(sort_id[n, k])

    return adj_sets

def collectGraphTrain(node_num, class_num, feat_dim = 2048, knn = 10, suffix = '_gem.npy'):
    '''
    (training dataset)
    collect info. about graph including: node, label, feature, neighborhood(adjacent) relationship.
    neighborhood(adjacent) relationship are constructed based on similarity between features.
    '''
    print('node_num:', node_num, '\nclass_num:', class_num)

    feature_map = np.load('train_feature_map/feature_map' + suffix)
    assert node_num == feature_map.shape[0], 'node_num does not match feature shape.'
    assert feat_dim == feature_map.shape[1], 'feat_dim does not match feature shape.'
    label = np.load('train_feature_map/label' + suffix)
    # adj_sets = getNeigh(node_num, feature_map, knn)
    neighs = np.load('train_feature_map/neighs' + suffix)
    adj_sets = defaultdict(set)
    for n in range(node_num):
        adj_sets[n] = set(neighs[n, 1:knn+1])

    return torch.from_numpy(label), torch.from_numpy(feature_map).float(), adj_sets

def collectGraphTest(feature_path, node_num, feat_dim = 2048, knn = 10, suffix = '_gem.npy'):
    print("node num.:", node_num)

    feature_map = np.load(os.path.join(feature_path, 'feature_map' + suffix))
    assert node_num == feature_map.shape[0], 'node_num does not match feature shape.'
    assert feat_dim == feature_map.shape[1], 'feat_dim does not match feature shape.'
    neighs = np.load(os.path.join(feature_path, 'neighs' + suffix))
    adj_sets = defaultdict(set)
    for n in range(node_num):
        adj_sets[n] = set(neighs[n, 1:knn+1])
    query_feature = np.load(os.path.join(feature_path, 'query' + suffix))

    return torch.from_numpy(feature_map).float(), adj_sets, torch.from_numpy(query_feature).float()