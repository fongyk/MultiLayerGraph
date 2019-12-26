import numpy as np
import random
import os
import math
from tqdm import tqdm

class SampleNeigh(object):
    '''
    sample h-hop knn for nodes.
    if the number of neighbors < k, then repeat some neighbors to reach k.
    '''
    def __init__(self, adj_sets, knn, hop_num):
        self.adj_sets = adj_sets
        self.knn = knn
        self.hop_num = hop_num

    def __call__(self, nodes):
        neighbors = []
        for h in range(self.hop_num):
            neigh_list = []
            for node in nodes:
                neigh = self.adj_sets[node]
                if len(neigh) > self.knn:
                    neigh_list.extend(random.sample(neigh, self.knn))
                else:
                    neigh = list(neigh)
                    while len(neigh) < self.knn:
                        neigh.extend(neigh)
                    neigh = neigh[:self.knn]
                    neigh_list.extend(neigh)
            neighbors.append(np.array(neigh_list))
            nodes = neigh_list[:]
        return neighbors

def collectNeighborFeatures(sampler, nodes, feature_map):
    neighs = sampler(nodes)
    features = []
    features.append(feature_map[nodes])
    for neigh in neighs:
        features.append(feature_map[neigh])
    return features


def getKnn(feature_path, suffix='_gem.npy'):
    '''
    write knn to npy.
    '''
    features = np.load(os.path.join(feature_path, 'feature_map' + suffix))
    n, d = features.shape
    knn = 100
    block_size = 1000
    block_num = int(math.ceil(n / float(block_size)))
    neighs = np.zeros((n, knn), dtype=np.int64)
    for b in tqdm(range(block_num)):
        sim = np.dot(features[b*block_size:(b+1)*block_size], features.T)
        sort_id = np.argsort(-sim, axis=1)
        neighs[b*block_size:(b+1)*block_size] = sort_id[:, :knn]
    np.save(os.path.join(feature_path, 'neighs' + suffix), neighs)

if __name__ == '__main__':
    getKnn('test_feature_map/oxford5k', '_rmac.npy')
    getKnn('test_feature_map/paris6k', '_rmac.npy')
