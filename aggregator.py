import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as nninit

import numpy as np
from sklearn import preprocessing

def getWeight(nodes_feature, neighs_feature, out_dim):
    '''
    input:
        nodes_feature: n x d
        neighs_feature: n x k x d
    output:
        weight: n x k x out_dim
    '''
    n, k, d = neighs_feature.size()
    similarity = (neighs_feature * nodes_feature.view(n, 1, d).repeat(1, k, 1)).sum(dim=2)
    # weight = similarity / similarity.sum(dim=1, keepdim=True) ## l1 weighting
    weight = F.softmax(similarity, dim=1) ## softmax weighting
    weight = weight.unsqueeze(2).repeat(1, 1, out_dim)
    return weight

class PoolAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggre_func, concate=False, activate=False, residue=True, weighted=False):
        super(PoolAggregator, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggre_func = aggre_func
        self.concate = concate
        self.activate = activate
        self.residue = residue
        self.weighted = weighted

        if not self.concate:
            self.W = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        else:
            self.W = nn.Parameter(torch.FloatTensor(2 * self.in_dim, self.out_dim))
        nninit.xavier_uniform_(self.W)

        self.P = nn.Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        nninit.xavier_uniform_(self.P)

    def forward(self, nodes_feature, neighs_feature, not_final_layer):
        n, d = nodes_feature.size()
        aggre_feature = neighs_feature.view(n, -1, d)
        weight = 1.0
        if not self.concate:
            nodes_feature = nodes_feature.view(n, 1, d)
            merge_feature = torch.cat((aggre_feature, nodes_feature), dim=1)
            if self.weighted:
                weight = getWeight(nodes_feature, merge_feature, self.out_dim)
            merge_feature = merge_feature.matmul(self.P)
            merge_feature = self.aggre_func(merge_feature * weight)
        else:
            if self.weighted:
                weight = getWeight(nodes_feature, aggre_feature, self.out_dim)
            aggre_feature = aggre_feature.matmul(self.P)
            aggre_feature = self.aggre_func(aggre_feature * weight)
            merge_feature = torch.cat((nodes_feature, aggre_feature), dim=1)
        new_feature = merge_feature.matmul(self.W)
        if self.activate and not_final_layer:
            new_feature = F.relu(new_feature)
        new_feature = F.normalize(new_feature, p=2, dim=1) ## L2 normalization

        return F.normalize(nodes_feature.view(n, d) + new_feature, p=2, dim=1) if self.residue else new_feature

    def queryForward(self, query_feature, not_final_layer):
        if not self.concate:
            merge_feature = query_feature.matmul(self.P)
        else:
            aggre_feature = query_feature.matmul(self.P)
            merge_feature = torch.cat((query_feature, aggre_feature), dim=1)
        new_feature = merge_feature.matmul(self.W)
        if self.activate and not_final_layer:
            new_feature = F.relu(new_feature)
        new_feature = F.normalize(new_feature, p=2, dim=1) ## L2 normalization

        return F.normalize(query_feature + new_feature, p=2, dim=1) if self.residue else new_feature


class MeanPoolAggregator(PoolAggregator):

    def __init__(self, in_dim, out_dim, concate=False, activate=False, residue=True, weighted=False):
        super(MeanPoolAggregator, self).__init__(**{
            'in_dim': in_dim,
            'out_dim': out_dim,
            'aggre_func': lambda x: x.mean(dim=1),
            'concate': concate,
            'activate': activate,
            'residue': residue,
            'weighted': weighted,
        })

class MaxPoolAggregator(PoolAggregator):

    def __init__(self, in_dim, out_dim, concate=False, activate=False, residue=True, weighted=False):
        super(MaxPoolAggregator, self).__init__(**{
            'in_dim': in_dim,
            'out_dim': out_dim,
            'aggre_func': lambda x: x.max(dim=1)[0],
            'concate': concate,
            'activate': activate,
            'residue': residue,
            'weighted': weighted,
        })

def naiveMeanAggr(nodes_feature, neighs_feature, weighted=False):
    n, d = nodes_feature.shape
    aggre_feature = neighs_feature.reshape(n, -1, d)
    nodes_feature = nodes_feature.reshape(n, 1, d)
    merge_feature = np.concatenate((aggre_feature, nodes_feature), axis=1)
    if weighted:
        weight = getWeight(torch.from_numpy(nodes_feature), torch.from_numpy(merge_feature), d)
        merge_feature = merge_feature * weight.numpy()
    new_feature = merge_feature.mean(axis=1)
    new_feature = preprocessing.normalize(new_feature, axis=1, norm='l2')
    return new_feature
