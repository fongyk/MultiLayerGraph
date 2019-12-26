import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as nninit

def adjust_learning_rate(optimizer, epoch):
    if epoch == 39:
        for params in optimizer.param_groups:
            params['lr'] *= 0.5
    if epoch == 59:
        for params in optimizer.param_groups:
            params['lr'] *= 0.1

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class BatchLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(BatchLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_features, positive_features, labels):
        target = labels.view(labels.size(0), -1)
        target = (target == target.t()).float()
        sqdist = 2 - 2 * torch.matmul(anchor_features, positive_features.t())
        pos_dist = torch.diagonal(sqdist)
        diff_dist = pos_dist.view(-1, 1).repeat(1, sqdist.size(0)) - sqdist
        loss = torch.sum(F.relu(diff_dist + self.margin), dim=1)
        return loss.mean()

class Graph(nn.Module):
    def __init__(self, aggre_layer_num, embed_dims, class_num, aggregator, combine=False, concate=False, activate=False, residue=True, weighted=False, margin=0.1):
        super(Graph, self).__init__()

        self.aggre_layer_num = aggre_layer_num
        self.embed_dims = embed_dims
        self.class_num = class_num
        self.aggregator = aggregator
        self.combine = combine
        self.concate = concate
        self.activate = activate
        self.residue = residue
        self.weighted = weighted
        self.margin = margin

        aggre_layers = []
        for layer in range(self.aggre_layer_num):
            in_dim, out_dim = self.embed_dims[layer:layer+2]
            aggre = self.aggregator(in_dim=in_dim, out_dim=out_dim, concate=self.concate, activate=self.activate, residue=self.residue, weighted=self.weighted)
            aggre_layers.append(aggre)
        self.aggre_layers = nn.Sequential(*aggre_layers)

        self.fc = nn.Sequential(
                nn.Linear(sum(self.embed_dims) if self.combine else self.embed_dims[-1], self.class_num, bias=True)
        )
        for module in self.fc.children():
            if isinstance(module, nn.Linear):
                nninit.xavier_uniform_(module.weight)
                nninit.constant_(module.bias, 0.01)

        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_rank = TripletLoss(margin=self.margin)
        self.criterion_batch = BatchLoss(margin=self.margin)

    def forward(self, all_features):
        '''
        all_features: a list, stores features of nodes & features of 1-hop neighbors & features of 2-hop neighbors & ...
        i-hop neighbors' feature: 1 * (nkd) torch.tensor, k is the number of neighbors.
        each aggre_layer update nodes' feature by using their neighbors, except the outermost-hop nodes.
        combine_feature: combine features of all layers.
        '''

        combine_feature = all_features[0] if self.combine else []
        for layer_id, aggre_layer in enumerate(self.aggre_layers.children()):
            all_features = [aggre_layer(all_features[k], all_features[k+1], layer_id != self.aggre_layer_num-1) for k in range(len(all_features)-1)]
            if self.combine:
                combine_feature = torch.cat((combine_feature, all_features[0]), dim=1)

        assert len(all_features) == 1, "len(all_features) != 1"

        hidden_activation = F.relu(combine_feature) if self.combine else F.relu(all_features[0])
        scores = self.fc(hidden_activation)

        if self.combine:
            return combine_feature, scores
        else:
            return all_features[0], scores

    def queryForward(self, query_feature):
        combine_query = query_feature if self.combine else []
        for layer_id, aggre_layer in enumerate(self.aggre_layers.children()):
            query_feature = aggre_layer.queryForward(query_feature, layer_id != self.aggre_layer_num-1)
            if self.combine:
                combine_query = torch.cat((combine_query, query_feature), dim=1)
        if self.combine:
            return combine_query
        else:
            return query_feature

    def classLoss(self, all_features, labels):
        _, scores = self.forward(all_features)
        return self.criterion_class(scores, labels.squeeze())

    def classRankLoss(self, anchor_features, positive_features, negative_features, positive_labels, negative_labels, omega=0.5):
        new_anchor_features, anchor_scores = self.forward(anchor_features)
        new_positive_features, positive_scores = self.forward(positive_features)
        new_negative_features, negative_scores = self.forward(negative_features)
        class_loss = (self.criterion_class(anchor_scores, positive_labels.squeeze()) + self.criterion_class(positive_scores, positive_labels.squeeze()) + self.criterion_class(negative_scores, negative_labels.squeeze())) / 3.0
        rank_loss = self.criterion_rank(new_anchor_features, new_positive_features, new_negative_features)
        return omega * rank_loss + (1.0 - omega) * class_loss

    def classBatchLoss(self, anchor_features, positive_features, labels, omega=0.5):
        new_anchor_features, anchor_scores = self.forward(anchor_features)
        new_positive_features, positive_scores = self.forward(positive_features)
        class_loss = (self.criterion_class(anchor_scores, labels.squeeze()) + self.criterion_class(positive_scores, labels.squeeze())) / 2.0
        batch_loss = self.criterion_batch(new_anchor_features, new_positive_features, labels)
        return omega * batch_loss + (1.0 - omega) * class_loss
