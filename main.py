from __future__ import print_function
from __future__ import division

from PACK import *
from torch.optim.lr_scheduler import StepLR

from model import Graph, adjust_learning_rate
from aggregator import MeanPoolAggregator, MaxPoolAggregator, naiveMeanAggr
from utils import buildTestData, configRefineDataset, evalRefineDataset, averageQueryExpansion
from collect_graph import collectGraphTrain, collectGraphTest
from sample_neighbors import SampleNeigh, collectNeighborFeatures

import numpy as np
import math
import time
from tqdm import tqdm
import os

import visdom

import argparse
import ast

eval_func = 'evaluation/compute_ap'
retrieval_result = 'retrieval'
test_dataset = {
    'oxford5k': {
        'node_num': 5063,
        'img_testpath': '/path/to/images',
        'feature_path': '/path/to/oxford5k',
        'gt_path': '/path/to/oxford5k_groundTruth',
    },
    'paris6k': {
        'node_num': 6392,
        'img_testpath': '/path/to/images',
        'feature_path': '/path/to/paris6k',
        'gt_path': '/path/to/paris6k_groundTruth',
    },
    'roxford5k':{
        'node_num': 4993,
        'dataset_path': '/path/to/datasets',
        'feature_path': '/path/to/roxford5k',
    },
    'rparis6k':{
        'node_num': 6322,
        'dataset_path': '/path/to/datasets',
        'feature_path': '/path/to/rparis6k',
    },
    'oxford105k': {
        'node_num': 105134,
        'img_testpath': '/path/to/images',
        'feature_path': '/path/to/oxford105k',
        'gt_path': '/path/to/oxford5k_groundTruth',
    },
    'paris106k': {
        'node_num': 106463,
        'img_testpath': 'test_par/images',
        'feature_path': 'test_feature_map/paris106k',
        'gt_path': '/path/to/paris6k_groundTruth',
    },
    'roxford105k':{
        'node_num': 105064,
        'dataset_path': '/path/to/datasets',
        'feature_path': '/path/to/roxford105k',
    },
    'rparis106k':{
        'node_num': 106393,
        'dataset_path': '/path/to/datasets',
        'feature_path': '/path/to/rparis106k',
    }
}
building_oxf = buildTestData(img_path=test_dataset['oxford5k']['img_testpath'], gt_path=test_dataset['oxford5k']['gt_path'], eval_func=eval_func)
building_par = buildTestData(img_path=test_dataset['paris6k']['img_testpath'], gt_path=test_dataset['paris6k']['gt_path'], eval_func=eval_func)
building_roxf = configRefineDataset(dataset='roxford5k', dir_main=test_dataset['roxford5k']['dataset_path'])
building_rpar = configRefineDataset(dataset='rparis6k', dir_main=test_dataset['rparis6k']['dataset_path'])
building_oxf_flk = buildTestData(img_path=test_dataset['oxford105k']['img_testpath'], gt_path=test_dataset['oxford105k']['gt_path'], eval_func=eval_func)
building_par_flk = buildTestData(img_path=test_dataset['paris106k']['img_testpath'], gt_path=test_dataset['paris106k']['gt_path'], eval_func=eval_func)
building_roxf_flk = configRefineDataset(dataset='roxford5k', dir_main=test_dataset['roxford105k']['dataset_path'])
building_rpar_flk = configRefineDataset(dataset='rparis6k', dir_main=test_dataset['rparis106k']['dataset_path'])
building = {
    'oxford5k': building_oxf,
    'paris6k': building_par,
    'roxford5k': building_roxf,
    'rparis6k': building_rpar,
    'oxford105k': building_oxf_flk,
    'paris106k': building_par_flk,
    'roxford105k': building_roxf_flk,
    'rparis106k': building_rpar_flk,
}

aggregators = {
    'max': MaxPoolAggregator,
    'mean': MeanPoolAggregator,
}

def train(args):
    if args.suffix.startswith('_rmac') or args.suffix.startswith('_gem'):
        node_num, class_num = 36460, 578
    else:
        raise ValueError("Wrong feature type.")

    assert args.aggre_layer_num + 1 == len(args.embed_dims), "layer_num does not match embed_dims."

    label, feature_map, adj_sets = collectGraphTrain(node_num, class_num, args.embed_dims[0], args.knn, args.suffix)
    sampler = SampleNeigh(adj_sets, knn=args.knn, hop_num=args.aggre_layer_num)

    model = Graph(args.aggre_layer_num, args.embed_dims, class_num, aggregators[args.aggre_type], args.combine, args.concate, args.activate, args.residue, args.weighted, args.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.learning_rate_decay)

    if args.use_cuda:
        model.cuda()
        label = label.cuda()
        feature_map = feature_map.cuda()

    assert args.train_num < node_num, 'train_num > node_num.'
    np.random.seed(2)
    rand_indices = np.random.permutation(node_num)
    train_nodes = rand_indices[:args.train_num]
    val_nodes = rand_indices[args.train_num:]
    val_num = val_nodes.shape[0]

    ## sample positive and negative for rank loss
    positive_nodes, negative_nodes = [], []
    if args.mode == 'classRank':
        for anchor in train_nodes:
            for ri in rand_indices:
                if ri != anchor and label[ri] == label[anchor]:
                    positive_nodes.append(ri)
                    break
            while True:
                rand_node = np.random.choice(rand_indices)
                if label[rand_node] != label[anchor]:
                    negative_nodes.append(rand_node)
                    break
        positive_nodes = np.array(positive_nodes)
        negative_nodes = np.array(negative_nodes)
    elif args.mode == 'classBatch':
        for anchor in train_nodes:
            for ri in rand_indices:
                if ri != anchor and label[ri] == label[anchor]:
                    positive_nodes.append(ri)
                    break
        positive_nodes = np.array(positive_nodes)

    batch_size = args.batch_size
    iter_num = int(math.ceil(args.train_num/batch_size))

    check_loss = []
    val_accuracy = []
    check_step = args.check_step
    train_loss = 0.0
    iter_cnt = 0
    for e in range(args.epoch_num):
        ## train
        model.train()

        np.random.shuffle(train_nodes)
        for batch in range(iter_num):
            if args.mode == 'class':
                batch_nodes = train_nodes[batch*batch_size: (batch+1)*batch_size]
                batch_labels = label[batch_nodes]
                batch_features = collectNeighborFeatures(sampler, batch_nodes, feature_map)
                optimizer.zero_grad()
                loss = model.classLoss(batch_features, batch_labels)
                loss.backward()
                optimizer.step()

            elif args.mode == 'classRank':
                batch_idx = range(batch*batch_size, min((batch+1)*batch_size, args.train_num))
                batch_anchors = train_nodes[batch_idx]
                positive_labels = label[batch_anchors]
                batch_positives = positive_nodes[batch_idx]
                batch_negatives = negative_nodes[batch_idx]
                negative_labels = label[batch_negatives]
                batch_anchor_features = collectNeighborFeatures(sampler, batch_anchors, feature_map)
                batch_positive_features = collectNeighborFeatures(sampler, batch_positives, feature_map)
                batch_negative_features = collectNeighborFeatures(sampler, batch_negatives, feature_map)
                optimizer.zero_grad()
                loss = model.classRankLoss(batch_anchor_features, batch_positive_features, batch_negative_features, positive_labels, negative_labels, args.omega)
                loss.backward()
                optimizer.step()
            elif args.mode == 'classBatch':
                batch_idx = range(batch*batch_size, min((batch+1)*batch_size, args.train_num))
                batch_anchors = train_nodes[batch_idx]
                batch_labels = label[batch_anchors]
                batch_positives = positive_nodes[batch_idx]
                batch_anchor_features = collectNeighborFeatures(sampler, batch_anchors, feature_map)
                batch_positive_features = collectNeighborFeatures(sampler, batch_positives, feature_map)
                optimizer.zero_grad()
                loss = model.classBatchLoss(batch_anchor_features, batch_positive_features, batch_labels, args.omega)
                loss.backward()
                optimizer.step()

            iter_cnt += 1
            train_loss += loss.cpu().item()
            if iter_cnt % check_step == 0:
                check_loss.append(train_loss / check_step)
                print(time.strftime('%Y-%m-%d %H:%M:%S'), "epoch: {}/{}, iter:{}, loss: {:.4f}".format(e, args.epoch_num-1, iter_cnt, train_loss/check_step))
                train_loss = 0.0

        ## validation
        model.eval()
        group = int(math.ceil(val_num / batch_size))
        accur_cnt = 0
        for batch in range(group):
            batch_nodes = val_nodes[batch*batch_size: (batch+1)*batch_size]
            batch_label = label[batch_nodes].squeeze().cpu().detach().numpy()
            batch_neighs = sampler(batch_nodes)
            batch_features = []
            batch_features.append(feature_map[batch_nodes])
            for neigh in batch_neighs:
                batch_features.append(feature_map[neigh])
            _, scores = model(batch_features)
            batch_predict = np.argmax(scores.cpu().data.numpy(), axis=1)
            accur_cnt += np.sum(batch_predict == batch_label)
        val_accuracy.append(accur_cnt / val_num)
        print(time.strftime('%Y-%m-%d %H:%M:%S'), "Epoch: {}/{}, Validation Accuracy: {:.4f}".format(e, args.epoch_num-1, accur_cnt/val_num))
        print("******" * 10)

        ## learning rate schedule
        # scheduler.step()
        adjust_learning_rate(optimizer, e)

    ## save model
    checkpoint_path = 'checkpoint_{}.pth'.format(time.strftime('%Y%m%d%H%M%S'))
    torch.save({
            'train_num': args.train_num,
            'epoch_num': args.epoch_num,
            'learning_rate': args.learning_rate,
            'knn': args.knn,
            'embed_dims': args.embed_dims,
            'optimizer_type': 'Adam',
            'optimizer_state_dict': optimizer.state_dict(),
            'graph_state_dict': model.state_dict(),
            },
            checkpoint_path
    )

    vis = visdom.Visdom(env='Graph', port='8099')
    vis.line(
            X = np.arange(1, len(check_loss)+1, 1) * check_step,
            Y = np.array(check_loss),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='itr.',
                ylabel='loss'
            )
    )
    vis.line(
            X = np.arange(1, len(val_accuracy)+1, 1),
            Y = np.array(val_accuracy),
            opts = dict(
                title=time.strftime('%Y-%m-%d %H:%M:%S'),
                xlabel='epoch',
                ylabel='accuracy'
            )
    )

    np.save('check_loss_{}_{}_{}_{}_{}.npy'.format(str(args.aggre_layer_num), str(args.knn), str(args.combine), args.mode, time.strftime('%Y%m%d%H%M')), np.array(check_loss))
    np.save('val_accuracy_{}_{}_{}_{}_{}.npy'.format(str(args.aggre_layer_num), str(args.knn), str(args.combine), args.mode, time.strftime('%Y%m%d%H%M')), np.array(val_accuracy))

    return checkpoint_path, class_num

def test(checkpoint_path, class_num, args):
    model = Graph(args.aggre_layer_num, args.embed_dims, class_num, aggregators[args.aggre_type], args.combine, args.concate, args.activate, args.residue, args.weighted, args.margin)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['graph_state_dict'])
    model.eval()
    if args.use_cuda:
        model.cuda()

    for key in building.keys():
        node_num = test_dataset[key]['node_num']
        old_feature_map, adj_sets, old_query = collectGraphTest(test_dataset[key]['feature_path'], node_num, args.embed_dims[0], args.knn, args.suffix)

        sampler = SampleNeigh(adj_sets, knn=args.knn, hop_num=args.aggre_layer_num)

        if args.use_cuda:
            old_feature_map = old_feature_map.cuda()
            query = old_query.cuda()

        ## process query
        new_query = model.queryForward(query)
        new_query = new_query.cpu().detach().numpy()
        old_query = old_query.detach().numpy()

        batch_num = int(math.ceil(node_num/args.batch_size))
        test_nodes = np.arange(node_num)
        new_feature_map = torch.FloatTensor()
        for batch in tqdm(range(batch_num)):
            batch_nodes = test_nodes[batch*args.batch_size: (batch+1)*args.batch_size]
            batch_neighs = sampler(batch_nodes)
            batch_features = []
            batch_features.append(old_feature_map[batch_nodes])
            for neigh in batch_neighs:
                batch_features.append(old_feature_map[neigh])
            new_feature, _ = model(batch_features)
            new_feature_map = torch.cat((new_feature_map, new_feature.cpu().data), dim=0)
        new_feature_map = new_feature_map.numpy()
        old_feature_map = old_feature_map.cpu().numpy()

        np.save('new_feature_map_{}.npy'.format(key), new_feature_map)

        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'eval {}'.format(key))
        if not key.startswith('r'):
            mAP_old = building[key].evalQuery(old_feature_map, old_query, retrieval_result)
            mAP_new = building[key].evalQuery(new_feature_map, new_query, retrieval_result)
            print('base feature: {}, new feature: {}'.format(old_feature_map.shape, new_feature_map.shape))
            print('base mAP: {:.4f}, new mAP: {:.4f}, improve: {:.4f}'.format(mAP_old, mAP_new, mAP_new - mAP_old))

        else:
            print('base feature: {}, new feature: {}'.format(old_feature_map.shape, new_feature_map.shape))
            print('          --base--')
            mapE, mapM, mapH, mprE, mprM, mprH = evalRefineDataset(building[key]['gnd'], old_feature_map, old_query)
            print('mAP E: {:.4f}, M: {:.4f}, H: {:.4f}'.format(mapE, mapM, mapH))
            print('mP@k [1 5 10]\n E: {}\n M: {}\n H: {}'.format(np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))
            print('          --graph--')
            mapE, mapM, mapH, mprE, mprM, mprH = evalRefineDataset(building[key]['gnd'], new_feature_map, new_query)
            print('mAP E: {:.4f}, M: {:.4f}, H: {:.4f}'.format(mapE, mapM, mapH))
            print('mP@k [1 5 10]\n E: {}\n M: {}\n H: {}'.format(np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))

        ## naive mean aggregation
        sampler = SampleNeigh(adj_sets, knn=args.knn, hop_num=1)
        mean_feature_map = np.zeros((node_num, args.embed_dims[0]))
        for batch in range(batch_num):
            batch_nodes = test_nodes[batch*args.batch_size: (batch+1)*args.batch_size]
            batch_neighs = sampler(batch_nodes)
            batch_features = [old_feature_map[batch_nodes], old_feature_map[batch_neighs[0]]]
            new_feature = naiveMeanAggr(*batch_features, weighted=args.weighted)
            mean_feature_map[batch_nodes] = new_feature

        if not key.startswith('r'):
            mAP_new = building[key].evalQuery(mean_feature_map, old_query, retrieval_result)
            print('mean mAP: {:.4f}, improve: {:.4f}'.format(mAP_new, mAP_new - mAP_old))

        else:
            print('          --mean--')
            mapE, mapM, mapH, mprE, mprM, mprH = evalRefineDataset(building[key]['gnd'], mean_feature_map, old_query)
            print('mAP E: {:.4f}, M: {:.4f}, H: {:.4f}'.format(mapE, mapM, mapH))
            print('mP@k [1 5 10]\n E: {}\n M: {}\n H: {}'.format(np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))

        ## average query expansion
        base_aqe = []
        graph_aqe = []
        mean_aqe = []
        for m in [1,3,5,7,9]:
            old_aug_query = averageQueryExpansion(old_feature_map, old_query, m)
            new_aug_query = averageQueryExpansion(new_feature_map, new_query, m)
            mean_aug_query = averageQueryExpansion(mean_feature_map, old_query, m)
            if not key.startswith('r'):
                mAP_old = building[key].evalQuery(old_feature_map, old_aug_query, retrieval_result)
                base_aqe.append((m, np.around(mAP_old, decimals=4)))
                mAP_new = building[key].evalQuery(new_feature_map, new_aug_query, retrieval_result)
                graph_aqe.append((m, np.around(mAP_new, decimals=4)))
                mAP_mean = building[key].evalQuery(mean_feature_map, mean_aug_query, retrieval_result)
                mean_aqe.append((m, np.around(mAP_mean, decimals=4)))
            else:
                mapE, mapM, mapH, mprE, mprM, mprH = evalRefineDataset(building[key]['gnd'], old_feature_map, old_aug_query)
                base_aqe.append((m, np.around(mapE, decimals=4), np.around(mapM, decimals=4), np.around(mapH, decimals=4), np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))
                mapE, mapM, mapH, mprE, mprM, mprH = evalRefineDataset(building[key]['gnd'], new_feature_map, new_aug_query)
                graph_aqe.append((m, np.around(mapE, decimals=4), np.around(mapM, decimals=4), np.around(mapH, decimals=4), np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))
                mapE, mapM, mapH, mprE, mprM, mprH = evalRefineDataset(building[key]['gnd'], mean_feature_map, mean_aug_query)
                mean_aqe.append((m, np.around(mapE, decimals=4), np.around(mapM, decimals=4), np.around(mapH, decimals=4), np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))
        print('         --base+aqe--')
        print(base_aqe)
        print('        --graph+aqe--')
        print(graph_aqe)
        print('        --mean+aqe--')
        print(mean_aqe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Graph Attention Network, train on Landmark_clean, test on Oxford5k and Paris6k.')
    parser.add_argument('--epoch_num', type=int, default=70, required=False, help='training epoch number.')
    parser.add_argument('--step_size', type=int, default=30, required=False, help='learning rate decay step_size.')
    parser.add_argument('--learning_rate_decay', type=float, default=0.5, required=False, help='learning rate decay factor.')
    parser.add_argument('--batch_size', type=int, default=64, required=False, help='training batch size.')
    parser.add_argument('--check_step', type=int, default=100, required=False, help='loss check step.')
    parser.add_argument('--use_cuda', type=ast.literal_eval, default=True, required=False, help='whether to use gpu (True) or not (False).')
    parser.add_argument('--learning_rate', type=float, default=0.001, required=False, help='training learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6, required=False, help='weight decay (L2 regularization).')
    parser.add_argument('--knn', type=int, default=10, required=False, help='number of neighbors to aggregate.')
    parser.add_argument('--suffix', type=str, default='_gem.npy', required=False, help='feature type, \'f\' for vggnet (512-d), \'fr\' for resnet (2048-d), \'frmac\' for vgg16_rmac (512-d).')
    parser.add_argument('--train_num', type=int, default=33000, required=False, help='number of training nodes (less than 36460). Left for validation.')
    parser.add_argument('--aggre_layer_num', type=int, default=1, required=False, help='number of aggregator layers.')
    parser.add_argument('--aggre_type', type=str, default='max', required=False, help='aggregator function.')
    parser.add_argument('--embed_dims', nargs='+', type=int, required=False, help='input dim and hidden layer dims.')
    parser.add_argument('--combine', type=ast.literal_eval, default=False, required=False, help='combine(True) features of all layers or not.')
    parser.add_argument('--concate', type=ast.literal_eval, default=False, required=False, help='concate(True) self_feature and aggre_features or average(False) them.')
    parser.add_argument('--activate', type=ast.literal_eval, default=False, required=False, help='whether to use non-linear(True) activation function on embeddings or not.')
    parser.add_argument('--margin', type=float, default=0.1, required=False, help='margin in triplet loss.')
    parser.add_argument('--mode', type=str, default='class', required=False, help='loss mode: softmax loss (class) or softmax + triplet loss (classRank).')
    parser.add_argument('--omega', type=float, default=0.1, required=False, help='weight trade off between softmax loss and triplet loss.')
    parser.add_argument('--residue', type=ast.literal_eval, default=True, required=False, help='residue connection in graph.')
    parser.add_argument('--weighted', type=ast.literal_eval, default=False, required=False, help='weighted graph edge or not.')

    args, _ = parser.parse_known_args()

    print("< < < < < < < < < < < Graph Neural NetWork > > > > > > > > > >")
    print("= = = = = = = = = = = PARAMETERS SETTING = = = = = = = = = = =")
    for k, v in vars(args).items():
        print k, ":", v
    print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")

    print("training ......")
    checkpoint_path, class_num = train(args)

    print("testing ......")
    test(checkpoint_path, class_num, args)
