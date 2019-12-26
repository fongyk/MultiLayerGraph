import os
from collections import OrderedDict
import subprocess
import numpy as np
import pickle
import time

from compute_mAP import compute_map

## oxford5k, paris6k
class buildTestData(object):
    def __init__(self, img_path, gt_path, eval_func):
        self.img_path = img_path
        self.gt_path = gt_path
        self.eval_func = eval_func

        self.build()
    def build(self):
        gt_files = np.sort(os.listdir(self.gt_path))
        ## get the image names without the extension
        self.img_names = [img[:-4] for img in np.sort(os.listdir(self.img_path))]

        self.relevant = {}
        self.non_relevant = {}
        self.junk = {}

        self.name_to_file = OrderedDict()
        for f in gt_files:
            if f.endswith('_query.txt'):
                q_name = f[:-len('_query.txt')]
                q_data = open("{}/{}".format(self.gt_path, f)).readline().split(' ')
                q_imgname = q_data[0][5:] if q_data[0].startswith('oxc1') else q_data[0]
                self.name_to_file[q_name] = q_imgname
                good = set([e.strip() for e in open("{}/{}_ok.txt".format(self.gt_path, q_name))])
                good = good.union(set([e.strip() for e in open("{}/{}_good.txt".format(self.gt_path, q_name))]))
                junk = set([e.strip() for e in open("{}/{}_junk.txt".format(self.gt_path, q_name))])
                good_plus_junk = good.union(junk)
                self.relevant[q_name] = [i for i in range(len(self.img_names)) if self.img_names[i] in good]
                self.non_relevant[q_name] = [i for i in range(len(self.img_names)) if self.img_names[i] not in good_plus_junk]
                self.junk[q_name] = [i for i in range(len(self.img_names)) if self.img_names[i] in junk]

        self.q_names = list(self.name_to_file.keys())
        self.q_index = np.array([self.img_names.index(self.name_to_file[q]) for q in self.q_names])
        self.img_num = len(self.img_names)
        self.q_num = len(self.q_index)

    def evalRetrieval(self, similarity, save_path):
        '''
        Assume that query is included in database.
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ranks = np.argsort(similarity, axis=1)[:,::-1]
        APs = [self.eval_q(i, ranks[q,:], save_path) for (i, q) in enumerate(self.q_index)]
        return np.mean(APs)

    def evalQuery(self, database_feature, query_feature, save_path):
        '''
        Assume that query is NOT included in database.
        Note that self.q_index is not sorted by image names.
        Query and database is sorted by image names.
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        similarity = query_feature.dot(database_feature.T)
        ranks = np.argsort(similarity, axis=1)[:,::-1]
        q_id = np.argsort(self.q_index)
        APs = [self.eval_q(q, ranks[i,:], save_path) for (i, q) in enumerate(q_id)]
        return np.mean(APs)

    def eval_q(self, q, rank, save_path):
        if rank.shape[0] > self.img_num:
            rank_list = ['nil'] * rank.shape[0]
            for i, r in enumerate(rank):
                if r < self.img_num:
                    rank_list[i] = self.img_names[r]
        else:
            rank_list = np.array(self.img_names)[rank]
        timestamp = time.strftime('%Y%m%d%H%M%S')
        with open("{}/{}_{}.rnkl".format(save_path, self.q_names[q], timestamp), 'w') as fw:
            fw.write("\n".join(rank_list)+"\n")
        command = "{0} {1}/{2} {3}/{4}_{5}.rnkl".format(self.eval_func, self.gt_path, self.q_names[q], save_path, self.q_names[q], timestamp)
        sp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        AP = float(sp.stdout.readlines()[0])
        sp.wait()
        return AP


DATASETS = ['roxford5k', 'rparis6k', 'revisitop1m']

def configRefineDataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset == 'roxford5k' or dataset == 'rparis6k':
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

    elif dataset == 'revisitop1m':
        # loading imlist from a .txt file
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist

def evalRefineDataset(gnd, db_feat, q_feat):
    sim = np.dot(db_feat, q_feat.T)
    ranks = np.argsort(-sim, axis=0)

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    return mapE, mapM, mapH, mprE, mprM, mprH

def averageQueryExpansion(db_feature, query_feature, m):
    '''
    average query expansion with top-m results.
    '''
    q_num = query_feature.shape[0]
    sim = np.dot(query_feature, db_feature.T)
    sort_id = np.argsort(-sim, axis=1)
    for q in range(q_num):
        query_feature[q] = np.mean(np.concatenate((query_feature[q].reshape(1,-1), db_feature[sort_id[q, :m]]), axis=0), axis=0)
    return query_feature
