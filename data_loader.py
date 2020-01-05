r"""
Interface used for dataset and batch-wise training.
"""

import os
import logging
import torch
import numpy as np

from collections import defaultdict
from torch.utils import data
from torch_geometric.data import Data
from torch_scatter import scatter_add


class KBDataset(data.Dataset):

    def __init__(self, triplets, num_entity, params, training=False):
        self.triplets = triplets
        self.num_entity = num_entity
        self.params = params
        self.training = training

    def collate_fn(self, batch):
        triple = torch.stack([_[0] for _ in batch], dim=0)
        triple_label = torch.stack([_[1] for _ in batch], dim=0)

        return triple, triple_label

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        # triple, label, sub_samp = torch.LongTensor(triplet['triple']), np.int32(triplet['label']), np.float32(triplet['sub_samp'])
        triple, label = torch.LongTensor(
            triplet['triple']), np.int32(triplet['label'])
        triple_label = self.get_label(label)

        if self.training is True and self.params.lbl_smooth != 0.0:
            triple_label = (1.0 - self.params.lbl_smooth) * \
                triple_label + (1.0 / self.num_entity)

        return triple, triple_label

    def get_label(self, label):
        y = np.zeros([self.num_entity], dtype=np.float32)
        for e in label:
            y[e] = 1.0
        return torch.FloatTensor(y)


class DataLoader(object):

    def __init__(self, dataset, params):
        self.data_dir = os.path.join('data', dataset)

        self.graph = self._load_data()

    def _load_data(self):
        # read entities
        self.entity2id = {}
        with open(os.path.join(self.data_dir, 'entity2id.txt'), 'r') as f:
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid)
        self.num_entity = len(self.entity2id)

        # read relations
        self.relation2id = {}
        with open(os.path.join(self.data_dir, 'relation2id.txt'), 'r') as f:
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
        self.num_relation = len(self.relation2id)
        self.relation2id.update({rel + '_reverse': idx + len(self.relation2id)
                                 for idx, rel in enumerate(self.relation2id.keys())})

        # read triplets
        data = defaultdict(list)
        sr2o = defaultdict(set)
        for data_type in ['train', 'valid', 'test']:
            with open(os.path.join(self.data_dir, data_type + '.txt'), 'r') as f:
                for line in f:
                    head, relation, tail = line.strip().split()
                    sub, rel, obj = self.entity2id[head], self.relation2id[relation], self.entity2id[tail]
                    data[data_type].append((sub, rel, obj))

                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.num_relation)].add(sub)

            if data_type == 'train':
                sr2o_train = {k: list(v) for k, v in sr2o.items()}
        sr2o_all = {k: list(v) for k, v in sr2o.items()}
        data = dict(data)

        self.triplets = defaultdict(list)

        for (sub, rel), objs in sr2o_train.items():
            self.triplets['train'].append(
                {'triple': (sub, rel, -1), 'label': objs, 'sub_samp': 1})

        for data_type in ['valid', 'test']:
            for sub, rel, obj in data[data_type]:
                rel_inv = rel + self.num_relation
                self.triplets['{}_{}'.format(data_type, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': sr2o_all[(sub, rel)]})
                self.triplets['{}_{}'.format(data_type, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': sr2o_all[(obj, rel_inv)]})
        self.triplets = dict(self.triplets)

        graph = self._build_graph(np.arange(self.num_entity, dtype=np.int64), np.array(
            data['train'], dtype=np.int64), bi_direction=True)

        # report the dataset
        logging.info('entity={}, relation={}, train_triplets={}, valid_triplets={}, test_triplets={}'.format(
            self.num_entity, self.num_relation, len(data['train']), len(data['valid']), len(data['test'])))

        return graph

    def _edge_normal(self, edge_type, edge_index, num_entity):

        edge_type, edge_index = torch.from_numpy(edge_type), torch.from_numpy(edge_index).long()
        counts = torch.ones_like(edge_type).to(torch.float)
        deg = scatter_add(counts, edge_index[1], dim_size=num_entity)
        edge_norm = 1 / deg[edge_index[1]]
        edge_norm[torch.isinf(edge_norm)] = 0

        return edge_norm

    def _build_graph(self, graph_nodes, triplets, bi_direction=True):
        """Create a graph when given triplets

        Args:
            graph_nodes: (np.ndarray) nodes ids in the target graph
            triplets: (np.ndarray) nodes and relation ids in the graph
        Return:
            torch_geometric.data.Data
        """
        src, rel, dst = triplets.transpose()
        # Create bi-directional graph
        if bi_direction is True:
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relation))

        edge_index = np.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_attr))
        data.entity = torch.from_numpy(graph_nodes)
        data.num_nodes = len(graph_nodes)
        data.edge_norm = self._edge_normal(
            edge_attr, edge_index, len(graph_nodes))

        return data

    def _get_dataset(self, data_type, params):

        if data_type == 'train':
            return KBDataset(self.triplets['train'], len(self.entity2id), params, training=True)
        elif data_type in ['valid_head', 'valid_tail', 'test_head', 'test_tail']:
            return KBDataset(self.triplets[data_type], len(self.entity2id), params)
        else:
            raise ValueError('Unkown data type')

    def _create_data_loader(self, dataset, batch_size, num_workers, shuffle):
        iterator = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=max(0, num_workers),
            shuffle=shuffle,
            collate_fn=dataset.collate_fn
        )

        return iterator

    def get_data_loaders(self, batch_size, num_workers, params):

        marks = ['train', 'valid_head', 'valid_tail', 'test_head', 'test_tail']

        data_iter = {
            mark: self._create_data_loader(self._get_dataset(mark, params), batch_size, num_workers, shuffle=True) for mark in marks
        }

        return data_iter
