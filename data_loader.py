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


class TrainDataset(data.Dataset):

    def __init__(self, triplets, num_node, num_rels, params):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class EvalDataset(data.Dataset):

    def __init__(self, triplets, num_node, num_rels, params):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class DataLoader(object):

    def __init__(self, dataset, params):
        self.data_dir = os.path.join('data', dataset)

    def _load_data(self):
        # read entities
        self.entity2id = {}
        with open(os.path.join(self.data_dir, 'entity2id.txt'), 'r') as f:
            for line in f:
                entity, eid = line.strip().split('\t')
                self.entity2id[entity] = int(eid)

        # read relations
        self.relation2id = {}
        with open(os.path.join(self.data_dir, 'relation2id.txt'), 'r') as f:
            for line in f:
                relation, rid = line.strip().split('\t')
                self.relation2id[relation] = int(rid)

        # read triplets
        self.triplets = defaultdict(list)
        for data_type in ['train', 'valid', 'test']:
            with open(os.path.join(self.data_dir, data_type + '.txt'), 'r') as f:
                for line in f:
                    head, relation, tail = line.strip().split('\t')
                    self.triplets[data_type].append((self.entity2id[head], self.relation2id[relation], self.entity2id[tail]))

        # report the dataset
        logging.info('num_entities: {}'.format(len(self.entity2id)))
        logging.info('num_relations: {}'.format(len(self.relation2id)))
        logging.info('num_train_triplets: {}'.format(len(self.triplets['train'])))
        logging.info('num_valid_triplets: {}'.format(len(self.triplets['val'])))
        logging.info('num_test_triplets: {}'.format(len(self.triplets['test'])))

    def _edge_normal(self, edge_type, edge_index, num_entity):

        edge_type, edge_index = torch.from_numpy(edge_type), torch.from_numpy(edge_index)
        counts = torch.ones_like(edge_type).to(torch.float)
        deg = scatter_add(counts, edge_index[1], dim_size=num_entity)
        edge_norm = 1 / deg[edge_index[1]]
        edge_norm[torch.isinf(edge_norm)] = 0

        return edge_norm

    def _build_graph_from_triplets(self, graph_nodes, triplets, bi_directional=True):
        """Create a graph when given triplets

        Args:
            graph_nodes: (np.ndarray) nodes ids in the target graph
            triplets: (np.ndarray) nodes and relation ids in the graph
        Return:
            torch_geometric.data.Data
        """
        src, rel, dst = triplets.transpose()
        # Create bi-directional graph
        if bi_directional is True:
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relation))

        edge_index = np.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr))
        data.entity = torch.from_numpy(graph_nodes)
        data.num_nodes = len(graph_nodes)
        data.edge_norm = self._edge_normal(edge_attr, edge_index, len(graph_nodes))

        return data

    def get_dataset(self, data_type, params):

        num_node, num_rels = len(self.entity2id), len(self.relation2id)

        if data_type == 'train':
            return TrainDataset(self.triplets['train'], num_node, num_rels, params)
        elif data_type == 'valid' or data_type == 'test':
            return EvalDataset(self.triplets[data_type], num_node, num_rels, params)
        else:
            raise ValueError('Unkown data type')

    def data_iterator(self, dataset, batch_size, num_worker, shuffle):
        iterator = data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_worker,
            shuffle=shuffle
        )

        return iterator
