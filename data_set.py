"""Build dataset for graph convolutinal networks"""

import os
import logging

import numpy as np
import torch
from torch_geometric.data import Data


class DataSet:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.sample_size = params.sample_size
        self.split_size = params.split_size
        self.negative_rate = params.negative_rate
        self.device = params.device

        # whether dataset exists
        assert os.path.exists(os.path.join(
            'data', self.dataset)), 'Dataset {} not found'.format(self.dataset)

        # the following operations only run once
        # load entites and relations
        self.entity2id = self._load_entries(self.dataset)
        self.relation2id = self._load_entries(self.dataset, load_entities=False)

        self.n_entity = len(self.entity2id)
        self.n_relation = len(self.relation2id)

        # load traiplets
        self.train_triplets, self.valid_triplets, self.test_triplets = self._load_data()

    def total_triplets(self):
        """Get all triplets for evaluation"""
        return np.concatenate((
            self.train_triplets, self.valid_triplets, self.test_triplets
        ))

    def _load_data(self):
        """Load training set, valid set and test set from files
        because of the 'filtering' setting in evaluation
        """
        # load triplets for training set, valid set and test set
        train_triplets = self._load_triplets(self.dataset, 'train')
        valid_triplets = self._load_triplets(self.dataset, 'valid')
        test_triplets = self._load_triplets(self.dataset, 'test')

        return train_triplets, valid_triplets, test_triplets

    def _load_entries(self, dataset, load_entities=True):
        """Load entities or relations from file, i.e. entities.dict or relations.dict

        Args:
            dataset: (string) dataset name
            load_entites: (bool) whether load entities or relations
        Return:
            entries2idx: (dict) map each entry to a number
        """
        entries = {}
        entry_file = 'entities.dict' if load_entities else 'relations.dict'
        with open(os.path.join('data', dataset, entry_file), 'r') as f:
            for line in f.readlines():
                eid, entry = line.strip().split()
                entries[entry] = int(eid)

        # logging
        entry_name = 'entities' if load_entities else 'relations'
        logging.info('Load {} {} from file'.format(len(entries), entry_name))

        return entries

    def _load_triplets(self, dataset, data_type):
        """Load triplets from train.txt, valid.txt or test.txt

        Args:
            dataset: (string) dataset name
            data_type: (string)  'train', 'valid' or 'test'
        Return:
            triplets: (list) [(h, r, t),...]
        """
        assert data_type in ['train', 'valid',
                             'test'], 'Invalid data type when loading triplets'

        triplets = []
        with open(os.path.join('data', dataset, data_type + '.txt'), 'r') as f:
            for line in f.readlines():
                head, relation, tail = line.strip().split()
                triplets.append((self.entity2id[head], self.relation2id[relation], self.entity2id[tail]))

        logging.info('Found {} triplets in {}.txt from dataset {}'.format(
            len(triplets), data_type, dataset))

        return np.array(triplets)

    def _sample_edge_uniform(self, n_triples, sample_size):
        """Generate the edge indices to sample"""
        return np.random.choice(np.arange(n_triples), sample_size, replace=False)

    def _negative_sampling(self, pos_samples, num_entity, negative_rate):
        """Sample negative triplets for training

        Args:
            pos_sample: positive samples
            num_entity: (int) the number of entities
            negative_rate: (int) the proportion of negative samples
        Return:
            samples: (array) samples including positive and negative samples
            labels: (array) labels whose values in [0, 1]
        """
        size = len(pos_samples)
        # genrate labels
        labels = np.zeros(size * (negative_rate + 1), dtype=np.float)
        labels[: size] = 1  # labels of positive samples

        # sample negative samples
        neg_samples = np.tile(pos_samples, (negative_rate, 1))
        values = np.random.choice(num_entity, size=size * negative_rate)
        choices = np.random.uniform(size=size * negative_rate)
        subj, obj = choices > 0.5, choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        return np.concatenate((pos_samples, neg_samples)), labels

    def build_train_graph(self):
        """Get training graph and signals
        First perform edge neighborhood sampling on graph,
        then perform negative sampling to generate negative samples
        """
        sample_size = self.sample_size
        split_size = self.split_size
        negative_rate = self.negative_rate

        edge_ids = self._sample_edge_uniform(len(self.train_triplets), sample_size)

        # select sampled edges
        edges = self.train_triplets[edge_ids]
        src, rel, dst = edges.transpose()
        uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # Negative sampling
        samples, labels = self._negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

        # further split graph, only half of the edges will be used as graph structure,
        # while the rest half is used as unseen positive samples
        split_size = int(sample_size * split_size)
        graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)

        src = torch.LongTensor(src[graph_split_ids]).contiguous()
        dst = torch.LongTensor(dst[graph_split_ids]).contiguous()
        rel = torch.LongTensor(rel[graph_split_ids]).contiguous()
        edge_ids = torch.LongTensor(edge_ids[graph_split_ids]).contiguous()

        # create bi-directional graph
        # src, dst = torch.cat((src, dst)), torch.cat((dst, src))
        # rel = torch.cat((rel, rel + self.n_relation))

        edge_index = torch.stack((src, dst))
        edge_type = rel

        data = Data(edge_index=edge_index)
        # data.entity = torch.from_numpy(uniq_entity)
        data.edge_type = edge_type
        # data.edge_norm = self._edge_normalization(edge_type, edge_index, len(uniq_entity), self.n_relation)
        data.samples = torch.from_numpy(samples)
        data.labels = torch.from_numpy(labels)

        data.to(self.device)

        return data
