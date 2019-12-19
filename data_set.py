"""Build dataset for graph convolutinal networks"""

import os
import logging

import numpy as np
import torch
from torch_geometric.data import Data


class DataSet:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.negative_rate = params.negative_rate
        self.device = params.device
        self.emb_dim = params.emb_dim

        # load pretrained embeddings if need
        if params.load_pretrain:
            self.pretrained_entity = self._load_pretrained_emb(
                os.path.join('data', self.dataset, 'entity2vec.txt')
            )

            self.pretrained_relation = self._load_pretrained_emb(
                os.path.join('data', self.dataset, 'relation2vec.txt')
            )

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
        """Load entities or relations from file, i.e. entity2id.txt or relation2id.txt

        Args:
            dataset: (string) dataset name
            load_entites: (bool) whether load entities or relations
        Return:
            entries2idx: (dict) map each entry to a number
        """
        entries = {}
        entry_file = 'entity2id.txt' if load_entities else 'relation2id.txt'
        with open(os.path.join('data', dataset, entry_file), 'r') as f:
            for line in f.readlines():
                entry, eid = line.strip().split()
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
                head, tail, relation = line.strip().split()
                triplets.append((self.entity2id[head], self.relation2id[relation], self.entity2id[tail]))

        logging.info('Found {} triplets in {}.txt from dataset {}'.format(
            len(triplets), data_type, dataset))

        return np.array(triplets)

    def _load_pretrained_emb(self, emb_file):
        """Load pretrained entity or relation embeddings from file if exists"""
        embeddings = []
        with open(emb_file, 'r') as f:
            for line in f:
                emb = [float(n) for n in line.strip().split()]
                embeddings.append(emb)
        return torch.from_numpy(np.array(embeddings))

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
        """Build graph using training set

        Attributes:
            edge_index: node indexes of edges, shape: [2, E]
            edge_attr: edge attributes, including edge id and edge type, shape: [E, 2]
        """
        src, rel, dst = self.train_triplets.transpose()
        src, rel, dst = torch.from_numpy(src), torch.from_numpy(rel), torch.from_numpy(dst)

        edge_index = torch.stack((src, dst))
        edge_attr = torch.cat((torch.arange(edge_index.size(1)).view(1, -1), rel.view(1, -1)), dim=0).transpose(0,1)
        data = Data(edge_index=edge_index, edge_attr=edge_attr)

        # add entity ids and mask
        data.x = torch.arange(self.n_entity)
        data.train_mask = torch.ones(self.n_entity, dtype=torch.bool)

        return data


def negative_sampling(pos_samples, num_entity, negative_rate, device):
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
        labels = torch.zeros(size * (negative_rate + 1), dtype=torch.float).to(device)
        labels[: size] = 1  # labels of positive samples

        # sample negative samples
        neg_samples = pos_samples.repeat(negative_rate, 1)
        values = torch.zeros(size * negative_rate, dtype=torch.int64).random_(0, num_entity)
        choices = torch.zeros(size * negative_rate).uniform_(0, 1)
        values = values.to(device)
        choices = choices.to(device)
        subj, obj = choices > 0.5, choices <= 0.5
        # subj, obj = subj.nonzero().view(-1), obj.nonzero().view(-1)
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        samples = torch.cat((pos_samples, neg_samples), dim=0)
        return samples, labels
