"""Build dataset for graph convolutinal networks"""

import os
import logging
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as GDataLoader


class DataSet:
    def __init__(self, dataset, params):
        self.dataset = dataset
        self.pretrained_entities = params.pretrained_entities
        self.pretrained_relations = params.pretrained_relations
        self.vector_size = params.vector_size

        # whether dataset exists
        assert os.path.exists(os.path.join(
            'data', self.dataset)), 'Dataset {} not found'.format(self.dataset)

        # the following operations only run once
        # load entites and relations
        self.entity2id = self.load_entries(self.dataset)
        self.relation2id = self.load_entries(self.dataset, load_entities=False)

        # initialize representations of entities and relations
        self.entity_repr = np.empty((len(self.entities2idx), self.vector_size))
        self.relation_repr = np.empty(
            (len(self.relations2idx), self.vector_size))

        # if exists pretrained entities or relations, load them
        if type(self.pretrained_entities) == str and len(self.pretrained_entities) > 0:
            assert os.path.exists(
                self.pretrained_entities), 'Pretrained entities not found'
            self.entity_repr = self.init_entry_repr(
                self.pretrained_entities, self.entities2idx)

        if type(self.pretrained_relations) == str and len(self.pretrained_relations) > 0:
            assert os.path.exists(
                self.pretrained_relations), 'Pretrained relations not found'
            self.relation_repr = self.init_entry_repr(
                self.pretrain_relations, self.relations2idx)

    @classmethod
    def data_iterator(self, dataset, batch_size, shuffle=False):
        return GDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def load_data(self, data_type):
        """Load training set, valid set or test set from files"""
        # load triplets for training set, valid set or test set
        triplets = self.load_triplets(self.dataset, data_type)

        # compute graph edges
        edge_index, edge_attr = self.graph_edges(
            self.entities2idx, self.relations2idx, triplets, self.relation_repr)

        # create pytorch_geometric data
        data = Data(
            x=torch.from_numpy(self.entity_repr),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.from_numpy(edge_attr)
        )

        return data

    def load_entries(self, dataset, load_entities=True):
        """Load entities or relations from file, i.e. entities.dict or relations.dict

        Args:
            dataset: (string) dataset name
            load_entites: (bool) whether load entities or relations
        Return:
            entries2idx: (dict) map each entry to a number
        """
        entries = {}
        with open(os.path.join('data', dataset, 'entities.dict' if load_entities else 'relations.dict'), 'r') as f:
            for line in f.readlines():
                eid, entry = line.strip().split()
                entries[entry] = eid

        # logging
        entry_name = 'entities' if load_entities else 'relations'
        logging.info('Load {} {} from file'.format(len(entries), entry_name))

        return entries

    def load_triplets(self, dataset, data_type):
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

        return triplets

    def init_entry_repr(self, filename, entries2idx):
        """Initialize the representations of entries using pretrained model

        Args:
            filename: (string) file containing pretrained representations
            entries2idx: (dict) returned by 'load_entryies' function
        Return:
            entries_repr: (numpy array) representation of entites or relations
        """
        pass

    def graph_edges(self, entities2idx, relations2idx, triplets, relation_repr):
        """Construct the edge_index matrix and edge_attr matrix from triplets

        Args:
            entities2idx: (dict) map each entity to a number or index
            relations2idx: (dict) map each relation to a number or index
            triplets: (list) a list containing triplets
            relation_repr: (2-d array) representations of relations
        Return:
            edge_index: (2-d array) shape: [2, edge_num]
            edge_attr: (2-d array) shape: [edge_num, vector_size]
        """
        # construct two matrix
        edge_index = np.zeros((2, len(triplets)))
        edge_attr = np.zeros((len(triplets), relation_repr.shape[1]))
        for idx, triplet in enumerate(triplets):
            head, relation, tail = triplet
            headidx, tailidx = entities2idx.get(head), entities2idx.get(tail)
            # set edge_index to entity index
            edge_index[0][idx], edge_index[1][idx] = headidx, tailidx
            # set edge_attr to relation representation
            edge_attr[idx][:] = relation_repr[relations2idx.get(relation)][:]

        return edge_index, edge_attr
