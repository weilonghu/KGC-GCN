"""
Knowledge Graph Dataset, e.g. WN18, FB15k and FB15k-237.
The dataset contains following files:
    -- entity2id.txt: two columns, one for entity symbol, one for entity id
    -- relation2id.txt: two columns, one for relation symbol, one for relation id
    -- train.txt: three columns, head entity, tail entity and relation
    -- valid.txt: same as 'train.txt'
    -- test.txt: same as 'train.txt'
    -- entity2vec.txt: (optional), pretrained entity embeddings
    -- relation2vec.txt: (optional) pretrained relation embeddings
All columns are splited by '\t'.

For clarification, all loaded data are numpy.ndarray instead of torch.Tensor.
"""

import os
import logging

import numpy as np
from torch.utils import data
from torch_geometric.data import Data


class DataSet(data.Dataset):
    """Knowledge graph datast

    Attributes:
        entity2id: (dict) map entity to id
        relation2id: (dict) map relation to id
        train_triplets: (numpy.ndarray) triplets in training set
        valid_triplets: (numpy.ndarray) triplets in validation set
        test_triplets: (numpy.ndarray) triplets in test set
        pretrain_entity: (numpy.ndarray) pretrained entity embeddings, optional
        pretrain_relation: (numpy.ndarray) pretrained relation embeddings, optional
    """

    def __init__(self, dataset):
        self.data_dir = os.path.join('data', dataset)

        # whether dataset exists
        assert os.path.exists(self.data_dir), 'Dataset {} not found'.format(self.dataset)

        # load entites and relations
        self.entity2id = self._load_entries(os.path.join(self.data_dir, 'entity2id.txt'))
        self.relation2id = self._load_entries(os.path.join(self.data_dir, 'relation2id.txt'))

        # load triplets
        self.triplets = {
            'train': self._load_triplets(self.data_dir, 'train'),
            'val': self._load_triplets(self.data_dir, 'valid'),
            'test': self._load_triplets(self.data_dir, 'test')
        }

        # load pretrained embeddings if exist
        if os.path.exists(os.path.join(self.data_dir, 'entity2vec.txt')):
            self.pretrain_entity = self._load_pretrained_emb(os.path.join(self.data_dir, 'entity2vec.txt'))

        if os.path.exists(os.path.join(self.data_dir, 'relation2vec.txt')):
            self.pretrain_relation = self._load_pretrained_emb(os.path.join(self.data_dir, 'relation2vec.txt'))
        
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)

        self.train_indices = list(range(len(self.triplets['train'])))

        # report the dataset
        logging.info('num_entities: {}'.format(self.num_entity))
        logging.info('num_relations: {}'.format(self.num_relation))
        logging.info('num_train_triplets: {}'.format(self.triplets['train'].shape[0]))
        logging.info('num_valid_triplets: {}'.format(self.triplets['val'].shape[0]))
        logging.info('num_test_triplets: {}'.format(self.triplets['test'].shape[0]))

    def _load_entries(self, filepath):
        """Load entities or relations from file, i.e. entity2id.txt or relation2id.txt"""

        entries = {}
        with open(filepath, 'r') as f:
            for line in f:
                entry, eid = line.strip().split()
                entries[entry] = int(eid)

        return entries

    def _load_triplets(self, data_dir, data_type):
        """Load triplets from train.txt, valid.txt or test.txt according to 'data_type' argument

        Args:
            data_dir: (string) directory contain dataset
            data_type: (string)  'train', 'valid' or 'test'
        Return:
            numpy.ndarray. shape=[num_triplets, 3]
        """
        assert data_type in ['train', 'valid',
                             'test'], 'Invalid data type when loading triplets'

        triplets = []
        with open(os.path.join(data_dir, data_type + '.txt'), 'r') as f:
            for line in f:
                head, tail, relation = line.strip().split()
                triplets.append((self.entity2id[head], self.relation2id[relation], self.entity2id[tail]))

        return np.array(triplets)

    def _load_pretrained_emb(self, emb_file):
        """Load pretrained entity or relation embeddings from file if exists"""
        embeddings = []
        with open(emb_file, 'r') as f:
            for line in f:
                emb = [float(n) for n in line.strip().split()]
                embeddings.append(emb)
        return np.array(embeddings)

    def total_triplets(self):
        """Concatenate all triplets from training set, valid set and test set"""
        return np.concatenate((
            self.triplets['train'], self.triplets['val'], self.triplets['test']
        ))

    def __len__(self):
        return len(self.train_indices)

    def __getitem__(self, index):
        """Only reture the selected index"""
        return self.train_indices[index]
