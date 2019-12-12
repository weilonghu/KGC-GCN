"""Build dataset for graph convolutinal networks"""

from __future__ import print_function
import argparse
import os
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='wn18',
                    help='Directory containing the dataset')
parser.add_argument('--pretrained_entities', default=None,
                    help='Pretrained representions of entities')
parser.add_argument('--pretrained_relations', default=None,
                    help='Pretrained representions of relations')
parser.add_argument('--vector_size', default=50,
                    help='Dimension of entities and relation')


def load_entries(dataset, load_entities=True):
    """Load entities or relations from file, i.e. entities.dict or relations.dict

    Args:
        dataset: (string) dataset name
        load_entites: (bool) whether load entities or relations
    Return:
        entries2idx: (dict) map each entry to a number
    """
    entries = set()
    entries_num = 0
    with open(os.path.join('data', dataset, 'entities.dict' if load_entities else 'relations.dict'), 'r') as f:
        for line in f.readlines():
            entry = line.strip().split()[-1]
            entries.add(entry)
            entries_num += 1
    # check the number of entries
    assert len(entries) == entries_num, 'There exists duplicated entries'
    entries2idx = {entity: idx for idx, entity in enumerate(entries)}

    # logging
    entry_name = 'entities' if load_entities else 'relations'
    print('Load {} {} from file'.format(entries_num, entry_name))

    return entries2idx


def load_triplets(dataset, data_type):
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
            triplets.append((head, relation, tail))

    print('Found {} triplets in {}.txt from dataset {}'.format(
        len(triplets), data_type, dataset))

    return triplets


def init_entry_repr(filename, entries2idx):
    """Initialize the representations of entries using pretrained model

    Args:
        filename: (string) file containing pretrained representations
        entries2idx: (dict) returned by 'load_entryies' function
    Return:
        entries_repr: (numpy array) representation of entites or relations
    """
    pass


def graph_edges(entities2idx, relations2idx, triplets, relation_repr):
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


def dump(entities2idx, relations2idx, entity_repr, relation_repr, triplets, filepath):
    """Dump graph information to file, including 'train.bin', 'valid.bin' and 'test.bin'"""
    edge_index, edge_attr = graph_edges(
        entities2idx, relations2idx, triplets, relation_repr)
    data = {
        'edge_attr': edge_attr,
        'edge_index': edge_index,
        'entity_repr': entity_repr,
        'relation_repr': relation_repr,
        'cls': np.zeros(len(entities2idx)),
        'node_num': len(entities2idx),
        'edge_num': len(relations2idx)
    }
    pickle.dump(data, open(filepath, 'wb'))


if __name__ == '__main__':
    args = parser.parse_args()

    # whether dataset exists
    assert os.path.exists(os.path.join('data', args.dataset)
                          ), 'Dataset {} not found'.format(args.dataset)

    # load entites and relations
    entities2idx, relations2idx = load_entries(
        args.dataset), load_entries(args.dataset, load_entities=False)

    # initialize representations of entities and relations
    if args.pretrained_entities is not None:
        entity_repr = init_entry_repr(args.pretrained_entities, entities2idx)
    else:
        entity_repr = np.empty((len(entities2idx), args.vector_size))

    if args.pretrained_relations is not None:
        relation_repr = init_entry_repr(args.pretrain_relations, relations2idx)
    else:
        relation_repr = np.empty((len(relations2idx), args.vector_size))

    # load triplets for training set, valid set and test set
    train_triplets = load_triplets(args.dataset, 'train')
    valid_triplets = load_triplets(args.dataset, 'valid')
    test_triplets = load_triplets(args.dataset, 'test')

    # dump to files
    dump(entities2idx, relations2idx, entity_repr, relation_repr,
         train_triplets, os.path.join('data', args.dataset, 'train.bin'))
    dump(entities2idx, relations2idx, entity_repr, relation_repr,
         valid_triplets, os.path.join('data', args.dataset, 'valid.bin'))
    dump(entities2idx, relations2idx, entity_repr, relation_repr,
         test_triplets, os.path.join('data', args.dataset, 'test.bin'))
