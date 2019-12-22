"""Interface used for dataset and batch-wise training"""

import os

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch_scatter import scatter_add

from data_set import DataSet


class DataManager(object):
    def __init__(self, dataset, params):
        self.data_set = DataSet(dataset)
        self.split_size = params.split_size
        self.negative_rate = params.negative_rate

        self.num_entity = self.data_set.num_entity
        self.num_relation = self.data_set.num_relation

        self.num_worker = params.num_worker

    def all_triplets(self):
        """Get all triplets for evaluation"""
        all_triplets = self.data_set.total_triplets()
        return torch.from_numpy(all_triplets)

    def fetch_triplets(self, data_type):
        """Get triplets accordding to 'data_type'"""
        assert data_type in ['train', 'val', 'test'], 'Invalid data type'

        return torch.from_numpy(self.data_set.triplets[data_type])

    def _self_collate_fn(self, batch):
        """Perform negative sampling and create sub-graph

        Args:
            batch: (numpy.ndarray)
        Return:
            torch_geometric.data.Data
        """
        src, rel, dst = np.array(batch).transpose()
        uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # Negative sampling
        samples, labels = self._negative_sampling(relabeled_edges, len(uniq_entity), self.negative_rate)

        # further split graph, only half of the edges will be used as graph
        # structure, while the rest half is used as unseen positive samples
        split_size = int(src.shape[0] * self.split_size)
        graph_split_ids = np.random.choice(np.arange(src.shape[0]),
                                        size=split_size, replace=False)

        src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
        dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
        rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

        # Create bi-directional graph
        # src, dst = torch.cat((src, dst)), torch.cat((dst, src))
        # rel = torch.cat((rel, rel + self.num_relation))

        edge_index = torch.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index = edge_index, edge_attr=edge_attr)
        data.entity = torch.from_numpy(uniq_entity)
        data.samples = torch.from_numpy(samples)
        data.labels = torch.from_numpy(labels)

        return data

    def data_iterator(self, batch_size, shuffle=True, drop_last=True):
        """Create a pytorch DataLoader object"""

        # zero worker when remote development
        method = os.environ.get('MULTIPROCESS_METHOD', None)
        num_workers = self.num_worker
        return DataLoader(self.data_set,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=self._self_collate_fn,
                          num_workers=num_workers,
                          drop_last=drop_last)

    def neighbor_sampler(self, batch_size, size=1, shuffle=True, num_hops=1, drop_last=True):
        """The neighbor sampler from the “Inductive Representation Learning on Large Graphs” paper implemented by torch_geometric

        Args:
            size: (int or [int]) The number of neighbors to sample (for each layer)
            num_hops: (int) The number of layers to sample
        Return:
            torch_geometric.data.Data object.
        """
        train_data = self.build_test_graph()
        loader = NeighborSampler(train_data,
                                 size=size,
                                 num_hops=num_hops,
                                 bipartite=False,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 add_self_loops=False,
                                 drop_last=drop_last)
        # create a generator for negative sampling and consistent interface
        def generator():
            for data in loader():
                n_id, e_id, edge_index = data.n_id, data.e_id, data.edge_index
                # construct batch triplets
                head, tail = edge_index[0], edge_index[1]
                rel = train_data.edge_attr[e_id]
                # negtive sampling
                pos_samples = torch.cat((head.view(-1, 1), rel.view(-1, 1), tail.view(-1, 1)), dim=1)
                samples, labels = self._negative_sampling(pos_samples, n_id.size(0), self.negative_rate)

                n_data = Data(edge_index = edge_index, edge_attr=rel)
                n_data.entity = n_id
                n_data.samples = torch.from_numpy(samples)
                n_data.labels = torch.from_numpy(labels)

                yield n_data

        return generator()

    """
    Functions used to build graph
    """
    def _negative_sampling(self, pos_samples, num_entity, negative_rate):
        """Sample negative triplets for training

        Args:
            pos_sample: positive samples
            num_entity: (int) the number of entities
            negative_rate: (int) the proportion of negative samples
        Return:
            samples: (numpy.ndarray) samples including positive and negative samples
            labels: (numpy.ndarray) labels whose values in [0, 1]
        """
        size = len(pos_samples)
        # genrate labels
        labels = np.zeros(size * (negative_rate + 1), dtype=np.float32)
        labels[: size] = 1  # labels of positive samples

        # sample negative samples
        neg_samples = np.tile(pos_samples, (negative_rate, 1))
        values = np.random.choice(num_entity, size=size * negative_rate)
        choices = np.random.uniform(size=size * negative_rate)
        subj, obj = choices > 0.5, choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        return np.concatenate((pos_samples, neg_samples)), labels

    def _edge_normal(self, edge_type, edge_index, num_entity, num_relation):
        '''
            Edge normalization trick
            - one_hot: (num_edge, num_relation)
            - deg: (num_node, num_relation)
            - index: (num_edge)
            - deg[edge_index[0]]: (num_edge, num_relation)
            - edge_norm: (num_edge)
        '''
        one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
        deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
        index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
        edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

        return edge_norm

    def build_test_graph(self):
        src, rel, dst = self.fetch_triplets('train').transpose(0, 1)

        # src, dst = torch.cat((src, dst)), torch.cat((dst, src))
        # rel = torch.cat((rel, rel + self.num_relation))

        edge_index = torch.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index = edge_index, edge_attr=edge_attr)
        data.entity = torch.from_numpy(np.arange(self.num_entity))

        return data
