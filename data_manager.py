"""Interface used for dataset and batch-wise training"""

import os
import logging

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch_scatter import scatter_add
from torch_geometric.utils import remove_isolated_nodes

from data_set import DataSet
from utils import MakeIter


class DataManager(object):
    def __init__(self, dataset, params):
        self.data_set = DataSet(dataset)
        self.split_size = params.split_size
        self.negative_rate = params.negative_rate
        self.batch_size = params.batch_size
        self.bi_directional = params.bi_directional

        self.num_entity = self.data_set.num_entity
        self.num_relation = self.data_set.num_relation

        try:
            self.pretrained_entity = torch.from_numpy(self.data_set.pretrain_entity)
            self.pretrained_relation = torch.from_numpy(self.data_set.pretrain_relation)
        except AttributeError:
            logging.info('Pretrained embeddings not found')

        self.num_worker = params.num_worker

    def all_triplets(self):
        """Get all triplets for evaluation"""
        all_triplets = self.data_set.total_triplets()
        return torch.from_numpy(all_triplets)

    def fetch_triplets(self, data_type):
        """Get triplets accordding to 'data_type'"""
        assert data_type in ['train', 'val', 'test'], 'Invalid data type'

        return torch.from_numpy(self.data_set.triplets[data_type])

    #######################################################################
    #
    # Utility function for building training and testing graphs
    #
    #######################################################################

    def _self_collate_fn(self, batch):
        """Perform negative sampling and create sub-graph

        Args:
            batch: ([numpy.ndarray])
        Return:
            torch_geometric.data.Data
        """
        train_triplets = self.fetch_triplets('train').numpy()
        edges = np.array(batch)

        return self._generate_sampled_graph_and_labels(train_triplets[edges])

    def _generate_sampled_graph_and_labels(self, edges):
        """Get training graph and signals,
        Fisrt perform edge neiborhood sampling on grapph,
        then perform negative sampling to generate negative samples.
        """
        src, rel, dst = edges.transpose()
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
        if self.bi_directional is True:
            src, dst = torch.cat((src, dst)), torch.cat((dst, src))
            rel = torch.cat((rel, rel + self.num_relation))

        edge_index = torch.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index = edge_index, edge_attr=edge_attr)
        data.entity = torch.from_numpy(uniq_entity)
        data.num_nodes = len(uniq_entity)
        data.edge_norm = self._edge_normal(edge_attr, edge_index, len(uniq_entity), self.num_relation)
        data.samples = torch.from_numpy(samples)
        data.labels = torch.from_numpy(labels)

        return data

    def _build_graph_from_triplets(self):
        pass

    def _negative_sampling(self, pos_samples: np.ndarray, num_entity, negative_rate: int):
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
        # one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
        # deg = scatter_add(one_hot, edge_index[1], dim = 0, dim_size = num_entity)
        # index = edge_type + torch.arange(len(edge_index[1])) * (2 * num_relation)
        # edge_norm = 1 / deg[edge_index[1]].view(-1)[index]

        counts = torch.ones_like(edge_type).to(torch.float)
        deg = scatter_add(counts, edge_index[1], dim_size=num_entity)
        edge_norm = 1 / deg[edge_index[1]]
        edge_norm[torch.isinf(edge_norm)] = 0

        return edge_norm

    def _get_adj_and_degrees(self, num_nodes: int, triplets: np.ndarray):
        """ Get adjacency list and degrees of the graph"""
        adj_list = [[] for _ in range(num_nodes)]
        for i,triplet in enumerate(triplets):
            adj_list[triplet[0]].append([i, triplet[2]])
            adj_list[triplet[2]].append([i, triplet[0]])

        degrees = np.array([len(a) for a in adj_list])
        adj_list = [np.array(a) for a in adj_list]
        return adj_list, degrees

    def _sample_edge_neighborhood(self, adj_list, degrees, n_triplets, sample_size):
        """Sample edges by neighborhool expansion.
        This guarantees that the sampled edges form a connected graph, which
        may help deeper GNNs that require information from more than one hop.
        """
        edges = np.zeros((sample_size), dtype=np.int32)

        #initialize
        sample_counts = np.array([d for d in degrees])
        picked = np.array([False for _ in range(n_triplets)])
        seen = np.array([False for _ in degrees])

        for i in range(0, sample_size):
            weights = sample_counts * seen

            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
                weights[np.where(sample_counts == 0)] = 0

            probabilities = (weights) / np.sum(weights)
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                            p=probabilities)
            chosen_adj_list = adj_list[chosen_vertex]
            seen[chosen_vertex] = True

            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

            while picked[edge_number]:
                chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
                chosen_edge = chosen_adj_list[chosen_edge]
                edge_number = chosen_edge[0]

            edges[i] = edge_number
            other_vertex = chosen_edge[1]
            picked[edge_number] = True
            sample_counts[chosen_vertex] -= 1
            sample_counts[other_vertex] -= 1
            seen[other_vertex] = True

        return edges

    def _sample_edge_uniform(self, n_triplets: int, sample_size: int):
        """Sample edges uniformly from all the edges."""
        all_edges = np.arange(n_triplets)
        return np.random.choice(all_edges, sample_size, replace=False)

    #######################################################################
    #
    # Utility function for building training and testing graphs
    #
    #######################################################################

    def build_train_graph(self):
        """Build train graph, only one batch"""
        # sample edges uniformly from all edges
        train_triplets = self.fetch_triplets('train').numpy()
        all_edges = np.arange(len(train_triplets))
        edges = np.random.choice(all_edges, self.batch_size, replace=False)

        return self._generate_sampled_graph_and_labels(train_triplets[edges])

    def build_test_graph(self):
        src, rel, dst = self.fetch_triplets('train').transpose(0, 1)

        # Create bi-directional graph
        if self.bi_directional is True:
            src, dst = torch.cat((src, dst)), torch.cat((dst, src))
            rel = torch.cat((rel, rel + self.num_relation))

        edge_index = torch.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index = edge_index, edge_attr=edge_attr)
        data.entity = torch.from_numpy(np.arange(self.num_entity))
        data.num_nodes = self.num_entity
        data.edge_norm = self._edge_normal(edge_attr, edge_index, self.num_entity, self.num_relation)

        return data

    def data_iterator(self, batch_size, shuffle=True, drop_last=False):
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
        # create graph
        src, rel, dst = self.fetch_triplets('train').transpose(0, 1)

        edge_index = torch.stack((src, dst))
        edge_attr = rel

        train_data = Data(edge_index = edge_index, edge_attr=edge_attr)
        train_data.entity = torch.from_numpy(np.arange(self.num_entity))
        train_data.num_nodes = self.num_entity

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
                # the neighborsampler first select b_ids as tail nodes, then select head nodes
                # if one node only present at head nodes, it becomes as a isolated node
                edge_index, rel, mask = remove_isolated_nodes(edge_index, edge_attr=train_data.edge_attr[e_id], num_nodes=n_id.size(0))
                n_id = n_id[mask]
                # construct batch triplets
                src, dst = edge_index[0], edge_index[1]
                # negtive sampling
                pos_samples = torch.cat((src.view(-1, 1), rel.view(-1, 1), dst.view(-1, 1)), dim=1)
                # make sure 'pos_samples' and 'neg_samples' are the same type, or they will share memory
                samples, labels = self._negative_sampling(pos_samples.numpy(), n_id.size(0), self.negative_rate)

                # Create bi-directional graph
                if self.bi_directional is True:
                    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
                    rel = torch.cat((rel, rel + self.num_relation))

                edge_index = torch.stack((src, dst))
                edge_attr = rel

                n_data = Data(edge_index = edge_index, edge_attr=edge_attr)
                n_data.entity = n_id
                n_data.samples = torch.from_numpy(samples)
                n_data.labels = torch.from_numpy(labels)
                n_data.num_nodes = n_id.size(0)
                n_data.edge_norm = self._edge_normal(edge_attr, edge_index, n_data.num_nodes, self.num_relation)

                yield n_data

        return MakeIter(generator)
