r"""
Interface used for dataset and batch-wise training.
For classification, we use numpy to process data and output tensor for model unless special operations like 'scatter_add'.

There are three types of DataLoader:
    1. The way used in the code of 'Modeling Relational Data with Graph Convolutional Networks', including 'uniform edge sampling'
        and 'neighborhood edge sampling'. In this way, we first sample edges from the whole graph to form a sub-graph for training.
    2. NeighborSampler implemented by 'torch_geometric'.
    3. Traditional batch-wise loader like 'uniform edge sampling', but each epoch contains multiple batches and each batch forms a sug-graph.
Evary DataLoader must can be iterable. So the every epoch, we can use following code to train our model:
    for data in loader:
        loss = model(data)
        ......
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_geometric.utils import degree

from data_set import DataSet


class DataManager(object):
    """ API for createing graph used in training evaluating

    Attributes:
        data_set: (data_set.DataSet) knowledge graph dataset containing triplets
            and pretrained embeddings.
        data_loader: (iterable) iterable data loader can be used in one epoch
    """
    def __init__(self, dataset, params):
        self.data_set = DataSet(dataset)
        self.split_size = params.split_size
        self.negative_rate = params.negative_rate
        self.batch_size = params.batch_size
        self.bi_directional = params.bi_directional

        self.num_entity = self.data_set.num_entity
        self.num_relation = self.data_set.num_relation

        self.edge_sampler = Sampler(self.fetch_triplets('train'), self.num_entity, self.fetch_triplets('train').shape[0])

    def all_triplets(self):
        """Get all triplets for evaluation"""
        all_triplets = self.data_set.total_triplets()
        return all_triplets

    def fetch_triplets(self, data_type, size=1):
        """Get triplets accordding to 'data_type'

        Args:
            data_type: (string) type can be choosed from ['train', 'va', 'test']
        Return:
            triplets: (np.ndarray) shape=[N, 3]
        """
        assert data_type in ['train', 'val', 'test'], 'Invalid data type'

        triplets = self.data_set.triplets[data_type]
        if size == 1:
            return triplets
        else:
            edges = np.random.choice(np.arange(triplets.shape[0]), size=int(size * triplets.shape[0]), replace=True)
            return triplets[edges]

    # =====================================================================
    # Utility functions only used in this class
    # =====================================================================

    def _generate_sampled_graph_and_labels(self, edges):
        """Get training graph and signals, Fisrt perform edge neiborhood sampling on grapph,
        then perform negative sampling to generate negative samples.

        Args:
            edges: (np.ndarray) sampled edges from a graph. shape=[N, 3]
        Return:
            torch_geometric.data.Data represent a graph
        """
        # relabel nodes to have consecutive node ids
        src, rel, dst = edges.transpose()
        uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # Negative sampling
        samples, labels = self._negative_sampling(relabeled_edges, len(uniq_entity), self.negative_rate)

        # edges dropout
        split_size = int(src.shape[0] * self.split_size)
        graph_split_ids = np.random.choice(np.arange(src.shape[0]), size=split_size, replace=False)
        src, rel, dst = src[graph_split_ids], rel[graph_split_ids], dst[graph_split_ids]

        data = self._build_graph_from_triplets(uniq_entity, src, rel, dst, self.bi_directional)

        data.samples = torch.from_numpy(samples)
        data.labels = torch.from_numpy(labels)

        return data

    def _build_graph_from_triplets(self, graph_nodes, src, rel, dst, bi_directional=False):
        """Create a graph give triplets consists of 'src', 'rel' and 'dst'

        Args:
            graph_nodes: (np.ndarray) nodes ids in the target graph
            src, rel, dst: (np.ndarray) nodes and relation ids in the graph
        Return:
            torch_geometric.data.Data
        """
        # Create bi-directional graph
        if bi_directional is True:
            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relation))

        edge_index = np.stack((src, dst))
        edge_attr = rel

        data = Data(edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr))
        data.entity = torch.from_numpy(graph_nodes)
        data.num_nodes = len(graph_nodes)
        data.edge_norm = self._edge_normal(edge_attr, edge_index, len(graph_nodes), self.num_relation)

        return data

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
        # one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
        # deg = scatter_add(one_hot, edge_index[1], dim=0, dim_size=num_entity)
        # index = edge_type + torch.arange(len(edge_index[1])) * (2 * num_relation)
        # edge_norm = 1 / deg[edge_index[1]].view(-1)[index]

        edge_type, edge_index = torch.from_numpy(edge_type), torch.from_numpy(edge_index)
        counts = torch.ones_like(edge_type).to(torch.float)
        deg = scatter_add(counts, edge_index[1], dim_size=num_entity)
        edge_norm = 1 / deg[edge_index[1]]
        edge_norm[torch.isinf(edge_norm)] = 0

        return edge_norm

    # ===============================================================================
    # Utility function for building training and testing graphs
    # ===============================================================================

    def build_train_graph(self, method):
        """Build train graph, only one batch"""

        train_triplets = self.fetch_triplets('train')
        edges = self.edge_sampler(method, self.batch_size)

        return self._generate_sampled_graph_and_labels(train_triplets[edges])

    def build_test_graph(self):
        src, rel, dst = self.fetch_triplets('train').transpose()

        data = self._build_graph_from_triplets(np.arange(self.num_entity), src, rel, dst, self.bi_directional)

        return data

    def get_data_loader(self, sampler_method):

        data = self.build_train_graph(sampler_method)

        return iter([data])


class Sampler(object):

    def __init__(self, triplets, num_node, num_edge, K=1):
        self.num_node = num_node
        self.num_edge = num_edge
        self.K = K

        self.weights = self._calculate_degrees(triplets)
        self.counts = np.ones(num_edge)

        self.func_map = {
            'uniform': self._sample_edge_uniform,
            'weighted': self._sample_edge_weighted
        }

    def __call__(self, method, size):
        assert method in self.func_map, 'Unspported sampling method'

        func = self.func_map.get(method)
        return func(size)

    def _calculate_degrees(self, triplets):
        triplets = torch.from_numpy(triplets)
        src, rel, dst = triplets.transpose(0, 1)
        node_degrees = degree(torch.cat((src, dst)), num_nodes=self.num_node)
        edge_degrees = node_degrees[src] + node_degrees[dst]

        return edge_degrees.numpy()

    def _sample_edge_uniform(self, size):
        """Sample edges uniformly from all the edges."""
        all_edges = np.arange(self.num_edge)
        return np.random.choice(all_edges, size, replace=False)

    def _sample_edge_weighted(self, size):
        norm_weights = self.weights / np.sum(self.weights)
        norm_counts = self.counts / np.sum(self.counts)

        p = self.K * norm_weights / np.power(norm_counts, 1 / 2)
        p = np.exp(p) / np.sum(np.exp(p))

        edges = np.random.choice(np.arange(self.num_edge), size, replace=False, p=p)
        self.counts[edges] += 1

        return edges
