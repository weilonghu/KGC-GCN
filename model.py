import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import utils


class MGCN(torch.nn.Module):
    """ Model using global and local relation embeddings for multi-graph knowledge graph embedding

    Attributes:
        entity_embedding: resulted entity embeddings
        relation_embedding: resulted relation embeddings, global relation embeddings
        edge_embedding: local relation embeddings
    """
    def __init__(self, num_entities, num_relations, num_edges, params):
        super(MGCN, self).__init__()

        self.dropout_ratio = params.dropout
        self.emb_dim = params.emb_dim
        self.norm = params.norm

        self.entity_embedding = nn.Embedding(num_entities, self.emb_dim)
        self.relation_embedding = nn.Embedding(num_relations, self.emb_dim)
        self.edge_embedding = nn.Embedding(num_edges, self.emb_dim)

        self.conv1 = MGCNConv(self.emb_dim, self.emb_dim, num_relations)
        self.conv2 = MGCNConv(self.emb_dim, self.emb_dim, num_relations)

        self.critetion = nn.MarginRankingLoss(margin=params.margin, reduction='mean')

        self._init_embedding()

    def _init_embedding(self):
        """Initialize embeddings of model with the way used in transE"""
        uniform_range = 6 / math.sqrt(self.emb_dim)
        self.entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)
        self.relation_embedding.weight.data.uniform_(-uniform_range, uniform_range)
        self.edge_embedding.weight.data.uniform_(-uniform_range, uniform_range)

    def from_pretrained_emb(self, pretrained_entity, pretrained_relation):
        """Initialize entity and relation embeddings using pretrained embeddings"""
        self.entity_embedding.from_pretrained(pretrained_entity)
        self.relation_embedding.from_pretrained(pretrained_relation)

    def forward(self, edge_attr, data):
        """Compute loss of using the embeddings and triplets

        Args:
            edge_index: edges of the sampled graph. shape: [2, E]
            edge_type: relation types of each edge. shape: [E]
            edge_ids: edge indices in triplets file. shape: [E]
        """

        edge_ids, edge_types = edge_attr[data.e_id].transpose(0, 1)
        edge_attr = self.relation_embedding(edge_types)

        x = self.entity_embedding(data.n_id)
        x = self.conv1(x, data.edge_index, edge_attr, edge_types)
        x = self.conv1(x, data.edge_index, edge_attr, edge_types)
        # x = F.relu(self.conv1(x, data.edge_index, edge_attr, edge_types))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, data.edge_index, edge_attr, edge_types)

        return x, data.n_id, data.e_id, data.edge_index

    def score_func(self, embedding, triplets):
        """Scoring each triplet in 'triplets'

        Args:
            embedding: (torch.FloatTensor), entity embeddings used in triplets
            triplets: (torch.LongTensor), triplets for scoring
        Return:
            torch.FloatTensor, one score for each triplet
        """
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding(triplets[:, 1])
        o = embedding[triplets[:, 2]]
        scores = torch.norm(s + r - o, p=self.norm, dim=1)
        # scores = torch.sum(s * o * r, dim=1)

        return scores

    def loss_func(self, embedding, triplets, labels):
        """Compute loss for the model"""
        scores = self.score_func(embedding, triplets)

        # compute margin loss
        pos_scores, neg_scores = scores.view(2, -1)
        target = torch.tensor([-1], dtype=torch.long, device=scores.device)
        loss = self.critetion(pos_scores, neg_scores, target)
        # loss = F.binary_cross_entropy_with_logits(scores, labels)

        # compute accuracy
        acc = torch.mean((pos_scores < neg_scores).float())

        return loss, acc


class MGCNConv(MessagePassing):
    """The relational graph convolutional operator

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of.
    """

    def __init__(self, in_channels, out_channels, num_relations, **kwargs):
        super(MGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.lin1 = torch.nn.Linear(in_channels, out_channels)

        self.relation_matrix = nn.Parameter(torch.Tensor(num_relations + 1, in_channels, out_channels))
        utils.uniform(in_channels, self.relation_matrix)

    def forward(self, x, edge_index, edge_attr, edge_type):
        """Perform message passing operator

        Args:
            x: (tensor) node features. shape: [N, in_channels]
            edge_index: (tensor) edges. shape: [2, E]
            edge_attr: (tensor) local edge embeddings. shape: [E, d]
        """
        # add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # edge_attr = torch.cat((edge_attr, self.loop_emb.repeat(x.size(0), 1)), dim=0)
        edge_type = torch.cat((edge_type, torch.ones(x.size(0), dtype=edge_type.dtype, device=edge_type.device) * self.num_relations))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr, edge_type=edge_type)

    def message(self, x_j, edge_index, size, edge_attr, edge_type):
        """
        Construct messages to node i in analogy to ϕ for each edge in (j, i).

        Args:
            x_j: embeddings of node j. shape: [E, d]
            edge_index: respective edges. shape: [E]
            edge_attr: edge embeddings. shape: [E, d]
        """
        # transform node features using edge features
        # x_j = x_j + edge_attr
        x_j = torch.matmul(x_j.unsqueeze_(dim=1), self.relation_matrix[edge_type])
        x_j = torch.squeeze(x_j, dim=1)
        return x_j

        # normalize node feature
        # row, col = edge_index
        # deg = degree(row, size[0], dtype=x_j.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        """ Return new node embeddings, aggr_out has shape [N, out_channels]"""
        return aggr_out
