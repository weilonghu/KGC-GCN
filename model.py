import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import uniform


class MGCN(torch.nn.Module):
    """ Model using global and local relation embeddings for multi-graph knowledge graph embedding
    Attributes:
        entity_embedding: resulted entity embeddings
        relation_embedding: resulted relation embeddings, global relation embeddings
        edge_embedding: local relation embeddings
    """
    def __init__(self, num_entities, num_relations, params):
        super(MGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = MGCNConv(
            100, 100, num_relations * 2, num_bases=8)
        self.conv2 = MGCNConv(
            100, 100, num_relations * 2, num_bases=8)

        self.dropout_ratio = params.dropout
        self.regularization = params.regularization

    def forward(self, data):
        """Encode entity in graph 'data' using graph convolutional network
        Args:
            data: (torch_geometric.data.Data) a graph or sub-graph of the KB.
        Return:
            embeddings of the entities in the graph
        """
        entity, edge_index, edge_type, edge_norm = data.entity, data.edge_index, data.edge_attr, data.edge_norm

        x = self.entity_embedding(entity)
        x = self.conv1(x, edge_index, edge_type, edge_norm)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)

        return x

    def score_func(self, embedding, triplets):
        """Scoring each triplet in 'triplets'
        Args:
            embedding: (torch.FloatTensor), entity embeddings used in triplets
            triplets: (torch.LongTensor), triplets for scoring
        Return:
            torch.FloatTensor, one score for each triplet
        """
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        # scores = torch.norm(s + r - o, p=self.norm, dim=1)
        scores = torch.sum(s * r * o, dim=1)

        return scores

    def reg_loss(self, embedding):
        """Regularization loss"""
        loss = torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

        return self.regularization * loss

    def loss_func(self, embedding, triplets, labels):
        """Compute loss for the model"""
        scores = self.score_func(embedding, triplets)

        # compute margin loss
        # pos_size = scores.size(0) // (1 + self.negative_rate)
        # pos_scores, neg_scores = scores[:pos_size], scores[pos_size:]
        # pos_scores = pos_scores.repeat((1, self.negative_rate)).view(-1)
        # target = torch.tensor([-1], dtype=torch.long, device=scores.device)
        # loss = self.critetion(pos_scores, neg_scores, target)

        # compute cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # compute accuracy based on distmult
        logits = torch.sigmoid(scores)
        pred = logits >= 0.5
        true = labels >= 0.5
        acc = torch.mean(pred.eq(true).float())

        # comput accuracy based on transE
        # pred = pos_scores < neg_scores
        # acc = torch.mean(pred.float())

        loss += self.reg_loss(embedding)

        return loss, acc


class MGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,
    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(MGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.weight = nn.Parameter(torch.Tensor(1, out_channels))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)
        uniform(size, self.weight)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_i, edge_index_j, edge_index, edge_type, edge_norm):
        alpha = (x_i * self.weight * x_j).sum(dim=1)
        alpha = softmax(alpha, edge_index_i, edge_index_i.size(0))

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return (out * edge_norm.view(-1, 1) + out * alpha.view(-1, 1)) / 2

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
