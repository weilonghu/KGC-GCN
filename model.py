import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from utils import uniform
from metric import score_func


class MGCN(torch.nn.Module):
    """ Model using global and local relation embeddings for multi-graph knowledge graph embedding

    Attributes:
        entity_embedding: resulted entity embeddings
        relation_embedding: resulted relation embeddings, global relation embeddings
        edge_embedding: local relation embeddings
    """
    def __init__(self, num_entities, num_relations, params):
        super(MGCN, self).__init__()

        self.dropout_ratio = params.dropout
        self.emb_dim = params.emb_dim
        self.norm = params.norm
        self.regularization = params.regularization
        self.negative_rate = params.negative_rate

        self.entity_embedding = nn.Embedding(num_entities, self.emb_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, self.emb_dim))

        direction = 2 if params.bi_directional is True else 1

        self.conv1 = MGCNConv(self.emb_dim, self.emb_dim, direction * num_relations, heads=8)
        self.conv2 = MGCNConv(self.emb_dim * 8, self.emb_dim, direction * num_relations, heads=1)

        self.critetion = nn.MarginRankingLoss(margin=params.margin, reduction='mean')

        self.reset_parameter()

    def reset_parameter(self):
        """Initialize embeddings of model with the way used in transE"""
        # uniform_range = 6 / math.sqrt(self.emb_dim)
        # self.entity_embedding.weight.data.uniform_(-uniform_range, uniform_range)
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('leaky_relu'))

    def from_pretrained_emb(self, pretrained_entity, pretrained_relation):
        """Initialize entity embeddings and relation embeddings with pretrained embeddings"""
        self.entity_embedding.from_pretrained(embeddings=pretrained_entity)
        self.relation_embedding.data.copy_(pretrained_relation)

    def forward(self, data):
        """Encode entity in graph 'data' using graph convolutional network

        Args:
            data: (torch_geometric.data.Data) a graph or sub-graph of the KB.
        Return:
            embeddings of the entities in the graph
        """
        entity, edge_index, edge_attr = data.entity, data.edge_index, data.edge_attr

        x = self.entity_embedding(entity)
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_attr)

        return x

    def reg_loss(self, embedding):
        """Regularization loss"""
        loss = torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

        return self.regularization * loss

    def loss_func(self, embedding, triplets, labels):
        """Compute loss for the model"""
        scores = score_func(embedding, self.relation_embedding, triplets)

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
    """The relational graph convolutional operator

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of.
    """

    def __init__(self, in_channels, out_channels, num_relation, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(MGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relation = num_relation
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relation + 1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain('leaky_relu', self.negative_slope))
        nn.init.xavier_uniform_(self.att, nn.init.calculate_gain('leaky_relu', self.negative_slope))
        uniform(self.bias.size(0), self.bias)

    def forward(self, x, edge_index, edge_attr):
        """Perform message passing operator

        Args:
            x: (tensor) node features. shape: [N, in_channels]
            edge_index: (tensor) edges. shape: [2, E]
            edge_attr: (tensor) local edge embeddings. shape: [E, d]
        """
        # add self-loops to the adjacency matrix
        # the edge type of self-loop is the padding relation
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = torch.cat((edge_attr,
                               torch.ones(x.size(0), dtype=edge_attr.dtype, device=edge_attr.device) * self.num_relation))

        x = torch.matmul(x, self.weight)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_index_i, edge_index_j, size_i, edge_attr):
        """
        Construct messages to node i in analogy to ϕ for each edge in (j, i).

        Args:
            x_j: embeddings of node j. shape: [E, d]
            edge_index: respective edges. shape: [E]
            edge_attr: edge embeddings. shape: [E, d]
        """
        # compute attention coefficients
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att[edge_attr]).sum(dim=-1)
        alpha = (x_i * self.att[edge_attr] * x_j).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # sample attention coefficients stochastically
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        """ Return new node embeddings"""
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
