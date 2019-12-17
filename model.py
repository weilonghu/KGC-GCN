import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MGCN(torch.nn.Module):
    """ Model using global and local relation embeddings for multi-graph knowledge graph embedding

    Attributes:
        entity_embedding: resulted entity embeddings
        relation_embedding: resulted relation embeddings, global relation embeddings
        edge_embedding: local relation embeddings
    """
    def __init__(self, num_entities, num_relations, num_edges, emb_dim, dropout):
        super(MGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, emb_dim)
        self.relation_embedding = nn.Embedding(num_relations, emb_dim)
        self.edge_embedding = nn.Embedding(num_edges, emb_dim)

        nn.init.xavier_uniform_(self.entity_embedding, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.edge_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = MGCNConv(emb_dim, emb_dim)
        self.conv2 = MGCNConv(emb_dim, emb_dim)

        self.dropout_ratio = dropout

    def forward(self, edge_index, edge_type, edge_ids):
        """Compute loss of using the embeddings and triplets

        Args:
            edge_index: edges of the sampled graph. shape: [2, E]
            edge_type: relation types of each edge. shape: [E]
            edge_ids: edge indices in triplets file. shape: [E]
        """
        # construct edge_attr using global and local relation embeddings
        edge_attr = self.relation_embedding(edge_type) * self.edge_embedding(edge_ids)
        x = self.entity_embedding
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class MGCNConv(MessagePassing):
    """The relational graph convolutional operator

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(MGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr):
        """Perform message passing operator

        Args:
            x: (tensor) node features. shape: [N, in_channels]
            edge_index: (tensor) edges. shape: [2, E]
            edge_attr: (tensor) local edge embeddings. shape: [E, d]
        """
        # add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_index, size, edge_attr):
        """
        Construct messages to node i in analogy to ϕ for each edge in (j, i).

        Args:
            x_j: embeddings of node j. shape: [E, d]
            edge_index: respective edges. shape: [E]
            edge_attr: edge embeddings. shape: [E, d]
        """
        # transform node features using edge features
        x_j = x_j * edge_attr

        # normalize node fetures
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        """ Return new node embeddings, aggr_out has shape [N, out_channels]"""
        return aggr_out
