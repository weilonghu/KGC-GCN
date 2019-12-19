import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, SAGEConv
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

        nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        self.conv1 = MGCNConv(emb_dim, emb_dim, num_relations)

        self.dropout_ratio = dropout

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
        # x = F.relu(self.conv1(x, data.edge_index, edge_attr, edge_types))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        return x, data.n_id, data.e_id, data.edge_index

    def score_func(self, embedding, triplets):
        """Use distmult to score triplets"""
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding(triplets[:, 1])
        o = embedding[triplets[:, 2]]
        score = torch.sum(torch.abs(s + r - o), dim=1)  # bugs

        return score

    def loss_func(self, embedding, triplets, target):
        """Compute loss for the model"""
        score = self.score_func(embedding, triplets)

        # compute loss
        # cls_loss = F.binary_cross_entropy_with_logits(score, target)
        # reg_loss = torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.weight.data.pow(2))
        pos_scores, neg_scores = score.view(2, -1)
        loss = torch.mean(F.relu(pos_scores - neg_scores + 5))

        # compute accuracy
        logits = torch.sigmoid(score)
        pred = logits >= 0.5
        true = target >= 0.5
        acc = torch.mean(pred.eq(true).float())

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
        nn.init.xavier_normal_(self.relation_matrix.data)

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

    def message(self, x_i, x_j, edge_index_i, edge_index_j, size, edge_attr, edge_type):
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

        return x_j.squeeze_(dim=1)

    def update(self, aggr_out):
        """ Return new node embeddings, aggr_out has shape [N, out_channels]"""
        return aggr_out
