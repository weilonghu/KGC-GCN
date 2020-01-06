import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class MGCN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, params):
        super(MGCN, self).__init__()
        self.params = params

        self.entity_embedding = nn.Embedding(num_entities, params.embed_dim)
        self.relation_embedding = nn.Embedding(2 * num_relations, params.embed_dim)

        self.conv1 = MGCNConv(params.embed_dim, params.embed_dim, num_relations * 2)
        self.conv2 = ConvE(params, num_entities)

    def forward(self, src, rel, data):
        entity, edge_index, edge_type, edge_norm = data.entity, data.edge_index, data.edge_attr, data.edge_norm

        # gcn
        entity_embeddings = self.entity_embedding(entity)
        entity_embeddings = self.conv1(entity_embeddings, edge_index, edge_type, edge_norm)
        entity_embeddings = F.dropout(entity_embeddings, p=self.params.hidden_drop, training=self.training)

        # ConvE
        src_emb, rel_emb, all_ent = entity_embeddings[src], self.relation_embedding(rel), entity_embeddings
        score = self.conv2(src_emb, rel_emb, all_ent)

        return score

    def loss(self, pred, label):

        return F.binary_cross_entropy(pred, label)


class MGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(MGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.weight = nn.Parameter(torch.Tensor(num_relations, out_channels))

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
        nn.init.xavier_uniform_(self.basis, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.att, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.root, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_i, x_j, edge_index_i, edge_index_j, edge_index, edge_type, edge_norm):

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)
        out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out * edge_norm.view(-1, 1)

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


class ConvE(nn.Module):

    def __init__(self, params, num_entities):
        super(ConvE, self).__init__()

        self.params = params

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(params.num_filter)
        self.bn2 = nn.BatchNorm1d(params.embed_dim)

        self.hidden_drop = torch.nn.Dropout(params.hidden_drop)
        self.hidden_drop2 = torch.nn.Dropout(params.hidden_drop2)
        self.feature_drop = torch.nn.Dropout(params.feat_drop)
        self.conv_e = torch.nn.Conv2d(
            in_channels=1,
            out_channels=params.num_filter,
            kernel_size=(params.kernel_size, params.kernel_size),
            stride=1,
            padding=0,
            bias=params.bias
        )

        flat_sz_h = int(2 * params.k_w) - params.kernel_size + 1
        flat_sz_w = params.k_h - params.kernel_size + 1
        self.flat_sz = flat_sz_h * flat_sz_w * params.num_filter
        self.fc = torch.nn.Linear(self.flat_sz, params.embed_dim)

        self.register_parameter('bias', nn.Parameter(torch.zeros(num_entities)))

    def forward(self, src_emb, rel_emb, all_ent):

        src_emb = src_emb.view(-1, 1, self.params.embed_dim)
        rel_emb = rel_emb.view(-1, 1, self.params.embed_dim)
        stack_inp = torch.cat([src_emb, rel_emb], dim=1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(-1, 1, 2 * self.params.k_w, self.params.k_h)

        x = self.bn0(stack_inp)
        x = self.conv_e(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score
