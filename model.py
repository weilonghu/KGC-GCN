import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class MGCN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, num_edges, params):
        super(MGCN, self).__init__()
        self.params = params

        self.entity_embedding = nn.Embedding(num_entities, params.embed_dim)
        self.relation_embedding = nn.Embedding(2 * num_relations, params.embed_dim)
        self.edge_embeddings = nn.Embedding(2 * num_edges, params.embed_dim)

        self.conv1 = MGCNConv(params.embed_dim, params.embed_dim, num_relations * 2)
        self.conv2 = ConvE(params, num_entities)

    def forward(self, src, rel, data):
        entity, edge_index, edge_norm = data.entity, data.edge_index, data.edge_norm
        edge_type, edge_ids = data.edge_attr

        # Loop-up entity, relation and edge embeddings for gcn encoder
        entity_embs = self.entity_embedding(entity)
        edge_embs = self.edge_embeddings(edge_ids)
        rels_embs = self.relation_embedding(edge_type)

        # GCN encoder
        all_ent = self.conv1(entity_embs, edge_index, edge_type, edge_norm, edge_embs, rels_embs)
        all_ent = F.dropout(all_ent, p=self.params.gcn_drop, training=self.training)

        # ConvE decoder
        src_emb, rel_emb = all_ent[src], self.relation_embedding(rel)
        score = self.conv2(src_emb, rel_emb, all_ent)

        return score

    def loss(self, pred, label):

        return F.binary_cross_entropy(pred, label)


class MGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, bias=True, root_weight=1, **kwargs):
        super(MGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.root_weight = root_weight

        self.ent_bn = nn.BatchNorm1d(out_channels)

        self.mlp = nn.Sequential(
            nn.Linear(3 * in_channels, out_channels, bias=True),
            nn.ReLU()
        )

        self.loop_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_type, edge_norm, edge_embs, rels_embs, size=None):

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm, edge_embs=edge_embs, rels_embs=rels_embs)

    def message(self, x_i, x_j, edge_index, edge_type, edge_norm, edge_embs, rels_embs):

        cat_feat = torch.cat([x_j, edge_embs, rels_embs], dim=1)
        repre = self.mlp(cat_feat)

        return repre * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):

        all_ent = aggr_out + self.loop_mlp(x)

        all_ent = self.ent_bn(all_ent)

        return all_ent

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
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score
