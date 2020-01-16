import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add

import utils


class MGCN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, num_edges, params):
        super(MGCN, self).__init__()
        self.params = params

        self.entity_embedding = utils.get_param((num_entities, params.gcn_in_dim))
        self.relation_embedding = utils.get_param((2 * num_relations, params.gcn_in_dim))
        self.edge_embeddings = utils.get_param((2 * num_edges, params.gcn_in_dim))

        self.conv1 = MGCNConv(params.gcn_in_dim, params.gcn_out_dim, num_relations * 2)
        self.conv2 = ConvE(params, num_entities)
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, src, rel, data):
        entity, edge_index, edge_norm = data.entity, data.edge_index, data.edge_norm
        edge_type, edge_ids = data.edge_attr

        # Loop-up entity, relation and edge embeddings for gcn encoder
        entity_embs = torch.index_select(self.entity_embedding, 0, entity)
        edge_embs = torch.index_select(self.edge_embeddings, 0, edge_ids)

        # GCN encoder
        all_ent, all_rel = self.conv1(entity_embs, edge_index, edge_type, edge_norm, edge_embs, self.relation_embedding)
        all_ent = F.dropout(all_ent, p=self.params.gcn_drop, training=self.training)

        # ConvE decoder
        src_emb, rel_emb = torch.index_select(all_ent, 0, src), torch.index_select(all_rel, 0, rel)
        score = self.conv2(src_emb, rel_emb, all_ent)

        return score

    def loss(self, pred, label):

        return self.loss_fn(pred, label)


class MGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, bias=False, dropout=0.1, **kwargs):
        super(MGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.ent_bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.act = torch.tanh

        self.loop_weight = utils.get_param((in_channels, out_channels))
        self.in_weight = utils.get_param((in_channels, out_channels))
        self.out_weight = utils.get_param((in_channels, out_channels))
        self.rels_weight = utils.get_param((in_channels, out_channels))
        self.loop_rel = utils.get_param((1, in_channels))
        self.loop_edge = utils.get_param((1, in_channels))

        if bias is True:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.register_parameter('bias', None)

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]

        return norm

    def forward(self, x, edge_index, edge_type, edge_norm, edge_embs, rels_embs, size=None):

        num_edges = edge_type.size(0) // 2
        num_ent = x.size(0)
        rels_embs = torch.cat([rels_embs, self.loop_rel], dim=0)

        in_edge_index, out_edge_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_edge_type, out_edge_type = edge_type[:num_edges], edge_type[num_edges:]
        in_edge_embs, out_edge_embs = edge_embs[:num_edges], edge_embs[num_edges:]
        loop_edge_embs = self.loop_edge.expand(num_ent, -1)

        loop_edge_index = torch.stack([torch.arange(num_ent, device=x.device), torch.arange(num_ent, device=x.device)])
        loop_edge_type = torch.full((num_ent,), rels_embs.size(0) - 1, dtype=torch.long, device=x.device)

        in_edge_norm = self.compute_norm(in_edge_index, num_ent)
        out_edge_norm = self.compute_norm(out_edge_index, num_ent)

        in_res = self.propagate(in_edge_index, size=size, x=x, edge_type=in_edge_type, edge_norm=in_edge_norm, edge_embs=in_edge_embs, rels_embs=rels_embs, mode='in')
        out_res = self.propagate(out_edge_index, size=size, x=x, edge_type=out_edge_type, edge_norm=out_edge_norm, edge_embs=out_edge_embs, rels_embs=rels_embs, mode='out')
        loop_res = self.propagate(loop_edge_index, size=size, x=x, edge_type=loop_edge_type, edge_norm=None, edge_embs=loop_edge_embs, rels_embs=rels_embs, mode='loop')

        out = (self.drop(in_res) + self.drop(out_res) + loop_res) / 3
        if self.bias is not None:
            out = out + self.bias
        all_ent = self.act(self.ent_bn(out))
        all_rel = torch.matmul(rels_embs, self.rels_weight)[:-1]

        return all_ent, all_rel

    def message(self, x_j, edge_index, edge_type, edge_norm, edge_embs, rels_embs, mode=None):

        weight = getattr(self, '{}_weight'.format(mode))
        rel_emb = torch.index_select(rels_embs, 0, edge_type)
        x_j_rel = x_j * rel_emb * edge_embs
        out = torch.matmul(x_j_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):

        return aggr_out

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
        self.bn2 = nn.BatchNorm1d(params.gcn_out_dim)

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
        self.fc = torch.nn.Linear(self.flat_sz, params.gcn_out_dim)

        self.register_parameter('bias', nn.Parameter(torch.zeros(num_entities)))

    def forward(self, src_emb, rel_emb, all_ent):

        src_emb = src_emb.view(-1, 1, self.params.gcn_out_dim)
        rel_emb = rel_emb.view(-1, 1, self.params.gcn_out_dim)
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
