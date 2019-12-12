import torch
import torch.nn.functional as F
from layers import LGCN


# create the learning mechanism that is to operate on the edge populations / multi-edges
class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def mm_CONV(conv_channels=20, out_channels=4):
    return torch.nn.Sequential(
        torch.nn.Conv1d(2, conv_channels, kernel_size=3, stride=1, padding=1),
        torch.nn.AdaptiveMaxPool1d(1),
        torch.nn.ReLU(),
        Reshape(-1, conv_channels),
        torch.nn.Linear(conv_channels, 2 * out_channels),
        torch.nn.Dropout(p=0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * out_channels, out_channels),
        torch.nn.ReLU()
    )


"""Create Example Nets
Parameter L determines size of latent representations of the edge populations.
Parameter H1 determines representation size in the intermediate node embedding layer.
H2 determines the size of the final output layer and should agree with the downstream task configured in the data set.

In the GCN layers, the following controls are available:

    make_bidirectional offers bidirectional propagation over directed graphs
    neighbor_nl offers additional per-neighbor nonlinearity inside the graph convolution (L-GCN+)
    DVE provides the option of embedding local neighborhood aggregations of the latent representations (mean-pool) directly on the nodes,
        before proceeding with the GCN (L-GCN+ & DVE)
In these examples, edge populations are pre-padded with zeros and sorted by original sequence length,
accompanied by batch cut-offs for faster processing. The edge_attr_cutoffs parameter may be omitted to proceed without batching.
"""


class LGCN_Net(torch.nn.Module):
    """L4-GCN(bidirectional propagation)"""

    def __init__(self, num_features, L=4, H1=20, H2=2):
        super().__init__()
        self.conv1 = LGCN(num_features, H1, mm_CONV(out_channels=L), L=L,
                          make_bidirectional=True)
        self.conv2 = LGCN(H1, H2, mm_CONV(out_channels=L), L=L,
                          make_bidirectional=True)

    def forward(self, data):
        x, edge_index, edge_attr, edge_attr_cutoffs = data.x, data.edge_index, data.edge_attr, data.edge_attr_cutoffs
        x = self.conv1(x, edge_index, edge_attr,
                       edge_attr_cutoffs=edge_attr_cutoffs)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr,
                       edge_attr_cutoffs=edge_attr_cutoffs)

        return F.log_softmax(x, dim=1)


class LGCN_Net2(torch.nn.Module):
    """L4-GCN+(bidirectional propagation)"""

    def __init__(self, num_features, L=4, H1=20, H2=2):
        super().__init__()
        self.conv1 = LGCN(num_features, H1, mm_CONV(out_channels=L), L=L,
                          make_bidirectional=True,
                          neighbor_nl=True)
        self.conv2 = LGCN(H1, H2, mm_CONV(out_channels=L), L=L,
                          make_bidirectional=True,
                          neighbor_nl=True)

    def forward(self, data):
        x, edge_index, edge_attr, edge_attr_cutoffs = data.x, data.edge_index, data.edge_attr, data.edge_attr_cutoffs
        x = self.conv1(x, edge_index, edge_attr,
                       edge_attr_cutoffs=edge_attr_cutoffs)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr,
                       edge_attr_cutoffs=edge_attr_cutoffs)

        return F.log_softmax(x, dim=1)


class LGCN_Net3(torch.nn.Module):
    """L4-GCN+ & DVE(bidirectional propagation)"""

    def __init__(self, num_features, L=4, H1=20, H2=2):
        super().__init__()
        self.conv1 = LGCN(num_features, H1, mm_CONV(out_channels=L), L=L,
                          make_bidirectional=True,
                          neighbor_nl=True,
                          DVE=True)
        self.conv2 = LGCN(H1, H2, mm_CONV(out_channels=L), L=L,
                          make_bidirectional=True,
                          neighbor_nl=True)

    def forward(self, data):
        x, edge_index, edge_attr, edge_attr_cutoffs = data.x, data.edge_index, data.edge_attr, data.edge_attr_cutoffs
        x = self.conv1(x, edge_index, edge_attr,
                       edge_attr_cutoffs=edge_attr_cutoffs)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr,
                       edge_attr_cutoffs=edge_attr_cutoffs)

        return F.log_softmax(x, dim=1)
