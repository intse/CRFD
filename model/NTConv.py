from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import HeteroLinear, Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import BatchNorm

class NTConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 num_node_types: int, num_edge_types: int,
                 edge_type_emb_dim: int,heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 root_weight: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.root_weight = root_weight
        self.edge_type_emb_dim=edge_type_emb_dim

        self.hetero_lin = HeteroLinear(in_channels, out_channels,
                                       num_node_types, bias=bias)

        self.edge_type_rep = Embedding(num_edge_types, edge_type_emb_dim)
        self.bn2 = BatchNorm(edge_type_emb_dim)

        self.edge_type_emb = Linear(num_edge_types, edge_type_emb_dim,bias=True)
        self.bn3 = BatchNorm(edge_type_emb_dim)

        self.att = Linear(2 * out_channels + edge_type_emb_dim,self.heads, bias=False)
        self.bn4 = BatchNorm(self.heads)

        self.lin = Linear(out_channels + edge_type_emb_dim, out_channels,bias=bias)
        self.bn5= BatchNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.hetero_lin.reset_parameters()
        self.edge_type_rep.reset_parameters()
        self.edge_type_emb.reset_parameters()
        self.att.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, node_type: Tensor,edge_type: Tensor) -> Tensor:
        """"""
        x = self.hetero_lin(x, node_type)
        edge_type_rep = F.leaky_relu(self.bn2(self.edge_type_rep(edge_type)),self.negative_slope)

        out = self.propagate(edge_index, x=x, edge_type=edge_type_rep, size=None)

        if self.concat:
            if self.root_weight:
                out += x.view(-1, 1, self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            if self.root_weight:
                out += x

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor, index: Tensor, ptr: OptTensor,size_i: Optional[int]) -> Tensor:
        edge_type = F.leaky_relu(self.bn3(self.edge_type_emb(edge_type)),self.negative_slope)

        alpha = torch.cat([x_i, x_j, edge_type], dim=-1)
        alpha = F.leaky_relu(self.bn4(self.att(alpha)), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.bn5(self.lin(torch.cat([x_j,edge_type], dim=-1))).unsqueeze(-2)
        return out * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
