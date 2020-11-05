from torch_geometric.nn import SAGEConv

from typing import Union
from torch_geometric.typing import (OptPairTensor, Adj, Size)

import torch
from torch import Tensor
import torch.nn.functional as F


class ModifiedSAGEConv(SAGEConv):
    def __init__(self, **kwargs):
        super(ModifiedSAGEConv, self).__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, edge_weight=None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        # This is the modified part:
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x[0].dtype,
                                         device=edge_index.device)

        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
