from torch_geometric.nn import GINConv

from typing import Union, Callable
from torch_geometric.typing import (OptPairTensor, Adj)

import torch
from torch import Tensor


class ModifiedGINConv(GINConv):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        super(ModifiedGINConv, self).__init__(nn=nn, eps=eps, train_eps=train_eps, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight=None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        # This is the modified part:
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x[0].dtype,
                                     device=edge_index.device)
        out = self.propagate(edge_index, x=x, size=None, edge_weight=edge_weight)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
