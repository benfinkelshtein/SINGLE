from model_functions.rgnn.models import RGNN

from functools import reduce
import torch


class RGNNModel(RGNN):
    def __init__(self, dataset, device):
        super(RGNNModel, self).__init__(n_features=dataset.num_features, n_classes=dataset.num_classes)

        # start of changes XXXXX
        self.num_layers = None
        data = dataset.data
        self.attack = False

        if hasattr(dataset, 'glove_matrix'):
            self.glove_matrix = dataset.glove_matrix.to(device)
        else:
            self.glove_matrix = torch.eye(data.x.shape[1]).to(device)

        self.name = 'RGNN'
        self.device = device
        self.edge_index = data.edge_index.to(device)
        self.edge_weight = None
        node_attribute_list = []
        for idx in range(data.x.shape[0]):
            node_attribute_list += [torch.nn.Parameter(data.x[idx].unsqueeze(0), requires_grad=False).to(device)]
        self.node_attribute_list = node_attribute_list
        # end of changes XXXXX

    # start of changes XXXXX
    def getInput(self):
        return torch.cat(self.node_attribute_list, dim=0)

    def setNodesAttribute(self, idx_node, idx_attribute, value):
        self.node_attribute_list[idx_node][0][idx_attribute] = value

    def setNodesAttributes(self, idx_node, values):
        self.node_attribute_list[idx_node][0] = values

    def is_zero_grad(self) -> bool:
        nodes_with_gradient = filter(lambda node: node.grad is not None, self.node_attribute_list)
        abs_gradients = map(lambda node: node.grad.abs().sum().item(), nodes_with_gradient)
        if reduce(lambda x, y: x + y, abs_gradients) == 0:
            return True
        else:
            return False

    def forward(self, input=None):
        if input is None:
            input = self.getInput().to(self.device)
        x = torch.matmul(input, self.glove_matrix).to(self.device)

        edge_idx, edge_weight = self._preprocess_adjacency_matrix(self.edge_index, x)

        # Enforce that the input is contiguous
        x, edge_idx, edge_weight = self._ensure_contiguousness(x, edge_idx, edge_weight)

        for layer in self.layers:
            layer = layer.to(self.device)
            x = layer((x, edge_idx, edge_weight))

        return x
    # end of changes XXXXX
