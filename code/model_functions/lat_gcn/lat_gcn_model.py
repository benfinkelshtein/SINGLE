from abc import get_cache_token
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Dropout

from functools import reduce


class LATGCNModel(torch.nn.Module):
    def __init__(self, dataset, device, **kwargs):
        super(LATGCNModel, self).__init__(**kwargs)

        self.device = device

        self.internal_channels = 128

        dropout_value = 0.2

        self.input_dropout = Dropout(p=dropout_value)
        self.conv1 = GCNConv(dataset.num_features, self.internal_channels).to(self.device)
        self.h1_dropout = Dropout(p=dropout_value)
        self.conv2 = GCNConv(self.internal_channels, dataset.num_classes).to(self.device)

        self.layers = torch.nn.ModuleList([self.conv1, self.conv2])
        self.num_layers = len(self.layers)


        # Our attack att
        # self.attack = True
        # start of changes XXXXX
        
        data = dataset.data
        self.data = data

        if hasattr(dataset, 'glove_matrix'):
            self.glove_matrix = dataset.glove_matrix.to(device)
        else:
            self.glove_matrix = torch.eye(data.x.shape[1]).to(device)

        self.name = 'LATGCN'
        self.device = device
        self.edge_index = data.edge_index.to(device)

        node_attribute_list = []
        for idx in range(data.x.shape[0]):
            node_attribute_list += [torch.nn.Parameter(data.x[idx].unsqueeze(0), requires_grad=False).to(device)]
        self.node_attribute_list = node_attribute_list

    def get_perturbation_shape(self):
        return (self.getInput().detach().shape[0], self.internal_channels)

    def is_zero_grad(self) -> bool:
        nodes_with_gradient = filter(lambda node: node.grad is not None, self.node_attribute_list)
        abs_gradients = map(lambda node: node.grad.abs().sum().item(), nodes_with_gradient)
        if reduce(lambda x, y: x + y, abs_gradients) == 0:
            return True
        else:
            return False

    def forward(self, input=None, perturbation=None, grad_perturbation=False):
        """
        grad_perturbation: tells whether we are in paper loop (5) or not.
            If true -> we are in loop (5) and we want to propagate gradient only for the perturbation
                not for the network
            If false -> we have already found the best perturbation, treat it as constant, and wish
                to propagate gradient to the network
        """

        if input is None:
            input = self.getInput().to(self.device)
        x = torch.matmul(input, self.glove_matrix).to(self.device)

        x_drop = self.input_dropout(x)

        h1 = F.relu(self.conv1(x_drop, self.edge_index))

        h1_d = self.h1_dropout(h1)

        h2 = self.conv2(h1_d, self.edge_index)

        h2 = F.log_softmax(h2, dim=1).to(self.device)

        R = None
        if perturbation is not None:
            if grad_perturbation:
                R = self.conv2(perturbation, self.edge_index)
            else:
                R = self.conv2(perturbation.detach(), self.edge_index)
            # return h2, torch.square(torch.norm(R, p='fro'))
            return h2, torch.norm(R, p='fro')

        else:
            return h2

    def getInput(self):
        return torch.cat(self.node_attribute_list, dim=0)

    def setNodesAttribute(self, idx_node, idx_attribute, value):
        self.node_attribute_list[idx_node][0][idx_attribute] = value

    def setNodesAttributes(self, idx_node, values):
        self.node_attribute_list[idx_node][0] = values
