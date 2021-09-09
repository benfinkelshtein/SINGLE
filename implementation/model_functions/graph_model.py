from classes.basic_classes import GNN_TYPE
from model_functions.basicTrainer import basicTrainer, test
from model_functions.robust_gcn import train
from model_functions.gal.gal_trainer import galTrainer
from model_functions.lat_gcn.lat_gcn_trainer import latgcnTrainer
from helpers.fileNamer import fileNamer
from adversarial_attack.adversarialTrainer import adversarialTrainer
from helpers.getGitPath import getGitPath
from classes.basic_classes import DatasetType
from dataset_functions.graph_dataset import GraphDataset
from classes.approach_classes import Approach

from typing import Tuple
import os.path as osp
import torch
from torch import nn
from torch.nn import functional as F
import copy
import numpy as np


class Model(torch.nn.Module):
    """
        Generic model class
        each gnn sets a different model

        Parameters
        ----------
        gnn_type: GNN_TYPE
        num_layers: int
        dataset: GraphDataset
        device: torch.cuda
    """
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, dataset: GraphDataset, device: torch.cuda):
        super(Model, self).__init__()
        self.attack = False
        self.layers = nn.ModuleList().to(device)
        data = dataset.data

        if hasattr(dataset, 'glove_matrix'):
            self.glove_matrix = dataset.glove_matrix.to(device)
        else:
            self.glove_matrix = torch.eye(data.x.shape[1]).to(device)

        num_initial_features = dataset.num_features
        num_final_features = dataset.num_classes
        hidden_dims = [32] * (num_layers - 1)
        all_channels = [num_initial_features] + hidden_dims + [num_final_features]

        # gnn layers
        if gnn_type is GNN_TYPE.SGC:
            self.layers.append(gnn_type.get_layer(in_dim=num_initial_features, out_dim=num_final_features,
                                                  K=num_layers).to(device))
        else:
            for in_channel, out_channel in zip(all_channels[:-1], all_channels[1:]):
                self.layers.append(gnn_type.get_layer(in_dim=in_channel, out_dim=out_channel).to(device))

        self.name = gnn_type.string()
        self.num_layers = num_layers
        self.device = device
        self.edge_index = data.edge_index.to(device)
        self.edge_weight = None

    def forward(self, x=None):
        if x is None:
            x = self.getInput().to(self.device)

        x = torch.matmul(x, self.glove_matrix).to(self.device)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight).to(self.device))\
                .to(self.device)
            x = F.dropout(x, training=self.training and not self.attack).to(self.device)

        x = self.layers[-1](x=x, edge_index=self.edge_index, edge_weight=self.edge_weight).to(self.device)
        return F.log_softmax(x, dim=1).to(self.device)

    def getInput(self) -> torch.Tensor:
        """
            a get function for the models input

            Returns
            ----------
            model_input: torch.Tensor
        """
        raise NotImplementedError

    def injectNode(self, dataset: GraphDataset, attacked_node: torch.Tensor) -> torch.Tensor:
        """
            injects a node to the model

            Parameters
            ----------
            dataset: GraphDataset
            attacked_node: torch.Tensor - the victim/attacked node

            Returns
            -------
            malicious_node: torch.Tensor - the injected/attacker/malicious node
            dataset: GraphDataset - the injected dataset
        """
        raise NotImplementedError

    def removeInjectedNode(self, attack):
        """
            removes the injected node from the model

            Parameters
            ----------
            attack: oneGNNAttack
        """
        raise NotImplementedError


class NodeModel(Model):
    """
        model for node attacks
        more information at Model
    """
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, dataset: GraphDataset, device: torch.cuda):
        super(NodeModel, self).__init__(gnn_type, num_layers, dataset, device)
        data = dataset.data
        node_attribute_list = []
        for idx in range(data.x.shape[0]):
            node_attribute_list += [torch.nn.Parameter(data.x[idx].unsqueeze(0), requires_grad=False).to(device)]
        self.node_attribute_list = node_attribute_list

    def getInput(self) -> torch.Tensor:
        return torch.cat(self.node_attribute_list, dim=0)

    def setNodesAttribute(self, idx_node: torch.Tensor, idx_attribute: torch.Tensor, value: float):
        """
            sets a value for a specific node's specific attribute in the node_attribute_list

            Parameters
            ----------
            idx_node: torch.Tensor - the specific node
            idx_attribute: torch.Tensor - the specific attribute
            value: float
        """
        self.node_attribute_list[idx_node][0][idx_attribute] = value

    def setNodesAttributes(self, idx_node: torch.Tensor, values: torch.Tensor):
        """
            sets the attributes for a specific node in the node_attribute_list

            Parameters
            ----------
            idx_node: torch.Tensor - the specific node
            values: torch.Tensor
        """
        self.node_attribute_list[idx_node][0] = values

    def injectNode(self, dataset: GraphDataset, attacked_node: torch.Tensor) -> torch.Tensor:
        """
            information at the generic base class Model
        """
        data = dataset.data

        # creating injection values
        malicious_node = torch.tensor([dataset.data.num_nodes]).to(self.device)
        injected_attributes = torch.zeros(size=(1, data.num_features)).to(self.device)
        if dataset.type is DatasetType.CONTINUOUS:
            injected_attributes = injected_attributes + 0.1
        elif dataset.type is DatasetType.DISCRETE:
            injected_attributes[0][0] = 1
        injected_edge = torch.tensor([[malicious_node.item()], [attacked_node.item()]]).to(self.device)

        # injecting to the model
        self.node_attribute_list += [torch.nn.Parameter(injected_attributes, requires_grad=False).to(self.device)]
        self.edge_index = torch.cat((self.edge_index, injected_edge), dim=1).type(torch.LongTensor).to(self.device)

        # injecting to the data
        false_tensor = torch.tensor([False]).to(self.device)
        data.train_mask = torch.cat((data.train_mask, false_tensor), dim=0)
        data.val_mask = torch.cat((data.val_mask, false_tensor), dim=0)
        data.test_mask = torch.cat((data.test_mask, false_tensor), dim=0)
        data.y = torch.cat((data.y, torch.tensor([0]).to(self.device)), dim=0)
        return malicious_node, dataset

    def removeInjectedNode(self, attack):
        """
            information at the generic base class Model
        """
        dataset = attack.getDataset()
        data = dataset.data

        # removing injection from the model
        self.node_attribute_list.pop()
        self.edge_index = self.edge_index[:, :-1]

        # removing injection from the data
        data.train_mask = data.train_mask[:-1]
        data.val_mask = data.val_mask[:-1]
        data.test_mask = data.test_mask[:-1]
        data.y = data.y[:-1]
        attack.setDataset(dataset=dataset)

    def is_zero_grad(self) -> bool:
        return False


class EdgeModel(Model):
    """
        model for edge attacks
        more information at Model
    """
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, dataset: GraphDataset, device: torch.cuda):
        super(EdgeModel, self).__init__(gnn_type, num_layers, dataset, device)
        data = dataset.data
        self.x = data.x.to(device)
        self.edge_weight = torch.nn.Parameter(torch.ones(data.edge_index.shape[1]), requires_grad=False).to(device)

    def getInput(self) -> torch.Tensor:
        """
            information at the generic base class Model
        """
        return self.x

    @torch.no_grad()
    def expandEdgesByMalicious(self, dataset: GraphDataset, approach: Approach, attacked_node: torch.Tensor,
                               neighbours: torch.Tensor, device: torch.cuda) -> torch.Tensor:
        """
            adds edges with zero weights to the malicious/attacker node according to the attack approach

            Parameters
            ----------
            dataset: GraphDataset
            approach: Approach
            attacked_node: torch.Tensor -  the victim node
            neighbours: torch.Tensor - 2d-tensor that includes
                                       1st-col - the nodes that are in the victim nodes BFS neighborhood
                                       2nd-col - the distance of said nodes from the victim node
            device: torch.cuda

            Returns
            ----------
            malicious_index: torch.Tensor - the injected/attacker/malicious node index
        """
        data = dataset.data
        clique = torch.cat((attacked_node, neighbours))
        n = data.num_nodes
        zero_dim_edge_index = []
        first_dim_edge_index = []
        malicious_index = None

        flag_global_approach = approach.isGlobal()
        if not approach.isGlobal():
            malicious_index = np.random.choice(data.num_nodes, 1).item()

        for neighbour_num, neighbour in enumerate(clique):
            ignore = dataset.reversed_arr_list[neighbour]  # edges which already exist

            tmp_zero_dim_edge_index = []
            if flag_global_approach:
                # adds all edges from the whole graph to the neighbourhood
                tmp_zero_dim_edge_index = [idx for idx in range(n) if idx not in ignore]
            else:
                # adds all edges from malicious index to the neighbourhood
                if malicious_index not in ignore:
                    tmp_zero_dim_edge_index = [malicious_index]

            zero_dim_edge_index += tmp_zero_dim_edge_index
            first_dim_edge_index += [neighbour.item()] * len(tmp_zero_dim_edge_index)

        if zero_dim_edge_index:
            model_edge_index = torch.tensor([zero_dim_edge_index, first_dim_edge_index]).to(device)
            model_edge_weight = torch.zeros(len(zero_dim_edge_index)).to(device)

            self.edge_index = torch.cat((self.edge_index, model_edge_index), dim=1)
            self.edge_weight.data = torch.cat((self.edge_weight.data, model_edge_weight))

        return malicious_index


class ModelWrapper(object):
    """
        a wrapper which includes the model and its generic functions

        Parameters
        ----------
        node_model: bool - whether or not this is a node-based-model
        gnn_type: GNN_TYPE
        num_layers: int
        dataset: GraphDataset
        patience: int
        device: torch.cuda
        seed: int
    """
    def __init__(self, node_model: bool, gnn_type: GNN_TYPE, num_layers: int, dataset: GraphDataset,
                 patience: int, device: torch.cuda, seed: int):
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        if node_model:
            if gnn_type.is_robust_model():
                self.model = gnn_type.get_model(dataset=dataset, device=device, num_layers=num_layers)
            else:
                self.model = NodeModel(gnn_type=gnn_type, num_layers=num_layers, dataset=dataset, device=device)
        else:
            if gnn_type.is_robust_model():
                exit("Robust GNNs are tested against node-based attacks only")
            else:
                self.model = EdgeModel(gnn_type=gnn_type, num_layers=num_layers, dataset=dataset, device=device)
        self.node_model = node_model
        self.patience = patience
        self.device = device
        self.seed = seed
        self._setOptimizer()

        self.basic_log = None
        self.clean = None

    def _setOptimizer(self):
        """
            sets an optimizer for the Model object
        """
        model = self.model
        # Only perform weight-decay on first convolution.
        list_dict_param = [dict(params=model.layers[0].parameters(), weight_decay=5e-4)]
        for layer in model.layers[1:]:
            list_dict_param += [dict(params=layer.parameters(), weight_decay=0)]
        self._setLR()
        self.optimizer = torch.optim.Adam(list_dict_param, lr=self.lr)

    def _setLR(self):
        """
            sets the learn rate
        """
        self.lr = 0.01

    def setModel(self, model: Model):
        """
            sets a specific model

            Parameters
            ----------
            model: Model
        """
        self.model = copy.deepcopy(model)

    def train(self, dataset: GraphDataset, attack=None):
        """
            prepare for train

            Parameters
            ----------
            dataset: GraphDataset
            attack: oneGNNAttack
        """
        model = self.model
        folder_name = osp.join(getGitPath(), 'models')
        if attack is None:
            folder_name = osp.join(folder_name, 'basic_models')
            targeted, continuous_epochs = None, None
        else:
            folder_name = osp.join(folder_name, 'adversarial_models')
            targeted, continuous_epochs = attack.targeted, attack.continuous_epochs

        file_name = fileNamer(node_model=self.node_model, dataset_name=dataset.name, model_name=model.name,
                              num_layers=model.num_layers, patience=self.patience, seed=self.seed, targeted=targeted,
                              continuous_epochs=continuous_epochs, end='.pt')
        model_path = osp.join(folder_name, file_name)

        # load model and optimizer
        if not osp.exists(model_path):
            # train model
            model, model_log, test_acc = self.useTrainer(dataset=dataset, attack=attack)
            torch.save((model.state_dict(), model_log, test_acc), model_path)
        else:
            model_state_dict, model_log, test_acc = torch.load(model_path)
            model.load_state_dict(model_state_dict)
            print(model_log + '\n')
        self.basic_log = model_log
        self.clean = test_acc

    def useTrainer(self, dataset: GraphDataset, attack=None) -> Tuple[Model, str, torch.Tensor]:
        """
            trains the model

            Parameters
            ----------
            dataset: GraphDataset
            attack: oneGNNAttack

            Returns
            -------
            model: Model
            model_log: str
            test_accuracy: torch.Tensor
        """
        data = dataset.data
        if self.gnn_type == GNN_TYPE.ROBUST_GCN:
            if dataset.type is DatasetType.DISCRETE:

                idx_train = data.train_mask.nonzero().T[0]
                idx_train = idx_train.cpu().detach().numpy()

                idx_unlabeled = data.test_mask.nonzero().T[0]
                idx_unlabeled = idx_unlabeled.cpu().detach().numpy()

                train(gcn_model=self.model, X=data.x, y=data.y, idx_train=idx_train, idx_unlabeled=idx_unlabeled, q=3)
                train_accuracy, val_accuracy, test_accuracy = test(model=self.model, data=data)
                model_log = 'Basic Model - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}' \
                    .format(train_accuracy, val_accuracy, test_accuracy)
                return self.model, model_log, test_accuracy
            elif dataset.type is DatasetType.CONTINUOUS:
                exit(" According to the ROBUST GCN paper, this gnn works only for discrete datasets")
        elif self.gnn_type == GNN_TYPE.GAL:  # RGG
            return galTrainer(self.model, data)
        elif self.gnn_type == GNN_TYPE.LAT_GCN:  # RGG
            return latgcnTrainer(self.model, self.optimizer, data, self.patience)

        return basicTrainer(self.model, self.optimizer, data, self.patience)


class AdversarialModelWrapper(ModelWrapper):
    """
        a wrapper which includes an adversarial model
        more information at ModelWrapper
    """
    def __init__(self, node_model, gnn_type, num_layers, dataset, patience, device, seed):
        super(AdversarialModelWrapper, self).__init__(node_model, gnn_type, num_layers, dataset, patience, device, seed)

    # override
    def _setLR(self):
        """
            information at the base class ModelWrapper
        """
        self.lr = 0.005

    def useTrainer(self, dataset: GraphDataset, attack=None) -> Tuple[Model, str, torch.Tensor]:
        """
            information at the base class ModelWrapper
        """
        return adversarialTrainer(attack=attack)
