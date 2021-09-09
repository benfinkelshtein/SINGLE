from helpers.algorithms import kBFS
from node_attack.attackTrainerHelpers import train
from node_attack.attackTrainerHelpers import test
from classes.approach_classes import Approach, EdgeApproach

import numpy as np
import torch


def edgeAttackVictim(attack, approach: Approach, print_flag: bool, attacked_node: torch.Tensor, y_target: torch.Tensor,
                     node_num: int) -> torch.Tensor:
    """
        chooses the edge we attack with from our pool of possible edges.
        the pool of possible edges changes per approach
        this BFS environments is also calculated according to our selected approach
        lastly, we attack using attackTrainer
        important note: the victim node is already known (attacked node)

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
        print_flag: bool - whether to print every iteration or not
        attacked_node: torch.Tensor - the victim node
        y_target: torch.Tensor - the target label of the attack
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)

        Returns
        -------
        attack_result: torch.Tensor
    """
    device = attack.device
    dataset = attack.getDataset()
    data = dataset.data
    model = attack.model_wrapper.model
    targeted = attack.targeted
    end_log_template = ', Attack Success: {}'

    neighbours_and_dist = kBFS(root=attacked_node, device=device, reversed_arr_list=dataset.reversed_arr_list,
                               K=model.num_layers - 1)
    if not neighbours_and_dist.nelement():
        if print_flag:
            print('Attack: {:03d}, Node: {} is a solo node'.format(node_num, attacked_node.item()), flush=True)
        return None
    malicious_indices = neighbours_and_dist[:, 0]
    if print_flag:
        print('Attack: {:03d}, Node: {}'.format(node_num, attacked_node.item()), flush=True, end='')

    # according to our approach choose the edge we wish to flip
    if approach is EdgeApproach.RANDOM:
        # select a random node on the graph and - malicious index
        # select a random node from our BFS of distance K-1 -  attacked node
        # use flipEdge
        malicious_index = np.random.choice(data.num_nodes, 1).item()
        new_attacked_node_index = np.random.choice(malicious_indices.shape[0] + 1, 1).item()
        if new_attacked_node_index == malicious_indices.shape[0]:
            new_attacked_node = attacked_node
        else:
            new_attacked_node = torch.tensor([malicious_indices[new_attacked_node_index].item()]).to(device)
        flipEdge(model=model, attacked_node=new_attacked_node, malicious_index=malicious_index, device=device)
        attack_results = test(data=data, model=model, targeted=targeted, attacked_nodes=new_attacked_node,
                              y_targets=y_target)

        if print_flag:
            print(end_log_template.format(attack_results[3]), flush=True)
    else:
        # EdgeApproach.SINGLE
        # select a random node on the graph - malicious index
        # Add all possible edges between the malicious index and the BFS of distance K-1
        # calculate the edge with the largest gradient and flip it, using edgeTrainer
        #
        # EdgeApproach.GRAD_CHOICE
        # Add all possible edges between all possible nodes and the BFS of distance K-1
        # calculate the edge with the largest gradient and flip it, using edgeTrainer
        malicious_index = model.expandEdgesByMalicious(dataset=dataset, approach=approach, attacked_node=attacked_node,
                                                       neighbours=malicious_indices, device=device)
        attack_results = edgeTrainer(data=data, approach=approach, targeted=targeted, model=model,
                                     attacked_node=attacked_node, y_target=y_target, node_num=node_num,
                                     malicious_index=malicious_index, device=device, print_flag=print_flag,
                                     end_log_template=end_log_template)
    if attack_results is None:
        print("Node approach doesnt exist", flush=True)
        quit()

    return attack_results[3]


def flipEdge(model, attacked_node: torch.Tensor, malicious_index: torch.Tensor, device: torch.cuda):
    """
        flips the edge between attacked node and malicious index

        Parameters
        ----------
        model: oneGNNAttack
        attacked_node: torch.Tensor - the victim node
        malicious_index: torch.Tensor - the attacker/malicious index
        device: torch.cuda

        Returns
        -------
        attack_result: torch.Tensor
    """
    attacked_node = attacked_node.item()
    if malicious_index == attacked_node:
        return

    # if edge existed
    for edge_num, edge in enumerate(model.edge_index.T):
        if edge[0] == malicious_index and edge[1] == attacked_node:
            model.edge_weight.data[edge_num] = 0
            return

    # if edge didn't existed
    model.edge_index = torch.cat((model.edge_index, torch.tensor([[malicious_index, attacked_node]]).to(device).T),
                                 dim=1)
    model.edge_weight.data = torch.cat((model.edge_weight.data, torch.tensor([1]).type(torch.FloatTensor).to(device)))


def edgeTrainer(data, approach: Approach, targeted: bool, model,
                attacked_node: torch.Tensor, y_target: torch.Tensor, malicious_index: torch.Tensor, node_num: int,
                device, print_flag, end_log_template):
    """
        a forward pass function which chooses the edge with the largest gradient in edge_weight and flips it
        for multi approaches this process is repeated for each edge with a non-zero gradient

        Parameters
        ----------
        data: torch_geometric.data.Data.data
        approach: Approach
        targeted: bool
        model: Model
        attacked_node: torch.Tensor - the victim node
        y_target: torch.Tensor - the target label of the attack
        malicious_index: torch.Tensor - the attacker/malicious index
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)
        device: torch.cuda
        print_flag: bool - whether to print every iteration or not
        end_log_template: str - suffix of the log format

        Returns
        -------
        attack_result: torch.Tensor
    """
    log_template = '\nAttack: {:03d}, #Edge: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

    edge_weight0 = model.edge_weight.clone().detach()
    optimizer_params = setRequiresGrad(model)
    optimizer = torch.optim.SGD(optimizer_params, lr=0.01)

    train(model=model, targeted=targeted, attacked_nodes=attacked_node, y_targets=y_target, optimizer=optimizer)

    with torch.no_grad():
        diff = model.edge_weight - edge_weight0
        mask1 = torch.logical_and(edge_weight0 == 1, diff > 0).to(device)
        mask2 = torch.logical_and(edge_weight0 == 0, diff < 0).to(device)
        mask = torch.logical_or(mask1, mask2).to(device)
        diff[mask] = 0
        abs_diff = torch.abs(diff)

        # when approach is grad you have the attacker chosen
        if not approach.isGlobal():
            malicious_mask = model.edge_index[0] != torch.tensor(malicious_index).to(device)
            abs_diff[malicious_mask] = 0

        # use of the best edge
        max_malicious_edge = torch.argmax(abs_diff).to(device)

        # when approach is globalGrad you can choose the attacker
        if approach.isGlobal():
            malicious_index = model.edge_index[0][max_malicious_edge]
            malicious_node_mask = model.edge_index[0] != malicious_index
            abs_diff[malicious_node_mask] = 0

        # return edge weights to back to original values and flip
        model.edge_weight.data = edge_weight0
        model.edge_weight.data[max_malicious_edge] = not model.edge_weight.data[max_malicious_edge]
        attack_results = test(data=data, model=model, targeted=targeted, attacked_nodes=attacked_node,
                              y_targets=y_target)
        if not approach.isMulti():
            if print_flag:
                print(end_log_template.format(attack_results[-1]), flush=True)
        else:
            malicious_node_abs_diff = abs_diff[abs_diff != 0]
            # sort edges by absolute diff
            _, sorted_malicious_edge = torch.sort(malicious_node_abs_diff, descending=True)
            if print_flag:
                print(', #Edges: {}'.format(sorted_malicious_edge.shape[0]), flush=True, end='')
                print(log_template.format(node_num, 1, *attack_results[:-1]), flush=True, end='')

            if not attack_results[3] and sorted_malicious_edge.shape[0] > 1:
                attack_results = \
                    findMinimalEdges(sorted_edges=sorted_malicious_edge[1:], data=data, model=model,
                                     targeted=targeted, attacked_node=attacked_node, y_target=y_target,
                                     node_num=node_num, print_flag=print_flag, log_template=log_template,
                                     end_log_template=end_log_template)
            elif print_flag:
                print(end_log_template.format(attack_results[-1]) + '\n', flush=True)
    return attack_results


def setRequiresGrad(model):
    """
        a helper which turns off the grad for the net layers and
        turns on the grad for the edge weights

        Parameters
        -------
        model: Model

        Returns
        -------
        optimization_params: List[Dict]
    """
    # specifying layer parameters
    for layer in model.layers:
        for p in layer.parameters():
            p.detach()
            p.requires_grad = False

    model.edge_weight.requires_grad = True
    return [dict(params=model.edge_weight)]


@torch.no_grad()
def findMinimalEdges(sorted_edges: torch.Tensor, data, model, targeted: bool,
                     attacked_node: torch.Tensor, y_target: torch.Tensor, node_num: int, print_flag: bool,
                     log_template, end_log_template):
    """
        flips each edge with a non-zero gradient
        this function is only available for multi approaches

        Parameters
        ----------
        sorted_edges: torch_geometric.data.Data.data - non-zero gradient edges, sorted by decreasing gradient
        data: torch_geometric.data.Data.data
        targeted: bool
        model: Model
        targeted: bool
        attacked_node: torch.Tensor - the victim node
        y_target: torch.Tensor - the target label of the attack
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)
        print_flag: bool - whether to print every iteration or not
        log_template: str - prefix of the log format
        end_log_template: str - suffix of the log format

        Returns
        -------
        attack_result: torch.Tensor
    """
    for edge_num, malicious_edge in enumerate(sorted_edges):
        model.edge_weight.data[malicious_edge] = not model.edge_weight.data[malicious_edge]
        attack_results = test(data=data, model=model, targeted=targeted, attacked_nodes=attacked_node,
                              y_targets=y_target)
        if print_flag:
            print(log_template.format(node_num, edge_num + 2, *attack_results[:-1]), flush=True, end='')
        if attack_results[3]:
            break
    if print_flag:
        print(end_log_template.format(attack_results[-1]) + '\n', flush=True)
    return attack_results
