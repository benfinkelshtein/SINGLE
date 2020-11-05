from helpers.algorithms import kBFS
from node_attack.attackTrainer import train
from node_attack.attackTrainer import test
from classes.approach_classes import EdgeApproach

import numpy as np
import torch


# chooses the edge we flip from our pool of possible edges, according to our approach
def edgeAttackVictim(attack, approach, print_flag, attacked_node, y_target, node_num):
    device = attack.device
    dataset = attack.dataset
    data = dataset.data
    model = attack.model_wrapper.model
    targeted = attack.targeted

    neighbours_and_dist = kBFS(root=attacked_node, device=device, reversed_arr_list=dataset.reversed_arr_list,
                               K=model.num_layers - 1)
    if not neighbours_and_dist.nelement():
        if print_flag:
            print('Attack: {:03d}, Node: {} is a solo node'.format(node_num, attacked_node.item()), flush=True)
        return None

    malicious_indices = neighbours_and_dist[:, 0]
    if print_flag:
        print('Attack: {:03d}, Node: {}'.format(node_num, attacked_node.item()), flush=True, end='')

    attack_results = None
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

    if approach is EdgeApproach.GRAD:
        # select a random node on the graph and - malicious index
        # Add all possible edges between the malicious index and the BFS of distance K-1
        # calculate the edge with the largest gradient and flip it, using edgeTrainer
        malicious_index = np.random.choice(data.num_nodes, 1).item()
        expansion_mode = {"full": False, "malicious_index": malicious_index}
        model.expandEdges(dataset=dataset, attacked_node=attacked_node, neighbours=malicious_indices,
                          device=device, expansion_mode=expansion_mode)
        model = edgeTrainer(targeted=targeted, model=model, attacked_node=attacked_node, y_target=y_target,
                            device=device)
        attack_results = test(data=data, model=model, targeted=targeted, attacked_nodes=attacked_node,
                              y_targets=y_target)

    if approach is EdgeApproach.GLOBAL_GRAD:
        # Add all possible edges between all possible nodes and the BFS of distance K-1
        # calculate the edge with the largest gradient and flip it, using edgeTrainer
        expansion_mode = {"full": True}
        model.expandEdges(dataset=dataset, attacked_node=attacked_node, neighbours=malicious_indices,
                          device=device, expansion_mode=expansion_mode)
        model = edgeTrainer(targeted=targeted, model=model, attacked_node=attacked_node, y_target=y_target,
                            device=device)
        attack_results = test(data=data, model=model, targeted=targeted, attacked_nodes=attacked_node,
                              y_targets=y_target)
    if attack_results is None:
        print("Node approach doesnt exist")
        quit()

    if print_flag:
        print(', Defense Success: {}'.format(not attack_results[3]))
    return attack_results[3]


# flips the edge between attacked node and malicious index
def flipEdge(model, attacked_node, malicious_index, device):
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
    model.edge_weight.data = torch.cat((model.edge_weight.data, torch.tensor([1]).to(device)))


# a forward pass function which chooses the edge with the largest gradient in edge_weight and flips it
def edgeTrainer(targeted, model, attacked_node, y_target, device):
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

        malicious_edge = torch.argmax(torch.abs(diff)).to(device)

        # return to old self and flip edge
        model.edge_weight.data = edge_weight0
        model.edge_weight.data[malicious_edge] = not model.edge_weight.data[malicious_edge]
    return model


# a helper which turns off the grad for the net layers and turns on the grad for the edge weight
def setRequiresGrad(model):
    # specifying layer parameters
    for layer in model.layers:
        for p in layer.parameters():
            p.detach()
            p.requires_grad = False

    model.edge_weight.requires_grad = True
    return [dict(params=model.edge_weight)]
