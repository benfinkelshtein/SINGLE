from node_attack.attackTrainer import attackTrainer
from classes.basic_classes import Print

import collections
import numpy as np
import torch
import copy


# a regular BFS algorithm with a k distance stopping rule
# important note: the BFS doesnt include the root in its neighbours
def kBFS(root, device, reversed_arr_list, K):
    neighbours_and_dist = []

    visited, queue = set(), collections.deque()
    queue.append((root.item(), 0))
    visited.add(root.item())

    while queue:
        # Dequeue a vertex from queue
        vertex, dist = queue.popleft()

        # If not visited, mark it as visited, and
        # enqueue it
        if dist < K != 0:
            for neighbour in reversed_arr_list[vertex]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, dist + 1))
                    neighbours_and_dist.append([neighbour, dist + 1])
    neighbours_and_dist = torch.tensor(neighbours_and_dist).to(device)
    return neighbours_and_dist


# our heuristic approach: our of all nodes with distance 1 choose the one with minimal in degree
# in the case of multiple such nodes, select at random
def heuristicApproach(reversed_arr_list, neighbours_and_dist, device):
    # the in degree of all nodes
    nodes_in_degree = []
    for nodes_leading_to_node_index in reversed_arr_list:
        nodes_in_degree.append(len(nodes_leading_to_node_index))
    nodes_in_degree = torch.tensor(nodes_in_degree).to(device)

    # find nodes with distance = 1
    distance_one_nodes = neighbours_and_dist[neighbours_and_dist[:, 1] == 1, 0]

    # in degree of nodes with distance = 1
    in_degree_of_distance_one_nodes = nodes_in_degree[distance_one_nodes]
    nodes_with_min_in_degree_and_distance_one =\
        distance_one_nodes[in_degree_of_distance_one_nodes == torch.min(in_degree_of_distance_one_nodes)]

    if nodes_with_min_in_degree_and_distance_one.shape[0] == 1:
        return nodes_with_min_in_degree_and_distance_one

    # choose one at random
    random_index = np.random.choice(nodes_with_min_in_degree_and_distance_one.shape[0], 1)
    return nodes_with_min_in_degree_and_distance_one[random_index]


# our gradient approach: attack node with the WHOLE BFS neighbourhood for 1 epoch
# choose the node with the maximal gradient
def gradientApproach(attack, attacked_node, y_target, node_num, neighbours_and_dist, approach):
    device = attack.device
    gradient_model = copy.deepcopy(attack.model_wrapper.model)
    x0 = attack.dataset.data.x.clone().detach()
    malicious_nodes = neighbours_and_dist[:, 0]

    # attack node with the whole bfs clique - for 1 epoch!
    attackTrainer(attack=attack, approach=approach, model=gradient_model, print_answer=Print.NO,
                  attacked_nodes=attacked_node, y_targets=y_target, malicious_nodes=malicious_nodes, node_num=node_num,
                  attack_epochs=1, lr=1e-2)
    # choose the largest norm-linf attribute
    nodes_norm_linf, _ = torch.max(torch.abs(gradient_model.getInput() - x0), dim=1)
    nodes_norm_linf = nodes_norm_linf.to(device)
        
    malicious_nodes_max_norm_linf_idx = torch.argmax(nodes_norm_linf[malicious_nodes]).to(device)
    malicious_nodes_max_norm_linf_idx = torch.tensor([malicious_nodes_max_norm_linf_idx.item()]).to(device)

    return malicious_nodes[malicious_nodes_max_norm_linf_idx]
