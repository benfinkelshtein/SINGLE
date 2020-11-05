from node_attack.attackTrainer import attackTrainer
from helpers.algorithms import kBFS
from helpers.algorithms import heuristicApproach
from helpers.algorithms import gradientApproach
from classes.approach_classes import NodeApproach
from classes.basic_classes import Print

import torch


# chooses the node we attack with (the malicious node) from our BFS environment
# this BFS environments is also calculated according to our selected approach
# lastly, we attack using attackTrainer
# important note: the victim node is already known (attacked node)
def attackVictim(attack, approach, print_answer, attacked_node, y_target, node_num):
    device = attack.device
    dataset = attack.dataset

    neighbours_and_dist = kBFS(root=attacked_node, device=device, reversed_arr_list=dataset.reversed_arr_list,
                               K=attack.num_layers)
    if neighbours_and_dist.nelement():
        neighbours_and_dist = manipulateNeighborhood(attack=attack, approach=approach, attacked_node=attacked_node,
                                                     neighbours_and_dist=neighbours_and_dist, device=device)
        attack_log = 'Attack: {:03d}, Node: {}, BFS clique: {}'.format(node_num, attacked_node.item(),
                                                                       neighbours_and_dist.shape[0] + 1)
    else:
        attack_log = 'Attack: {:03d}, Node: {} is a solo node'.format(node_num, attacked_node.item())
    # in adversarial mode add #Epoch
    if attack.mode.isAdversarial():
        attack_log = 'Adv Epoch: {:03d}, '.format(attack.idx) + attack_log

    # special cases of solo node and duo node for double
    BFS_size = neighbours_and_dist.shape[0]
    if not neighbours_and_dist.nelement():
        if print_answer is Print.YES:
            print(attack_log, flush=True)
        return None
    if approach is NodeApproach.TWO_ATTACKERS and BFS_size == 1:
        if print_answer is Print.YES:
            print(attack_log + ': Too small for two attackers', flush=True)
        return None

    if print_answer is Print.YES:
        print(attack_log, flush=True)
    malicious_node = approach.getMaliciousNode(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                               node_num=node_num, neighbours_and_dist=neighbours_and_dist,
                                               BFS_size=BFS_size)
    # calculates the malicious node for the irregular approach of agree
    if approach is NodeApproach.AGREE:
        print()
        malicious_node_heuristic = heuristicApproach(reversed_arr_list=attack.dataset.reversed_arr_list,
                                                     neighbours_and_dist=neighbours_and_dist,
                                                     device=attack.device)
        malicious_node_gradient = gradientApproach(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                                   node_num=node_num, neighbours_and_dist=neighbours_and_dist,
                                                   approach=approach)
        attack_results = torch.zeros(1, 2)
        attack_results[0][0] = malicious_node_heuristic == malicious_node_gradient
        return attack_results
    if malicious_node is None:
        quit("Node approach doesnt exist")

    attack_results = attackTrainer(attack=attack, approach=approach, model=attack.model_wrapper.model,
                                   print_answer=print_answer, attacked_nodes=attacked_node, y_targets=y_target,
                                   malicious_nodes=malicious_node, node_num=node_num,
                                   attack_epochs=attack.attack_epochs, lr=attack.lr)
    return attack_results


# a helper function which manipulates the BFS neighborhood according to our selected approach
def manipulateNeighborhood(attack, approach, attacked_node, neighbours_and_dist, device):
    if attack.mode.isDistance():
        neighbours_and_dist = neighbours_and_dist[neighbours_and_dist[:, 1] == attack.current_distance, :]

    if approach is NodeApproach.INDIRECT:
        return neighbours_and_dist[neighbours_and_dist[:, 1] != 1, :]
    elif approach is NodeApproach.DIRECT:
        return torch.tensor([[attacked_node.item(), 0]]).to(device)
    else:
        return neighbours_and_dist
