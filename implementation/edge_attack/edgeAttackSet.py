from edge_attack.edgeAttackVictim import edgeAttackVictim
from node_attack.attackSet import getClassificationTargets
from node_attack.attackVictim import checkNodeClassification
from classes.basic_classes import Print
from classes.approach_classes import Approach, EdgeApproach

import copy
import numpy as np
import torch


def edgeAttackSet(attack, approach: Approach, print_flag: bool) -> torch.Tensor:
    """
        a wrapper that chooses a victim node and attacks it using attackVictim

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
        print_flag: bool - whether to print every iteration or not

        Returns
        -------
        defence_rate: torch.Tensor
    """
    device = attack.device
    dataset = attack.getDataset()
    data = dataset.data

    if print_flag:
        printEdgeAttackHeader(attack=attack, approach=approach)

    num_attacks = torch.sum(data.test_mask).item()
    nodes_to_attack = np.where(np.array(data.test_mask.tolist()))[0]
    attacked_nodes = np.random.choice(nodes_to_attack, num_attacks, replace=False)
    attacked_nodes = torch.from_numpy(attacked_nodes).to(device)

    y_targets = getClassificationTargets(attack=attack, dataset=dataset, num_attacks=num_attacks,
                                         attacked_nodes=attacked_nodes)

    # chooses a victim node and attacks it using oneNodeEdgeAttack
    defence_rate = 0
    attack.model_wrapper.model.attack = True
    model0 = copy.deepcopy(attack.model_wrapper.model)
    for node_num in range(num_attacks):
        attacked_node = torch.tensor([attacked_nodes[node_num]], dtype=torch.long).to(device)
        y_target = torch.tensor([y_targets[node_num]]).to(device)
        classified_to_target = checkNodeClassification(attack=attack, dataset=dataset, attacked_node=attacked_node,
                                                       y_target=y_target, print_answer=Print.NO,
                                                       attack_num=node_num + 1)
        # important note: the victim is attacked only if it is classified to y_target!
        if classified_to_target:
            fail = edgeAttackVictim(attack=attack, approach=approach, print_flag=print_flag,
                                    attacked_node=attacked_node, y_target=y_target, node_num=node_num + 1)
            # the defence rate is raised only if we classify correctly both before and after the attack
            if (not fail) and (fail is not None):
                defence_rate += 1 / num_attacks
        else:
            if print_flag:
                print('Attack: {:03d}, Node: {}, Misclassified already!'.format(node_num + 1, attacked_node.item()))
                if approach is EdgeApproach.MULTI or approach is EdgeApproach.MULTI_GRAD_CHOICE:
                    print()
        attack.setModel(model0)
    attack.model_wrapper.model.attack = False
    if print_flag:
        print()
        print("######################## Attack Results ######################## ", flush=True)
        printEdgeAttackHeader(attack=attack, approach=approach)
    printEdgeAttack(attack=attack, defence_rate=defence_rate)
    return torch.tensor([defence_rate])


def printEdgeAttackHeader(attack, approach: Approach):
    """
        print the header of the attack

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
    """
    # the general print header
    targeted_attack_str = 'Targeted' if attack.targeted else 'Untargeted'
    print("######################## " + targeted_attack_str + " " + approach.string() +
          " Attack ########################", flush=True)


# a function which prints final results
def printEdgeAttack(attack, defence_rate: torch.Tensor):
    """
        print the final results for the attack

        Parameters
        ----------
        attack: oneGNNAttack
        defence_rate: torch.Tensor
    """
    # more specific values
    if attack.model_wrapper.basic_log is not None:
        print(attack.model_wrapper.basic_log + ', ', flush=True, end='')
    print('Defence Success: {:.4f}\n'.format(defence_rate), flush=True)
