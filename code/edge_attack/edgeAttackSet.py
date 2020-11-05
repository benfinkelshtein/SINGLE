from edge_attack.edgeAttackVictim import edgeAttackVictim
from node_attack.attackSet import getClassificationTargets
from node_attack.attackSet import checkNodeClassification
from classes.basic_classes import Print

import copy
import numpy as np
import torch


# a wrapper that chooses a victim node and attacks it using edgeAttackVictim
def edgeAttackSet(attack, approach, print_flag):
    device = attack.device
    data = attack.dataset.data

    if print_flag:
        printEdgeAttackHeader(attack=attack, approach=approach)

    num_attacks = torch.sum(data.test_mask).item()
    nodes_to_attack = np.where(np.array(data.test_mask.tolist()))[0]
    attacked_nodes = np.random.choice(nodes_to_attack, num_attacks, replace=False)
    attacked_nodes = torch.from_numpy(attacked_nodes).to(device)

    y_targets = getClassificationTargets(attack=attack, num_attacks=num_attacks, attacked_nodes=attacked_nodes)

    # chooses a victim node and attacks it using oneNodeEdgeAttack
    defence_rate = 0
    attack.model_wrapper.model.attack = True
    model0 = copy.deepcopy(attack.model_wrapper.model)
    for node_num in range(num_attacks):
        attacked_node = torch.tensor([attacked_nodes[node_num]], dtype=torch.long).to(device)
        y_target = torch.tensor([y_targets[node_num]]).to(device)
        attack.setModel(model0)

        classified_to_target = checkNodeClassification(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                                       print_answer=Print.NO, attack_num=node_num + 1)
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
    attack.model_wrapper.model.attack = False
    if print_flag:
        print()
        print("######################## Attack Results ######################## ", flush=True)
        printEdgeAttackHeader(attack=attack, approach=approach)
    printEdgeAttack(attack=attack, defence_rate=defence_rate)
    return torch.tensor([defence_rate])


# a function which prints the header for the final results
def printEdgeAttackHeader(attack, approach):
    # the general print header
    targeted_attack_str = 'Targeted' if attack.targeted else 'Untargeted'
    print("######################## " + targeted_attack_str + " " + approach.string() +
          " Attack ########################", flush=True)


# a function which prints final results
def printEdgeAttack(attack, defence_rate):
    # more specific values
    if attack.model_wrapper.basic_log is not None:
        print(attack.model_wrapper.basic_log + ', ', flush=True, end='')
    print('Defence Success: {:.4f}\n'.format(defence_rate), flush=True)
