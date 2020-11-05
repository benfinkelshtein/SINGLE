from node_attack.attackVictim import attackVictim
from node_attack.attackTrainer import test
from classes.basic_classes import Print

import copy
import numpy as np
import torch


# a wrapper that chooses a victim node and attacks it using attackVictim
# an attack can have 3 modes described in parameter attack_params
# regular - for regular attacks
# adversarial - for adversarial training
# distance - for attack as a function of distance from the victim node
# important note: these modes only matter for the print output of the functions not for its functionality
def attackSet(attack, approach, print_answer, trainset):
    device = attack.device
    dataset = attack.dataset
    data = dataset.data

    if print_answer is not Print.NO:
        printAttackHeader(attack=attack, approach=approach)
    num_attacks, nodes_to_attack = getNodesToAttack(data=data, trainset=trainset)

    attacked_nodes = np.random.choice(nodes_to_attack, num_attacks, replace=False)
    attacked_nodes = torch.from_numpy(attacked_nodes).to(device)
    y_targets = getClassificationTargets(attack=attack, num_attacks=num_attacks, attacked_nodes=attacked_nodes)

    # chooses a victim node and attacks it using oneNodeAttack
    attack_results_for_all_attacked_nodes = []
    attack.model_wrapper.model.attack = True
    model0 = copy.deepcopy(attack.model_wrapper.model)
    for node_num in range(num_attacks):
        attacked_node = torch.tensor([attacked_nodes[node_num]], dtype=torch.long).to(device)
        y_target = torch.tensor([y_targets[node_num]], dtype=torch.long).to(device)
        # check if the model is changed in between one node attacks
        if not attack.mode.isAdversarial():
            attack.setModel(model0)

        classified_to_target = checkNodeClassification(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                                       print_answer=print_answer, attack_num=node_num + 1)
        # important note: the victim is attacked only if it is classified to y_target!
        if classified_to_target:
            attack_results = attackVictim(attack=attack, approach=approach, print_answer=print_answer,
                                          attacked_node=attacked_node, y_target=y_target, node_num=node_num + 1)
            # in case of an impossible attack (i.e. double attack with bfs of 1)
            if attack_results is None:
                attack_results = torch.tensor([[0, 0]])

        # in case of a miss-classification
        else:
            attack_results = torch.tensor([[1, 0]])

        attack_results_for_all_attacked_nodes.append(attack_results)

    # print results and save accuracies
    attack_results_for_all_attacked_nodes = torch.cat(attack_results_for_all_attacked_nodes)
    mean_defence_results = getDefenceResultsMean(attack_results=attack_results_for_all_attacked_nodes,
                                                 num_attributes=data.x.shape[1])
    attack.model_wrapper.model.attack = False

    if print_answer is Print.YES:
        print("######################## Attack Results ######################## ", flush=True)
        printAttackHeader(attack=attack, approach=approach)
    if not trainset:
        printAttack(dataset=dataset, basic_log=attack.model_wrapper.basic_log,
                    mean_defence_results=mean_defence_results, num_attributes=data.x.shape[1])
    return mean_defence_results, attacked_nodes, y_targets


# a function which prints the header for the final results
def printAttackHeader(attack, approach):
    distance_log = ''
    if attack.mode.isDistance():
        distance_log += 'Distance: {:02d} '.format(attack.current_distance)

    # the general print header
    targeted_attack_str = 'Targeted' if attack.targeted else 'Untargeted'
    print("######################## " + distance_log + targeted_attack_str + " " + approach.string() + " " +
          attack.model_wrapper.model.name + " Attack ########################", flush=True)
    info = "######################## Max Attack Epochs:" + str(attack.attack_epochs)
    if attack.l_inf is not None:
        info += " Linf:{:.2f}".format(attack.l_inf)
    print(info + " lr:" + str(attack.lr) + " ########################", flush=True)


# a helper that checks that we do not exceed  the num of nodes available in our train/test sets
def getNodesToAttack(data, trainset):
    if trainset:
        num_attacks = torch.sum(data.train_mask).item()
        nodes_to_attack = np.where(np.array(data.train_mask.tolist()))[0]
    else:
        num_attacks = torch.sum(data.test_mask).item()
        nodes_to_attack = np.where(np.array(data.test_mask.tolist()))[0]
    return num_attacks, nodes_to_attack


# a helper that returns the target of the attack task
# if the attack is targeted it will return the target classification
# if the attack is untargeted it will return the correct classification of the node
def getClassificationTargets(attack, num_attacks, attacked_nodes):
    dataset = attack.dataset
    data = dataset.data
    device = attack.device

    if attack.targeted:
        y_targets = np.random.random_integers(0, dataset.num_classes - 2, size=num_attacks)
        y_targets = torch.from_numpy(y_targets).to(device)
        for idx, _ in enumerate(y_targets):
            if y_targets[idx] == data.y[attacked_nodes[idx]]:
                y_targets[idx] = dataset.num_classes - 1
    else:
        y_targets = data.y[attacked_nodes].to(device)
    return y_targets.type(torch.LongTensor).to(device)


# a helper function that checks if the node is currecly classified to y_target
@torch.no_grad()
def checkNodeClassification(attack, attacked_node, y_target, print_answer, attack_num):
    results = test(attack.dataset.data, attack.model_wrapper.model, attack.targeted, attacked_node, y_target)
    classified_to_target = not results[3]

    if not classified_to_target and print_answer is Print.YES:
        attack_log = 'Attack: {:03d}, Node: {}, Misclassified already!\n' \
            .format(attack_num, attacked_node.item())
        if attack.mode.isAdversarial():
            attack_log = 'Adv Epoch: {:03d}, '.format(attack.idx) + attack_log
        print(attack_log, flush=True)
    return classified_to_target


def getDefenceResultsMean(attack_results, num_attributes):
    attack_results = attack_results.type(torch.FloatTensor)
    mean_defence_results = attack_results.mean(dim=0)
    mask1 = (attack_results[:, 1] != 0)
    mask2 = (attack_results[:, 1] != num_attributes)
    mask = torch.logical_and(mask1, mask2)
    if torch.sum(mask) > 0:
        mean_defence_results[1] = torch.mean(attack_results[mask, 1], dim=0)
    else:
        mean_defence_results[1] = num_attributes

    mean_defence_results[0] = 1 - mean_defence_results[0]
    return mean_defence_results


# a function which prints final results
def printAttack(dataset, basic_log, mean_defence_results, num_attributes):
    attack_log = ''
    if basic_log is not None:
        attack_log += basic_log + ', '
    attack_log += 'Test Defence Success: {:.4f}\n'
    attack_log = attack_log.format(mean_defence_results[0].item())
    if not dataset.skip_attributes:
        if mean_defence_results[1] != num_attributes:
            num_of_attack_attributes = mean_defence_results[1].item()
            mus = tuple([num_of_attack_attributes] + [num_of_attack_attributes / num_attributes])
            attack_log += '#Success attack attributes: {:.1f}, #Success attack attributes%: {:.3f}\n'.format(*mus)
        else:
            attack_log += 'All attacks fail, no #attributes'

    print(attack_log, flush=True)
