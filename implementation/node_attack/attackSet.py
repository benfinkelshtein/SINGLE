from node_attack.attackVictim import attackVictim
from classes.basic_classes import Print, DatasetType
from classes.approach_classes import Approach, NodeApproach
from node_attack.attackVictim import checkNodeClassification

import copy
import numpy as np
import torch
from typing import Tuple
import torch_geometric


def attackSet(attack, approach: Approach, trainset: bool) -> Tuple[torch.Tensor]:
    """
        a wrapper that chooses a victim node and attacks it using attackVictim

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
        trainset: bool - whether or not this function is used for training or testing

        Returns
        -------
        attack_results_for_all_attacked_nodes: torch.Tensor - 2d-tensor that includes the following for each node
                                                              1st-col - the attack
                                                              2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
        attacked_nodes: torch.Tensor - the victim nodes
        y_targets: torch.Tensor - the target labels of the attack
    """
    device = attack.device
    dataset = attack.getDataset()
    data = dataset.data
    print_answer = attack.print_answer

    if print_answer is not Print.NO:
        printAttackHeader(attack=attack, approach=approach)
    num_attacks, nodes_to_attack = getNodesToAttack(data=data, trainset=trainset)

    attacked_nodes = np.random.choice(nodes_to_attack, num_attacks, replace=False)
    attacked_nodes = torch.from_numpy(attacked_nodes).to(device)
    y_targets = getClassificationTargets(attack=attack, dataset=dataset, num_attacks=num_attacks,
                                         attacked_nodes=attacked_nodes)

    # chooses a victim node and attacks it using oneNodeAttack
    attack_results_for_all_attacked_nodes = []

    attack.model_wrapper.model.attack = True
    model0 = copy.deepcopy(attack.model_wrapper.model)
    for node_num in range(num_attacks):
        attacked_node = torch.tensor([attacked_nodes[node_num]], dtype=torch.long).to(device)
        y_target = torch.tensor([y_targets[node_num]], dtype=torch.long).to(device)
        classified_to_target = checkNodeClassification(attack=attack, dataset=dataset, attacked_node=attacked_node,
                                                       y_target=y_target, print_answer=print_answer,
                                                       attack_num=node_num + 1)
        # important note: the victim is attacked only if it is classified to y_target!
        if classified_to_target:
            attack_results = attackVictim(attack=attack, approach=approach, attacked_node=attacked_node,
                                          y_target=y_target, node_num=node_num + 1)
            # in case of an impossible attack (i.e. double attack with bfs of 1)
            if attack_results is None:
                attack_results = torch.tensor([[0, 0]])

        # in case of a miss-classification
        else:
            attack_results = torch.tensor([[1, 0]])

        attack_results_for_all_attacked_nodes.append(attack_results.type(torch.long))
        # check if the model is changed in between one node attacks
        if not (attack.mode.isAdversarial() and trainset):
            attack.setModel(model0)

    # print results and save accuracies
    attack_results_for_all_attacked_nodes = torch.cat(attack_results_for_all_attacked_nodes)
    mean_defence_results = \
        getDefenceResultsMean(attack=attack, approach=approach, attack_results=attack_results_for_all_attacked_nodes)
    attack.model_wrapper.model.attack = False

    if not trainset:
        print("######################## Attack Results ######################## ", flush=True)
        printAttackHeader(attack=attack, approach=approach)
        num_of_attackers = attack.default_multiple_num_of_attackers if approach.isMultiple() else 1

        printAttack(basic_log=attack.model_wrapper.basic_log, mean_defence_results=mean_defence_results,
                    approach=approach, max_attributes=data.x.shape[1] * num_of_attackers)

    return attack_results_for_all_attacked_nodes, attacked_nodes, y_targets


# a function which prints the header for the final results
def printAttackHeader(attack, approach: Approach):
    """
        print the header of the attack

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
    """
    distance_log = ''
    if attack.mode.isDistance():
        distance_log += 'Distance: {:02d} '.format(attack.current_distance)

    # the general print header
    targeted_attack_str = 'Targeted' if attack.targeted else 'Untargeted'
    print("######################## " + distance_log + targeted_attack_str + " " + approach.string() + " " +
          attack.model_wrapper.model.name + " Attack ########################", flush=True)
    info = "########################"
    if attack.dataset_type is DatasetType.CONTINUOUS:
        info += " Max Attack Epochs:" + str(attack.continuous_epochs)
    if approach.isMultiple():
        info += " Attackers:{}".format(attack.default_multiple_num_of_attackers)

    if attack.getDataset().type is DatasetType.CONTINUOUS:
        if attack.l_inf is not None:
            info += " Linf:{:.2f}".format(attack.l_inf)

    if attack.l_0 is not None:
        info += " l_0:{:.2f}".format(attack.l_0)

    info += " lr:" + str(attack.lr)
    print(info + " ########################", flush=True)


def getNodesToAttack(data: torch_geometric.data.Data, trainset: bool) -> Tuple[int, torch.Tensor]:
    """
        a wrapper that chooses a victim node and attacks it using attackVictim

        Parameters
        ----------
        data: torch_geometric.data.Data
        trainset: bool - whether or not this function is used for training or testing

        Returns
        -------
        num_attacks: int - the number of nodes to attack
        nodes_to_attack: torch.Tensor - the attacked/victim nodes in the train/val/test set
    """
    if trainset:
        num_attacks = torch.sum(data.train_mask).item()
        nodes_to_attack = np.where(np.array(data.train_mask.tolist()))[0]
    else:
        num_attacks = torch.sum(data.test_mask).item()
        nodes_to_attack = np.where(np.array(data.test_mask.tolist()))[0]
    return num_attacks, nodes_to_attack


def getClassificationTargets(attack, dataset: torch_geometric.data.Data, num_attacks: int, attacked_nodes: torch.Tensor) -> torch.Tensor:
    """
        a helper that returns the target of the attack task
        if the attack is targeted - returns the target classification
        if the attack is untargeted - returns the correct classification of the node

        Parameters
        ----------
        attack: oneGNNAttack
        dataset: torch_geometric.data.Data
        num_attacks: int - the number of nodes to attack
        attacked_nodes: torch.Tensor - the victim nodes

        Returns
        -------
        y_targets: torch.Tensor - the target labels of the attack
    """
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


def getDefenceResultsMean(attack, approach: Approach, attack_results: torch.Tensor) -> torch.Tensor:
    """
        calculates the mean for the defence results and the ratio of attributes used

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
        attack_results: torch.Tensor

        Returns
        -------
        mean_defence_results: torch.Tensor - 2d-tensor that includes
                                             1st-col - the defence
                                             2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
    """
    attack_results = attack_results.type(torch.FloatTensor)
    mask_attack_success = attack_results[:, 0] == 1

    mean_attack_success = mask_attack_success.sum().type(torch.FloatTensor) / attack_results.shape[0]
    if approach is not NodeApproach.AGREE and not attack.targeted:
        mean_attack_success = 1 - mean_attack_success

    mask_attributes_no_misclassification = torch.logical_and(mask_attack_success, attack_results[:, 1] != 0)
    mean_attributes = attack_results[mask_attributes_no_misclassification, 1].mean(dim=0)

    return torch.tensor([mean_attack_success, mean_attributes])


def printAttack(basic_log: str, mean_defence_results: torch.Tensor, approach: Approach, max_attributes: int):
    """
        print the final results for the attack

        Parameters
        ----------
        basic_log: str - a starting log template
        mean_defence_results: torch.Tensor - 2d-tensor that includes
                                             1st-col - the defence
                                             2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
        approach: Approach
        max_attributes: int
    """
    attack_log = ''
    if basic_log is not None:
        attack_log += basic_log + ', '
    if approach is NodeApproach.AGREE:
        attack_log += 'Agreement: {:.4f}\n'
        attack_log = attack_log.format(mean_defence_results[0].item())
    else:
        attack_log += 'Test Defence Success: {:.4f}\n'
        attack_log = attack_log.format(mean_defence_results[0].item())

        num_of_attack_attributes = mean_defence_results[1].item()
        mus = tuple([num_of_attack_attributes] + [num_of_attack_attributes / max_attributes])
        attack_log += '#Success attack attributes: {:.1f}, #Success attack l_0: {:.3f}\n'.format(*mus)

    print(attack_log, flush=True)
