from node_attack.attackTrainerGeneric import attackTrainer
from node_attack.attackTrainerHelpers import test, createLogTemplate
from helpers.algorithms import kBFS, heuristicApproach, gradientApproach
from classes.approach_classes import Approach, NodeApproach
from classes.basic_classes import Print, DatasetType

import torch_geometric
import torch
import copy


def attackVictim(attack, approach: Approach, attacked_node: torch.Tensor, y_target: torch.Tensor, node_num: int)\
        -> torch.Tensor:
    """
        chooses the node we attack with (the malicious node) from our BFS environment
        this BFS environments is also calculated according to our selected approach
        lastly, we attack using attackTrainer
        important note: the victim node is already known (attacked node)

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
        attacked_node: torch.Tensor - the victim node
        y_target: torch.Tensor - the target label of the attack
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)

        Returns
        -------
        attack_results: torch.Tensor - 2d-tensor that includes
                                       1st-col - the defence
                                       2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
    """
    device = attack.device
    dataset = attack.getDataset()
    print_answer = attack.print_answer

    neighbours_and_dist = kBFS(root=attacked_node, device=device, reversed_arr_list=dataset.reversed_arr_list,
                               K=attack.num_layers)
    if neighbours_and_dist.nelement():
        neighbours_and_dist = manipulateNeighborhood(attack=attack, approach=approach, attacked_node=attacked_node,
                                                     neighbours_and_dist=neighbours_and_dist, device=device)
        attack_log = 'Attack: {:03d}, Node: {}, BFS clique: {}'.format(node_num, attacked_node.item(),
                                                                       neighbours_and_dist.shape[0] + 1)
    else:
        attack_log = 'Attack: {:03d}, Node: {} is a solo node'.format(node_num, attacked_node.item())

    # special cases of solo node and duo node for double
    BFS_size = neighbours_and_dist.shape[0]
    if not neighbours_and_dist.nelement():
        if print_answer is Print.YES:
            print(attack_log, flush=True)
        return None

    if print_answer is Print.YES:
        print(attack_log, end='', flush=True)
        if approach is not NodeApproach.MULTIPLE_ATTACKERS and print_answer is Print.YES:
            print()
    malicious_node, attack = approach.getMaliciousNode(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                                       node_num=node_num, neighbours_and_dist=neighbours_and_dist,
                                                       BFS_size=BFS_size)
    # calculates the malicious node for the irregular approaches
    if approach is NodeApproach.AGREE:
        if print_answer is Print.YES:
            print()
        malicious_node_heuristic = heuristicApproach(reversed_arr_list=dataset.reversed_arr_list,
                                                     neighbours_and_dist=neighbours_and_dist,
                                                     device=attack.device)
        malicious_node_gradient = gradientApproach(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                                   node_num=node_num, neighbours_and_dist=neighbours_and_dist)
        attack_results = torch.zeros(1, 2)
        attack_results[0][0] = malicious_node_heuristic == malicious_node_gradient  # in attackSet we change to equal
        return attack_results

    if approach is NodeApproach.ZERO_FEATURES:
        model = attack.model_wrapper.model
        data = dataset.data
        zero_model = copy.deepcopy(model)
        # train
        zero_model.node_attribute_list[malicious_node][:] = 0

        # test correctness
        changed_attributes = (zero_model.getInput() != model.getInput())[malicious_node].sum().item()

        # test
        results = test(data=data, model=zero_model, targeted=attack.targeted,
                       attacked_nodes=attacked_node, y_targets=y_target)
        if print_answer is Print.YES:
            log_template = createLogTemplate(attack=attack, dataset=dataset) + ', Attack Success: {}\n'
            if dataset.type is DatasetType.DISCRETE:
                print(log_template.format(node_num, 1, changed_attributes, *results), flush=True)
            if dataset.type is DatasetType.CONTINUOUS:
                print(log_template.format(node_num, 1, *results), flush=True)
        attack_results = torch.tensor([[results[3], changed_attributes]])
        return attack_results

    if approach is NodeApproach.MULTIPLE_ATTACKERS:
        if malicious_node is None:
            if print_answer is Print.YES:
                print(f': Too small for {attack.default_multiple_num_of_attackers} attackers\n', flush=True)
            return None
        else:
            print()

    if approach is NodeApproach.INJECTION:
        dataset = attack.getDataset()
        classified_to_target = checkNodeClassification(attack=attack, dataset=dataset,
                                                       attacked_node=attacked_node, y_target=y_target,
                                                       print_answer=Print.NO, attack_num=node_num + 1)
        if not classified_to_target:
            if print_answer is Print.YES:
                print("misclassified right after injection!\n", flush=True)
            attack.model_wrapper.model.removeInjectedNode(attack=attack)
            return torch.tensor([[1, 0]])

    attack_results = attackTrainer(attack=attack, attacked_nodes=attacked_node, y_targets=y_target,
                                   malicious_nodes=malicious_node, node_num=node_num)

    if approach is NodeApproach.INJECTION:
        attack.model_wrapper.model.removeInjectedNode(attack=attack)
    return attack_results


def manipulateNeighborhood(attack, approach: Approach, attacked_node: torch.Tensor, neighbours_and_dist,
                           device: torch.cuda) -> torch.tensor:
    """
        manipulates the BFS neighborhood according to our selected approach

        Parameters
        ----------
        attack: oneGNNAttack
        approach: Approach
        attacked_node: torch.Tensor - the victim node
        neighbours_and_dist: torch.Tensor - 2d-tensor that includes
                                            1st-d - the nodes that are in the victim nodes BFS neighborhood
                                            2nd-d - the distance of said nodes from the victim node
        device: torch.cuda

        Returns
        -------
        neighbours_and_dist: torch.Tensor
    """
    if attack.mode.isDistance():
        neighbours_and_dist = neighbours_and_dist[neighbours_and_dist[:, 1] == attack.current_distance, :]

    if approach is NodeApproach.INDIRECT:
        return neighbours_and_dist[neighbours_and_dist[:, 1] != 1, :]
    elif approach is NodeApproach.DIRECT:
        return torch.tensor([[attacked_node.item(), 0]]).to(device)
    else:
        return neighbours_and_dist


@torch.no_grad()
def checkNodeClassification(attack, dataset: torch_geometric.data.Data, attacked_node: torch.Tensor, y_target: torch.Tensor,
                            print_answer: Print, attack_num):
    """
        checks if the node is currecly classified to y_target

        Parameters
        ----------
        attack: oneGNNAttack
        dataset: torch_geometric.data.Data
        attacked_node: torch.Tensor - the victim node
        y_target: torch.Tensor - the target labels of the attack
        print_answer: Print - the type of print
        attack_num: int - the index of the node (out of the train/val/test-set)

        Returns
        -------
        classified_to_target: torch.Tensor - the defence of the model
    """
    results = test(dataset.data, attack.model_wrapper.model, attack.targeted, attacked_node, y_target)
    classified_to_target = not results[3]

    if not classified_to_target and print_answer is Print.YES:
        attack_log = 'Attack: {:03d}, Node: {}, Misclassified already!\n' \
            .format(attack_num, attacked_node.item())
        print(attack_log, flush=True)
    return classified_to_target
