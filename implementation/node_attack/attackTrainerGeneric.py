from classes.basic_classes import DatasetType
from node_attack.attackTrainerContinuous import attackTrainerContinuous
from node_attack.attackTrainerDiscrete import attackTrainerDiscrete

import torch


def attackTrainer(attack, attacked_nodes: torch.Tensor, y_targets: torch.Tensor, malicious_nodes: torch.Tensor,
                  node_num: int, discrete_stop_after_1iter: bool = False):
    """
        a gateway function between the two attack algorithms

        Parameters
        ----------
        attack: oneGNNAttack
        attacked_nodes: torch.Tensor - the victim nodes
        y_targets: torch.Tensor - the target labels of the attack
        malicious_nodes: torch.Tensor - the attacker/malicious node
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)
        discrete_stop_after_1iter: bool - whether or not to stop the discrete after 1 iteration
                                          this is a specific flag for the GRAD_CHOICE Approach

        Returns
        -------
        attack_results: torch.Tensor - 2d-tensor that includes
                                       1st-col - the defence
                                       2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
    """
    dataset = attack.getDataset()
    if dataset.type is DatasetType.CONTINUOUS:
        return attackTrainerContinuous(attack, attacked_nodes, y_targets, malicious_nodes, node_num)
    elif dataset.type is DatasetType.DISCRETE:
        return attackTrainerDiscrete(attack, attacked_nodes, y_targets, malicious_nodes, node_num,
                                     discrete_stop_after_1iter)
    else:
        quit("Unrecognised dataset")
