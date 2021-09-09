from helpers.algorithms import heuristicApproach, gradientApproach

from typing import List
from enum import Enum, auto
import torch
import numpy as np


class Approach(Enum):
    """
        a generic object for the different approaches
    """
    def string(self):
        """
            converts approach to string

            Returns
            -------
            approach_name: str
        """
        raise NotImplementedError

    @staticmethod
    def convertApprochesListToStringList(approches_list: List[Enum]) -> List[str]:
        """
            converts a list of GNNs to a list of strings

            Parameters
            ----------
            approches_list: List[Approach]

            Returns
            -------
            approach_names: List[str]
        """
        return [approch.string() for approch in approches_list]


class NodeApproach(Approach):
    """
        an object for the different node-based-attack approaches
    """
    SINGLE = auto()
    INDIRECT = auto()
    DIRECT = auto()
    TOPOLOGY = auto()
    GRAD_CHOICE = auto()
    AGREE = auto()

    INJECTION = auto()
    ZERO_FEATURES = auto()

    MULTIPLE_ATTACKERS = auto()

    def getMaliciousNode(self, attack, attacked_node: torch.Tensor, y_target: torch.Tensor, node_num: int,
                         neighbours_and_dist: torch.Tensor, BFS_size: int):
        """
            a get function for the malicious/attacker node, chosen by approach

            the agree approach is an exception as it test whether or not
            GRAD_CHOICE and TOPOLOGY select the same node

            Parameters
            ----------
            attack: oneGNNAttack
            attacked_node: torch.Tensor -  the victim node
            y_target: torch.Tensor - the target label of the attack
            node_num: int - the index of the attacked/victim node (out of the train/val/test-set)
            neighbours_and_dist: torch.Tensor - 2d-tensor that includes
                                                1st-col - the nodes that are in the victim nodes BFS neighborhood
                                                2nd-col - the distance of said nodes from the victim node
            BFS_size: int - the size of the victim nodes BFS neighborhood (does not include the victim node itself)

            Returns
            -------
            malicious_node:  torch.Tensor
            attack: oneGNNAttack
        """
        malicious_node = None
        if self is NodeApproach.SINGLE or self is NodeApproach.INDIRECT or self is NodeApproach.DIRECT or self is \
                NodeApproach.ZERO_FEATURES:
            malicious_index = np.random.choice(BFS_size, 1)
            malicious_node, _ = neighbours_and_dist[malicious_index.item()]
            malicious_node = torch.tensor([malicious_node.item()]).to(attack.device)
        elif self is NodeApproach.TOPOLOGY:
            malicious_node = heuristicApproach(reversed_arr_list=attack.getDataset().reversed_arr_list,
                                               neighbours_and_dist=neighbours_and_dist,
                                               device=attack.device)
        elif self is NodeApproach.GRAD_CHOICE:
            malicious_node = gradientApproach(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                              node_num=node_num, neighbours_and_dist=neighbours_and_dist)
        elif self is NodeApproach.INJECTION:
            malicious_node, dataset = attack.model_wrapper.model.injectNode(dataset=attack.getDataset(),
                                                                            attacked_node=attacked_node)
            attack.setDataset(dataset=dataset)
        elif self is NodeApproach.MULTIPLE_ATTACKERS:
            if BFS_size < attack.default_multiple_num_of_attackers:
                return None, attack
            malicious_indices = np.random.choice(BFS_size, attack.default_multiple_num_of_attackers, replace=False)
            malicious_indices = torch.tensor(malicious_indices.tolist()).to(attack.device)
            malicious_node = neighbours_and_dist[malicious_indices, 0]
        return malicious_node, attack

    def isMultiple(self) -> bool:
        """
            whether or not the selected approach is the MULTIPLE_ATTACKERS approach

            Returns
            -------
            is_multiple: bool
        """
        if self is NodeApproach.MULTIPLE_ATTACKERS:
            return True
        return False

    def string(self) -> str:
        """
            converts approach to string

            Returns
            -------
            approach_name: str
        """
        if self is NodeApproach.SINGLE:
            return "single"
        elif self is NodeApproach.INDIRECT:
            return "indirect"
        elif self is NodeApproach.DIRECT:
            return "direct"
        elif self is NodeApproach.TOPOLOGY:
            return "topology"
        elif self is NodeApproach.GRAD_CHOICE:
            return "grad-choice"
        elif self is NodeApproach.AGREE:
            return "agree"
        elif self is NodeApproach.INJECTION:
            return "injection"
        elif self is NodeApproach.ZERO_FEATURES:
            return "zero-features"
        elif self is NodeApproach.MULTIPLE_ATTACKERS:
            return "multiple-attackers"


class EdgeApproach(Approach):
    """
        an object for the different edge-based-attack approaches
    """
    RANDOM = auto()
    SINGLE = auto()
    GRAD_CHOICE = auto()

    MULTI = auto()
    MULTI_GRAD_CHOICE = auto()

    def isGlobal(self) -> bool:
        """
            whether or not the selected approach is a GLOBAL approach

            Returns
            -------
            is_global: bool
        """
        if self is EdgeApproach.GRAD_CHOICE or self is EdgeApproach.MULTI_GRAD_CHOICE:
            return True
        return False

    def isMulti(self) -> bool:
        """
            whether or not the selected approach is a MUTLI approach

            Returns
            -------
            is_multi: bool
        """
        if self is EdgeApproach.MULTI or self is EdgeApproach.MULTI_GRAD_CHOICE:
            return True
        return False

    def string(self) -> str:
        """
            converts approach to string

            Returns
            -------
            approach_name: str
        """
        if self is EdgeApproach.RANDOM:
            return "random"
        elif self is EdgeApproach.SINGLE:
            return "single"
        elif self is EdgeApproach.GRAD_CHOICE:
            return "grad-choice"
        elif self is EdgeApproach.MULTI:
            return "multi"
        elif self is EdgeApproach.MULTI_GRAD_CHOICE:
            return "multi-grad-choice"
