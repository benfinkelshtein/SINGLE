from enum import Enum, auto
import torch
import numpy as np

from helpers.algorithms import heuristicApproach
from helpers.algorithms import gradientApproach


class Approach(Enum):
    def string(self):
        raise NotImplementedError

    @staticmethod
    def convertApprochesListToStringList(approches_list):
        return [approch.string() for approch in approches_list]


class NodeApproach(Approach):
    SINGLE = auto()
    INDIRECT = auto()
    TWO_ATTACKERS = auto()
    DIRECT = auto()
    TOPOLOGY = auto()
    GRAD_CHOICE = auto()
    AGREE = auto()

    # a helper function which calculates the malicious node for the basic approaches (every approach except agree)
    def getMaliciousNode(self, attack, attacked_node, y_target, node_num, neighbours_and_dist, BFS_size):
        malicious_node = None
        if self is NodeApproach.SINGLE or self is NodeApproach.INDIRECT or self is NodeApproach.DIRECT:
            malicious_index = np.random.choice(BFS_size, 1)
            malicious_node, _ = neighbours_and_dist[malicious_index.item()]
            malicious_node = torch.tensor([malicious_node.item()]).to(attack.device)
        elif self is NodeApproach.TOPOLOGY:
            malicious_node = heuristicApproach(reversed_arr_list=attack.dataset.reversed_arr_list,
                                               neighbours_and_dist=neighbours_and_dist,
                                               device=attack.device)
        elif self is NodeApproach.GRAD_CHOICE:
            malicious_node = gradientApproach(attack=attack, attacked_node=attacked_node, y_target=y_target,
                                              node_num=node_num, neighbours_and_dist=neighbours_and_dist, approach=self)
        elif self is NodeApproach.TWO_ATTACKERS:
            malicious_indices = np.random.choice(BFS_size, 2)
            malicious_indices = torch.tensor(malicious_indices.tolist()).to(attack.device)
            malicious_node, _ = neighbours_and_dist[malicious_indices, :]
        return malicious_node

    def isMultipleMaliciousNodes(self):
        if self is NodeApproach.AGREE or self is NodeApproach.GRAD_CHOICE:
            return True
        else:
            return False

    def string(self):
        if self is NodeApproach.SINGLE:
            return "single"
        elif self is NodeApproach.INDIRECT:
            return "indirect"
        elif self is NodeApproach.TWO_ATTACKERS:
            return "two-attackers"
        elif self is NodeApproach.DIRECT:
            return "direct"
        elif self is NodeApproach.TOPOLOGY:
            return "topology"
        elif self is NodeApproach.GRAD_CHOICE:
            return "grad-choice"
        elif self is NodeApproach.AGREE:
            return "agree"


class EdgeApproach(Approach):
    RANDOM = auto()
    GRAD = auto()
    GLOBAL_GRAD = auto()

    def string(self):
        if self is EdgeApproach.RANDOM:
            return "random"
        elif self is EdgeApproach.GRAD:
            return "grad"
        elif self is EdgeApproach.GLOBAL_GRAD:
            return "global-grad"
