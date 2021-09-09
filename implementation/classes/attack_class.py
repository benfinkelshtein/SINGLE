from classes.approach_classes import Approach, NodeApproach, EdgeApproach
from classes.basic_classes import GNN_TYPE
from attacks import (NodeGNNSAttack, EdgeGNNSAttack, NodeGNNSLinfAttack, NodeGNNSL0Attack,
                     NodeGNNSDistanceAttack, NodeGNNSAdversarialAttack, NodeGNNSMultipleAttack)

from enum import Enum, auto
from typing import List


class AttackMode(Enum):
    """
        an object for the different attack modes
    """
    NODE = auto()
    EDGE = auto()
    NODE_LINF = auto()
    NODE_L0 = auto()

    DISTANCE = auto()
    ADVERSARIAL = auto()

    MULTIPLE = auto()

    @staticmethod
    def from_string(s):
        try:
            return AttackMode[s]
        except KeyError:
            raise ValueError()

    def getAttack(self):
        """
            a get function for the required attack mode
            
            Returns
            -------
            attack_mode: AttackMode
        """
        if self is AttackMode.NODE:
            return NodeGNNSAttack
        elif self is AttackMode.NODE_LINF:
            return NodeGNNSLinfAttack
        elif self is AttackMode.EDGE:
            return EdgeGNNSAttack
        elif self is AttackMode.DISTANCE:
            return NodeGNNSDistanceAttack
        elif self is AttackMode.ADVERSARIAL:
            return NodeGNNSAdversarialAttack
        elif self is AttackMode.MULTIPLE:
            return NodeGNNSMultipleAttack
        elif self is AttackMode.NODE_L0:
            return NodeGNNSL0Attack

    def getApproaches(self, any_robust_gnn, is_twitter) -> List[Approach]:
        """
            gets the approaches for each attack mode

            Parameters
            ----------
            any_robust_gnn: bool - whether a robust_gnn iss included in the list of gnns
            is_twitter: bool - whether dataset is the TWITTER dataset

            Returns
            -------
            approaches: List[Approach]
        """
        if self is AttackMode.NODE or self is AttackMode.ADVERSARIAL:
            approaches = [NodeApproach.SINGLE, NodeApproach.INDIRECT, NodeApproach.MULTIPLE_ATTACKERS,
                          NodeApproach.DIRECT, NodeApproach.TOPOLOGY, NodeApproach.GRAD_CHOICE,
                          NodeApproach.ZERO_FEATURES]
            if not any_robust_gnn and not is_twitter:
                approaches.append(NodeApproach.INJECTION)
            return approaches
        elif self is AttackMode.EDGE:
            return [EdgeApproach.SINGLE, EdgeApproach.GRAD_CHOICE]

    def getGNN_TYPES(self) -> List[GNN_TYPE]:
        """
            gets the GNN types for each attack mode

            Returns
            -------
            gnn_types: List[GNN_TYPE]
        """
        if self is AttackMode.NODE or self is AttackMode.EDGE or self is AttackMode.NODE_LINF \
                or self is AttackMode.NODE_L0 or self is AttackMode.DISTANCE or self is AttackMode.MULTIPLE:
            return [GNN_TYPE.GCN, GNN_TYPE.GAT, GNN_TYPE.GIN, GNN_TYPE.SAGE]
        elif self is AttackMode.ADVERSARIAL:
            return [GNN_TYPE.GCN]

    def isNodeModel(self) -> bool:
        """
            whether or not the selected attack mode is node-based

            Returns
            -------
            is_node_based: bool
        """
        if self is AttackMode.EDGE:
            return False
        return True

    def isDistance(self) -> bool:
        """
            whether or not the selected attack mode is AttackMode.DISTANCE

            Returns
            -------
            is_distance: bool
        """
        if self is AttackMode.DISTANCE:
            return True
        return False

    def isAdversarial(self) -> bool:
        """
            whether or not the selected attack mode is AttackMode.ADVERSARIAL

            Returns
            -------
            is_adversarial: bool
        """
        if self is AttackMode.ADVERSARIAL:
            return True
        return False

    def getModeNode(self):
        """
            a get function for the AttackMode.Node object

            Returns
            -------
            the AttackMode.Node object
        """
        return AttackMode.NODE
