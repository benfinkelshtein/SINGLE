from classes.approach_classes import NodeApproach, EdgeApproach
from classes.basic_classes import GNN_TYPE
from attacks import (NodeGNNSAttack, EdgeGNNSAttack, NodeGNNSLinfAttack, NodeGNNSDistanceAttack,
                     NodeGNNSAdversarialAttack)

from enum import Enum, auto


class AttackMode(Enum):
    NODE = auto()
    EDGE = auto()
    NODE_LINF = auto()

    DISTANCE = auto()
    ADVERSARIAL = auto()

    @staticmethod
    def from_string(s):
        try:
            return AttackMode[s]
        except KeyError:
            raise ValueError()

    def getAttack(self):
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

    def getApproaches(self):
        if self is AttackMode.NODE:
            return [NodeApproach.SINGLE, NodeApproach.INDIRECT, NodeApproach.TWO_ATTACKERS, NodeApproach.DIRECT,
                    NodeApproach.TOPOLOGY, NodeApproach.GRAD_CHOICE, NodeApproach.AGREE]
        elif self is AttackMode.EDGE:
            return [EdgeApproach.RANDOM, EdgeApproach.GRAD, EdgeApproach.GLOBAL_GRAD]
        elif self is AttackMode.NODE_LINF or self is AttackMode.DISTANCE or self is AttackMode.ADVERSARIAL:
            return [NodeApproach.SINGLE]

    def getGNN_TYPES(self):
        if self is AttackMode.NODE or self is AttackMode.EDGE or self is AttackMode.NODE_LINF or \
                self is AttackMode.DISTANCE:
            return [GNN_TYPE.GCN, GNN_TYPE.GAT, GNN_TYPE.GIN, GNN_TYPE.SAGE]
        elif self is AttackMode.ADVERSARIAL:
            return [GNN_TYPE.GCN]

    def isNodeModel(self):
        if self is AttackMode.EDGE:
            return False
        return True

    def isDistance(self):
        if self is AttackMode.DISTANCE:
            return True
        return False

    def isAdversarial(self):
        if self is AttackMode.ADVERSARIAL:
            return True
        return False

    def getModeNode(self):
        return AttackMode.NODE
