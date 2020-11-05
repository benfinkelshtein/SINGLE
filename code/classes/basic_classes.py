from enum import Enum, auto
from torch import nn
from torch_geometric.nn import GCNConv
from classes.modified_gat import ModifiedGATConv
from classes.modified_gin import ModifiedGINConv
from classes.modified_sage import ModifiedSAGEConv


class Print(Enum):
    YES = auto()
    PARTLY = auto()
    NO = auto()


class DatasetType(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()


class DataSet(Enum):
    PUBMED = auto()
    CORA = auto()
    CITESEER = auto()
    TWITTER = auto()

    @staticmethod
    def from_string(s):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def get_type(self):
        if self is DataSet.PUBMED or self is DataSet.TWITTER:
            return DatasetType.CONTINUOUS
        elif self is DataSet.CORA or self is DataSet.CITESEER:
            return DatasetType.DISCRETE

    def string(self):
        if self is DataSet.PUBMED:
            return "PubMed"
        elif self is DataSet.CORA:
            return "Cora"
        elif self is DataSet.CITESEER:
            return "CiteSeer"
        elif self is DataSet.TWITTER:
            return "twitter"


class GNN_TYPE(Enum):
    GCN = auto()
    GAT = auto()
    SAGE = auto()
    GIN = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(in_channels=in_dim, out_channels=out_dim)
        elif self is GNN_TYPE.GAT:
            return ModifiedGATConv(in_channels=in_dim, out_channels=out_dim)
        elif self is GNN_TYPE.SAGE:
            return ModifiedSAGEConv(in_channels=in_dim, out_channels=out_dim)
        elif self is GNN_TYPE.GIN:
            sequential = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                       nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
            return ModifiedGINConv(sequential)

    def string(self):
        if self is GNN_TYPE.GCN:
            return "GCN"
        elif self is GNN_TYPE.GAT:
            return "GAT"
        elif self is GNN_TYPE.SAGE:
            return "SAGE"
        elif self is GNN_TYPE.GIN:
            return "GIN"

    @staticmethod
    def convertGNN_TYPEListToStringList(gnn_list):
        return [gnn.string() for gnn in gnn_list]
