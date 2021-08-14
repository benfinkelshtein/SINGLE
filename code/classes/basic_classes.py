from enum import Enum, auto
from torch import nn
from typing import List, Optional, Tuple
from torch_geometric.nn import GCNConv, SGConv
from model_functions.modified_gnns import ModifiedGATConv, ModifiedGINConv, ModifiedSAGEConv


class Print(Enum):
    """
        an object for the different types of print
    """
    YES = auto()
    PARTLY = auto()
    NO = auto()


class DatasetType(Enum):
    """
        an object for the different types of datasets
    """
    CONTINUOUS = auto()
    DISCRETE = auto()


class DataSet(Enum):
    """
        an object for the different datasets
    """
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

    def get_type(self) -> DatasetType:
        """
            gets the dataset type for each dataset

            Returns
            -------
            DatasetType
        """
        if self is DataSet.PUBMED or self is DataSet.TWITTER:
            return DatasetType.CONTINUOUS
        elif self is DataSet.CORA or self is DataSet.CITESEER:
            return DatasetType.DISCRETE

    def get_l_inf(self) -> float:
        """
            Get the default l_inf

            Returns
            -------
            l_inf: float
        """
        if self.get_type() is DatasetType.DISCRETE:
            return 1
        if self is DataSet.PUBMED:
            return 0.04
        if self is DataSet.TWITTER:
            return 0.001

    def get_l_0(self) -> float:
        """
            Get the default l_0

            Returns
            -------
            l_0: float
        """
        if self.get_type() is DatasetType.DISCRETE:
            return 0.01
        if self is DataSet.PUBMED:
            return 0.05
        if self is DataSet.TWITTER:
            return 0.05

    def string(self) -> str:
        """
            converts dataset to string

            Returns
            -------
            dataset_name: str
        """
        if self is DataSet.PUBMED:
            return "PubMed"
        elif self is DataSet.CORA:
            return "Cora"
        elif self is DataSet.CITESEER:
            return "CiteSeer"
        elif self is DataSet.TWITTER:
            return "twitter"


class GNN_TYPE(Enum):
    """
        an object for the different GNNs
    """
    GCN = auto()
    GAT = auto()
    SAGE = auto()
    GIN = auto()

    ROBUST_GCN = auto()
    SGC = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim: int, out_dim: int, K: Optional[int] = None):
        """
            get the GNN layer

            Parameters
            ----------
            in_dim: int - input dimension
            out_dim: int - output dimension
            K: int - number of layers for SGC only

            Returns
            -------
            layer: torch_geometric.nn
        """
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
        elif self is GNN_TYPE.SGC:
            return SGConv(in_channels=in_dim, out_channels=out_dim, K=K)

    def string(self) -> str:
        """
            converts gnn to string

            Returns
            -------
            gnn_name: str
        """
        if self is GNN_TYPE.GCN:
            return "GCN"
        elif self is GNN_TYPE.GAT:
            return "GAT"
        elif self is GNN_TYPE.SAGE:
            return "SAGE"
        elif self is GNN_TYPE.GIN:
            return "GIN"
        elif self is GNN_TYPE.ROBUST_GCN:
            return "ROBUST_GCN"
        elif self is GNN_TYPE.SGC:
            return "SGC"

    @staticmethod
    def convertGNN_TYPEListToStringList(gnn_list) -> List[str]:
        """
            converts a list of GNNs to a list of strings

            Parameters
            ----------
            gnn_list: List[GNN_TYPE]

            Returns
            -------
            gnn_names: List[str]
        """
        return [gnn.string() for gnn in gnn_list]
