import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import os.path


def read_edge_index() -> np.array:
    """
        reads the edges of the dataset

        Returns
        ----------
        np_edges: np.array - the requested model
    """
    with open('./data/users.edges') as f:
        edges_str = f.read().splitlines()
        edge_int = []
        for edge in edges_str:
            node = edge.split(" ")
            edge_int.append(np.array(node, dtype=np.int32))
        np_edges = np.array(edge_int)
    return np.transpose(np_edges)


map_label_to_index = {'hateful': 2, 'normal': 0, 'other':1}


class TwitterDataset(InMemoryDataset):
    """
        a base class for the Twitter dataset

        more information at torch_geometric.data.InMemoryDataset
    """
    def __init__(self, root_path: os.path, transform=None, pre_transform=None, feature_matrix=None, label_matrix=None):
        self.features = feature_matrix
        self.labels = label_matrix
        super(TwitterDataset, self).__init__(root_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_hate_data.dataset'] #, '../data/input/processed_hate_all_data.dataset']

    def download(self):
        pass

    def process(self):
        #size: (2, E)
        np_edge_index = read_edge_index()
        #size (num_node, 1) (num_node, feature vector size)
        y, np_node_features = self.labels, self.features
        torch_edge_index = torch.LongTensor(np_edge_index)
        torch_node_features = torch.FloatTensor(np_node_features)
        torch_y = torch.LongTensor(y)
        data_list = [Data(x=torch_node_features, edge_index = torch_edge_index, y=torch_y)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
