from dataset_functions.twitter_dataset import TwitterDataset
from classes.basic_classes import DataSet
from helpers.getGitPath import getGitPath

from typing import NamedTuple
import os.path as osp
import pickle
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid


class Masks(NamedTuple):
    """
        a Mask object with the following fields:
    """
    train: torch.tensor
    val: torch.tensor
    test: torch.tensor


class GraphDataset(object):
    """
        a base class for datasets

        Parameters
        ----------
        dataset: DataSet
        device: torch.device
    """
    def __init__(self, dataset: DataSet, device: torch.device):
        super(GraphDataset, self).__init__()
        name = dataset.string()
        self.name = name
        self.device = device

        data = self._loadDataset(dataset, device)

        if dataset is DataSet.TWITTER:
            train_mask, val_mask, test_mask = torch.load('./masks/twitter.dat')
            setattr(data, 'train_mask', train_mask)
            setattr(data, 'val_mask', val_mask)
            setattr(data, 'test_mask', test_mask)
            setattr(data, 'test_mask', test_mask)
            self.num_features = 200# after multiplying x by the golve matrix the new feature dim is 200
            self.num_classes = data.num_classes
        else:
            self._setMasks(data, name)

        self._setReversedArrayList(data)

        self.data = data
        self.type = dataset.get_type()

    def _loadDataset(self, dataset: DataSet, device: torch.device) -> torch_geometric.data.Data:
        """
            a loader function for the requested dataset
        """
        dataset_path = osp.join(getGitPath(), 'datasets')
        if dataset is DataSet.PUBMED or dataset is DataSet.CORA or dataset is DataSet.CITESEER:
            dataset = Planetoid(dataset_path, dataset.string())
        elif dataset is DataSet.TWITTER:
            twitter_glove_path = osp.join(dataset_path, 'twitter', 'glove.pkl')
            if not osp.exists(twitter_glove_path):
                exit("Go to README and follow the download instructions to the TWITTER dataset")
            else:
                dataset = TwitterDataset(osp.dirname(twitter_glove_path))
                with open(twitter_glove_path, 'rb') as file:
                    glove_matrix = pickle.load(file)
                self.glove_matrix = torch.tensor(glove_matrix, dtype=torch.float32).to(device)

        data = dataset[0].to(self.device)
        setattr(data, 'num_classes', dataset.num_classes)

        self.num_features = data.num_features
        self.num_classes = dataset.num_classes
        return data

    def _setMasks(self, data: torch_geometric.data.Data, name: str):
        """
            sets train,val and test mask for the data

            Parameters
            ----------
            data: torch_geometric.data.Data
            name: str
        """
        if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
            self.train_percent = train_percent = 0.1
            self.val_percent = val_percent = 0.3
            masks = self._generateMasks(data, name, train_percent, val_percent)
        else:
            masks = Masks(data.train_mask, data.val_mask, data.test_mask)

        setattr(data, 'train_mask', masks.train)
        setattr(data, 'val_mask', masks.val)
        setattr(data, 'test_mask', masks.test)

    def _generateMasks(self, data: torch_geometric.data.Data, name: str, train_percent: float, val_percent: float)\
            -> Masks:
        """
            generates train,val and test mask for the data

            Parameters
            ----------
            data: torch_geometric.data.Data
            name: str
            train_percent: float
            val_percent: float

            Returns
            -------
            masks: Masks
        """
        train_mask = torch.zeros(data.num_nodes).type(torch.bool)
        val_mask = torch.zeros(data.num_nodes).type(torch.bool)
        test_mask = torch.zeros(data.num_nodes).type(torch.bool)

        # taken from Planetoid
        for c in range(data.num_classes):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            num_train_per_class = round(idx.size(0) * train_percent)
            num_val_per_class = round(idx.size(0) * val_percent)

            idx_permuted = idx[torch.randperm(idx.size(0))]
            train_idx = idx_permuted[:num_train_per_class]
            val_idx = idx_permuted[num_train_per_class:num_train_per_class + num_val_per_class]
            test_idx = idx_permuted[num_train_per_class + num_val_per_class:]

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            print(f'Class {c}: training: {train_idx.size(0)}, val: {val_idx.size(0)}, test: {test_idx.size(0)}')

        masks = Masks(train_mask, val_mask, test_mask)
        torch.save(masks, osp.join(getGitPath(), 'masks', name + '.dat'))
        return masks

    # converting graph edge index representation to graph array list representation
    def _setReversedArrayList(self, data: torch_geometric.data.Data):
        """
            creates a reversed array list from the edges

            Parameters
            ----------
            data: torch_geometric.data.Data
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        reversed_arr_list = [[] for _ in range(num_nodes)]

        for idx, column in enumerate(edge_index.T):
            edge_from = edge_index[0, idx].item()
            edge_to = edge_index[1, idx].item()

            # swapping positions to find all the neighbors that can go to the root
            reversed_arr_list[edge_to].append(edge_from)

        self.reversed_arr_list = reversed_arr_list
