from dataset_functions.twitter_dataset import TwitterDataset
from classes.basic_classes import DatasetType, DataSet
from helpers.getGitPath import getGitPath

from typing import NamedTuple
import os.path as osp
import pickle
import torch
from torch_geometric.datasets import Planetoid


class Masks(NamedTuple):
    train: torch.tensor
    val: torch.tensor
    test: torch.tensor


class GraphDataset(object):
    def __init__(self, dataset, device):
        super(GraphDataset, self).__init__()
        name = dataset.string()
        self.name = name
        self.device = device

        data = self._loadDataset(dataset, device)
        self._setMasks(data, name)
        self._setReversedArrayList(data)

        self.data = data
        self.type = dataset.get_type()
        self.skip_attributes = True if dataset.get_type() is DatasetType.CONTINUOUS else False

    def _loadDataset(self, dataset, device):
        dataset_path = osp.join(getGitPath(), 'datasets')
        if dataset is DataSet.PUBMED or dataset is DataSet.CORA or dataset is DataSet.CITESEER:
            dataset = Planetoid(dataset_path, dataset.string())
        elif dataset is DataSet.TWITTER:
            twitter_glove_path = osp.join(dataset_path, 'twitter', 'glove.pkl')
            if not osp.exists(twitter_glove_path):
                quit("Go to README and follow the download instructions to the TWITTER dataset")
            else:
                dataset = TwitterDataset(osp.dirname(twitter_glove_path))
                with open(twitter_glove_path, 'rb') as file:
                    glove_matrix = pickle.load(file)
                self.glove_matrix = torch.tensor(glove_matrix, dtype=torch.float32).to(device)
        # elif dataset is DataSet.PPI:
        #     PPI_path = osp.join(datasets_path, 'PPI')
        #     data_split = []
        #     for split in ('train', 'val', 'test'):
        #         PPI.url = 'https://data.dgl.ai/dataset/ppi.zip'
        #         data_split.append(PPI(PPI_path, split=split))
        # elif dataset is DataSet.FLICKR:
        #     dataset = Flickr(osp.join(datasets_path, 'Flickr'))

        data = dataset[0].to(self.device)
        setattr(data, 'num_classes', dataset.num_classes)

        self.num_features = data.num_features
        self.num_classes = dataset.num_classes
        return data

    def _setMasks(self, data, name):
        if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
            self.train_percent = train_percent = 0.1
            self.val_percent = val_percent = 0.3
            masks = self._generateMasks(data, name, train_percent, val_percent)
        else:
            masks = Masks(data.train_mask, data.val_mask, data.test_mask)

        setattr(data, 'train_mask', masks.train)
        setattr(data, 'val_mask', masks.val)
        setattr(data, 'test_mask', masks.test)

    def _generateMasks(self, data, name, train_percent, val_percent):
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
    def _setReversedArrayList(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        reversed_arr_list = [[] for _ in range(num_nodes)]

        for idx, column in enumerate(edge_index.T):
            edge_from = edge_index[0, idx].item()
            edge_to = edge_index[1, idx].item()

            # swapping positions to find all the neighbors that can go to the root
            reversed_arr_list[edge_to].append(edge_from)

        self.reversed_arr_list = reversed_arr_list
