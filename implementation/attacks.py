from model_functions.graph_model import Model, ModelWrapper, AdversarialModelWrapper
from dataset_functions.graph_dataset import GraphDataset
from node_attack.attackSet import attackSet, printAttackHeader, getDefenceResultsMean
from classes.basic_classes import Print, DatasetType, GNN_TYPE, DataSet
from helpers.fileNamer import fileNamer
from classes.approach_classes import Approach
from edge_attack.edgeAttackSet import edgeAttackSet

from argparse import ArgumentParser
import torch_geometric
import torch
import numpy as np
import random
import pandas as pd
from typing import Tuple
import copy


class oneGNNAttack(object):
    """
        Generic attack class

        Parameters
        ----------
        args: ArgumentParser - command line inputs
        start_to_file: str - string to insert at the start of the saved file
        print_answer: Print - the type of output print
                              more information at classes.basic_classes.Print
    """
    def __init__(self, args: ArgumentParser, start_to_file: str, print_answer: Print):
        self.start_to_file = start_to_file
        self.print_answer = print_answer
        self.end_to_file = '.csv'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed = args.seed

        self.mode = args.attMode
        self.dataset_name = args.dataset
        dataset = GraphDataset(args.dataset, device)
        self.__dataset = dataset
        self.dataset_type = args.dataset.get_type()

        self.singleGNN = args.singleGNN
        if args.singleGNN is None:
            self.gnn_types = args.attMode.getGNN_TYPES()
        else:
            self.gnn_types = [args.singleGNN]

        self.num_layers = args.num_layers if args.num_layers is not None else 2
        self.patience = args.patience

        if dataset.type is DatasetType.CONTINUOUS:
            self.continuous_epochs = args.continuous_epochs
        self.lr = args.lr

        if args.l_inf is None:
            args.l_inf = args.dataset.get_l_inf()
        self.l_inf = args.l_inf
        if args.l_0 is None:
            args.l_0 = args.dataset.get_l_0()
        self.l_0 = args.l_0
        self.targeted = args.targeted

        self.max_distance = args.distance

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.device = device

        print(f'######################## STARTING ATTACK ########################')
        self.print_args(args)

        # use set functions
        self.setFileName(dataset, args)

        # *PARTLY* checking correctness of the inputs
        self.checkDistanceFlag(args)

    def setDataset(self, dataset: torch_geometric.data.Data):
        """
            Sets a dataset
            
            Parameters
            ----------
            dataset: torch_geometric.data.Data
        """
        self.__dataset = dataset

    def getDataset(self):
        """
            get a copy of the dataset
            
            Returns
            -------
            dataset: torch_geometric.data.Data
        """
        return copy.deepcopy(self.__dataset)

    def checkDistanceFlag(self, args: ArgumentParser):
        """
            Validates that the distance argument is not requested
            
            Parameters
            ----------
            args: ArgumentParser - command line inputs
        """
        if args.distance is not None:
            exit("This attack doesn't requires the distance flag")

    def setModelWrapper(self, gnn_type: GNN_TYPE):
        """
            Sets a ModelWrapper object and trains said ModelWrapper
            
            Parameters
            ----------
            gnn_type: GNN_TYPE - the type of the gnn
                                 more information at classes.basic_classes.GNN_TYPE
        """
        dataset = self.getDataset()
        self.model_wrapper = ModelWrapper(node_model=self.mode.isNodeModel(), gnn_type=gnn_type,
                                          num_layers=self.num_layers, dataset=dataset, patience=self.patience,
                                          device=self.device, seed=self.seed)
        print(f'######################## LOADING MODEL {self.model_wrapper.model.name} ########################')
        self.model_wrapper.train(dataset)

    def print_args(self, args: ArgumentParser):
        """
            a print of the arguments passed to the main.py
            
            Parameters
            ----------
            args: ArgumentParser - command line inputs
        """
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print()

    def run(self):
        """
            executes the requested attack for all gnn_types and approaches
        """
        defence, attributes = [], []
        for gnn_type in self.gnn_types:
            self.setModelWrapper(gnn_type)
            tmp_defence, tmp_attributes = self.attackPerGNN()
            defence.append(tmp_defence)
            attributes.append(tmp_attributes)

        defence = torch.cat(defence).to(self.device)
        attributes = torch.cat(attributes).to(self.device)
        self.saveResults(defence=defence, attributes=attributes)

    def attackPerApproachWrapper(self, approach: Approach) -> Tuple[torch.Tensor]:
        """
            sets seeds before the execution of the requested attack
            (for a specific approach on a specific gnn_type)
            
            Parameters
            ----------
            approach: Approach - the type of attack approach
                                 more information at classes.approach_classes.Approach
        """
        seed = self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return self.attackPerApproach(approach=approach)

    def setFileName(self, dataset: GraphDataset, args: ArgumentParser):
        """
            sets the generic name for the output file
            
            Parameters
            ----------
            dataset: GraphDataset
            args: ArgumentParser - command line inputs
        """
        if self.singleGNN is None:
            self.file_name = fileNamer(dataset_name=dataset.name, l_inf=args.l_inf, l_0=args.l_0,
                                       num_layers=args.num_layers, seed=args.seed, targeted=args.targeted,
                                       continuous_epochs=args.continuous_epochs, start=self.start_to_file,
                                       end=self.end_to_file)
        else:
            self.file_name = fileNamer(dataset_name=dataset.name, model_name=args.singleGNN.string(), l_inf=args.l_inf,
                                       l_0=args.l_0, num_layers=args.num_layers, seed=args.seed, targeted=args.targeted,
                                       continuous_epochs=args.continuous_epochs, start=self.start_to_file,
                                       end=self.end_to_file)

    def extendLog(self, log_start: str, log_end: str) -> str:
        """
            sets the generic output log format
            
            Parameters
            ----------
            log_start: str - prefix of the log format
            log_end: str -  suffix of the log format
            
            Returns
            -------
            log: str - output log format
        """
        if self.mode.isDistance():
            log = log_start + ' Distance: {:02d}'.format(self.current_distance) + log_end
        else:
            log = log_start + log_end
        return log

    def setModel(self, model: Model):
        """
            sets the requested model in the ModeWrapper
            
            Parameters
            ----------
            model: Model - the requested model
        """
        self.model_wrapper.setModel(model)

    def saveResults(self, defence: torch.Tensor, attributes: torch.Tensor):
        """
            saves the results of the attack
            
            Parameters
            ----------
            defence: torch.Tensor - the defence %
            attributes: torch.Tensor - the % of attributes used for a *successful attack*
        """
        raise NotImplementedError

    def attackPerGNN(self) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for all approaches on a specific gnn_type
        """
        raise NotImplementedError

    def attackPerApproach(self, approach) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for a specific approach on a specific gnn_type
            
            Parameters
            ----------
            approach: Approach - the type of attack approach
                                 more information at classes.approach_classes.Approach
            Returns
            -------
            defence: torch.Tensor - the defence %
            attributes: torch.Tensor - the % of attributes used for a *successful attack*
        """
        raise NotImplementedError


class NodeGNNSAttack(oneGNNAttack):
    """
        the basic Node-based-attack class
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
        start_to_file: str - string to insert at the start of the saved file
        print_answer: Print - the type of output print
                              more information at classes.basic_classes.Print
    """
    def __init__(self, args: ArgumentParser, start_to_file: str = None, print_answer: str = None):
        start_to_file = 'NodeAttack' if start_to_file is None else start_to_file
        print_answer = Print.YES if print_answer is None else print_answer
        self.default_multiple_num_of_attackers = 2

        super(NodeGNNSAttack, self).__init__(args=args, start_to_file=start_to_file,
                                             print_answer=print_answer)

        any_robust_gnn = sum([gnn.is_robust_model() for gnn in self.gnn_types])
        is_twitter = self.dataset_name is DataSet.TWITTER
        self.approaches = self.mode.getApproaches(any_robust_gnn=any_robust_gnn, is_twitter=is_twitter)

    # a must-create
    def saveResults(self, defence: torch.Tensor, attributes: torch.Tensor):
        """
            information at the generic base class oneGNNSAttack
        """
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        approaches = Approach.convertApprochesListToStringList(self.approaches)
        header = ['', 'clean'] + approaches
        defence_df = pd.DataFrame(defence.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    def attackPerGNN(self) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for all approaches on a specific gnn_type
        """
        defence = torch.zeros(len(self.approaches) + 1).to(self.device)
        attributes = torch.zeros(len(self.approaches) + 1).to(self.device)
        for approach_idx, approach in enumerate(self.approaches):
            tmp_defence, tmp_attributes = self.attackPerApproachWrapper(approach)
            defence[approach_idx + 1] = tmp_defence
            attributes[approach_idx + 1] = tmp_attributes

        defence[0] = self.model_wrapper.clean
        attributes[0] = 0
        return defence.unsqueeze(0), attributes.unsqueeze(0)

    def attackPerApproach(self, approach: Approach) -> Tuple[torch.Tensor]:
        """
            information at the generic base class oneGNNSAttack
        """
        results, _, _ = attackSet(self, approach=approach, trainset=False)
        mean_results = getDefenceResultsMean(attack=self, approach=approach, attack_results=results)
        return mean_results[0], mean_results[1]

    # create
    def setDefaultNumOfAttackers(self, num_of_attackers: int):
        """
            sets the number of attackers
            
            Parameters
            ----------
            num_of_attackers: int
        """
        self.default_multiple_num_of_attackers = num_of_attackers


class EdgeGNNSAttack(NodeGNNSAttack):
    """
        the basic Edge-based-attack class
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    def __init__(self, args: ArgumentParser):
        if args.singleGNN is not None and args.singleGNN == GNN_TYPE.ROBUST_GCN:
            exit("Robust_GCN is not available for an Edge attack")

        super(EdgeGNNSAttack, self).__init__(args=args, start_to_file='EdgeAttack', print_answer=Print.YES)

    # overriding
    def attackPerApproach(self, approach: Approach) -> Tuple[torch.Tensor]:
        """
            information at the generic base class oneGNNSAttack
        """
        if self.print_answer is Print.YES:
            print_flag = True
        else:
            print_flag = False
        defence = edgeAttackSet(self, approach=approach, print_flag=print_flag)
        return defence, torch.zeros_like(defence)


class NodeGNNSLinfAttack(NodeGNNSAttack):
    """
        a Node-based-attack class that tests different Linf values
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    def __init__(self, args: ArgumentParser):
        super(NodeGNNSLinfAttack, self).__init__(args=args, start_to_file='NodeLinfAttack', print_answer=Print.YES)
        self.l_inf_list = [0.01, 0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.checkL_infFlag(self.getDataset())

    # a must-create
    def saveResults(self, defence: torch.Tensor, attributes: torch.Tensor):
        """
            information at the generic base class oneGNNSAttack
        """
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        l_infs_string = [str(l_inf) for l_inf in self.l_inf_list]
        header = [''] + l_infs_string
        defence_df = pd.DataFrame(defence.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv('Def_' + self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def checkL_infFlag(self, dataset: GraphDataset):
        """
            Validates that the dataset is not discrete in an L_inf based attack
            
            Parameters
            ----------
            dataset: GraphDataset
        """
        if dataset.type is DatasetType.DISCRETE:
            exit("L_inf attack isn't suitable for discrete datasets")

    def attackPerGNN(self) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for the requested l_inf values on a specific gnn_type
        """
        defence = torch.zeros(len(self.l_inf_list)).to(self.device)
        attributes = torch.zeros(len(self.l_inf_list)).to(self.device)
        for l_inf_idx, l_inf in enumerate(self.l_inf_list):
            self.setLinf(l_inf)
            tmp_defence, tmp_attributes = self.attackPerApproachWrapper(approach=NodeApproach.SINGLE)
            defence[l_inf_idx] = tmp_defence
            attributes[l_inf_idx] = tmp_attributes

        return defence.unsqueeze(0), attributes.unsqueeze(0)

    def setFileName(self, dataset: GraphDataset, args: ArgumentParser):
        """
            information at the generic base class oneGNNSAttack
        """
        if self.singleGNN is None:
            self.file_name = fileNamer(dataset_name=dataset.name, l_0=args.l_0,
                                       num_layers=args.num_layers, seed=args.seed, targeted=args.targeted,
                                       continuous_epochs=args.continuous_epochs, start=self.start_to_file,
                                       end=self.end_to_file)
        else:
            self.file_name = fileNamer(dataset_name=dataset.name, model_name=args.singleGNN.string(),
                                       l_0=args.l_0, num_layers=args.num_layers,
                                       seed=args.seed, targeted=args.targeted, continuous_epochs=args.continuous_epochs,
                                       start=self.start_to_file, end=self.end_to_file)

    # creating
    def setLinf(self, l_inf: float):
        """
            sets the l_inf
            
            Parameters
            ----------
            l_inf: float
        """
        self.l_inf = l_inf


class NodeGNNSL0Attack(NodeGNNSAttack):
    """
        a Node-based-attack class that tests different allowed attribute ratios
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    def __init__(self, args: ArgumentParser):
        super(NodeGNNSL0Attack, self).__init__(args=args, start_to_file='NodeL0Attack',
                                               print_answer=Print.YES)

    # a must-create
    def saveResults(self, defence: torch.Tensor, attributes: torch.Tensor):
        """
            information at the generic base class oneGNNSAttack
        """
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        l_0_string = [str(l_0) for l_0 in self.l_0_list]
        header = [''] + l_0_string
        defence_df = pd.DataFrame(defence.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv('Def_' + self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def attackPerGNN(self) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for the requested attribute ratios on a specific gnn_type
        """
        if self.dataset_name.get_type() is DatasetType.CONTINUOUS:
            self.l_0_list = np.arange(0.05, 1.05, 0.05).tolist()
            return self.attackPerGNNContinuous()
        if self.dataset_name.get_type() is DatasetType.DISCRETE:
            self.l_0_list = np.arange(0.01, 1.01, 0.01).tolist()
            return self.attackPerGNNDiscrete()

    def setFileName(self, dataset: GraphDataset, args: ArgumentParser):
        """
            information at the generic base class oneGNNSAttack
        """
        if self.singleGNN is None:
            self.file_name = fileNamer(dataset_name=dataset.name, l_inf=args.l_inf, num_layers=args.num_layers,
                                       seed=args.seed, targeted=args.targeted, continuous_epochs=args.continuous_epochs,
                                       start=self.start_to_file, end=self.end_to_file)
        else:
            self.file_name = fileNamer(dataset_name=dataset.name, model_name=args.singleGNN.string(), l_inf=args.l_inf,
                                       num_layers=args.num_layers, seed=args.seed, targeted=args.targeted,
                                       continuous_epochs=args.continuous_epochs, start=self.start_to_file,
                                       end=self.end_to_file)

    # creating
    def attackPerGNNContinuous(self) -> Tuple[torch.Tensor]:
        """
            attackPerGNN for CONTINUOUS datasets
        """
        defence = torch.zeros(len(self.l_0_list)).to(self.device)
        attributes = torch.zeros(len(self.l_0_list)).to(self.device)
        for l_0_idx, l_0 in enumerate(self.l_0_list):
            self.setL0(l_0)
            tmp_defence, tmp_attributes = self.attackPerApproachWrapper(approach=NodeApproach.SINGLE)
            defence[l_0_idx] = tmp_defence
            attributes[l_0_idx] = tmp_attributes

        return defence.unsqueeze(0), attributes.unsqueeze(0)

    def attackPerGNNDiscrete(self) -> Tuple[torch.Tensor]:
        """
            attackPerGNN for DISCRETE datasets
        """
        max_attributes = self.getDataset().data.x.shape[1]
        defence = torch.zeros(len(self.l_0_list)).to(self.device)
        attributes = torch.zeros(len(self.l_0_list)).to(self.device)

        self.setL0(1.0)
        results, _, _ = attackSet(self, approach=NodeApproach.SINGLE, trainset=False)
        results = results.type(torch.FloatTensor)
        for l_0_idx, l_0 in enumerate(self.l_0_list):
            self.setL0(l_0)
            attribute_mask = (results[:, 1] <= l_0 * max_attributes)
            mask = torch.logical_and(attribute_mask, results[:, 0])

            defence[l_0_idx] = 1 - (mask.sum().type(torch.FloatTensor) / results.shape[0])
            attributes[l_0_idx] = results[mask, 1].mean(dim=0) / max_attributes

        return defence.unsqueeze(0), attributes.unsqueeze(0)

    def setL0(self, l_0: float):
        """
            sets the l_0
            
            Parameters
            ----------
            l_0: float
        """
        self.l_0 = l_0


class NodeGNNSDistanceAttack(NodeGNNSAttack):
    """
        a Node-based-attack class that tests different distances from the attacked node
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    def __init__(self, args: ArgumentParser):
        super(NodeGNNSDistanceAttack, self).__init__(args=args, start_to_file='NodeDistanceAttack',
                                                     print_answer=Print.YES)
        self.num_layers = self.max_distance

    # a must-create
    def saveResults(self, defence: torch.Tensor, attributes: torch.Tensor):
        """
            executes the requested attack for the requested distances on a specific gnn_type
        """
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        distance_string = [str(distance) for distance in range(1, self.max_distance + 1)]
        header = [''] + distance_string
        defence_df = pd.DataFrame(defence.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def attackPerGNN(self) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for the requested distance on a specific gnn_type
        """
        defence = torch.zeros(self.max_distance).to(self.device)
        attributes = torch.zeros(self.max_distance).to(self.device)
        for distance in range(1, self.max_distance + 1):
            self.setCurrentDistance(distance)
            tmp_defence, tmp_attributes = self.attackPerApproachWrapper(approach=NodeApproach.SINGLE)
            defence[distance - 1] = tmp_defence
            attributes[distance - 1] = tmp_attributes

        return defence.unsqueeze(0), attributes.unsqueeze(0)

    def checkDistanceFlag(self, args: ArgumentParser):
        """
            Validates that the distance argument is requested
            
            Parameters
            ----------
            args: ArgumentParser - command line inputs
        """
        if args.distance is None:
            exit("This attack requires the distance flag")

    # creating
    def setCurrentDistance(self, distance: int):
        """
            sets the distance
            
            Parameters
            ----------
            distance: int
        """
        self.current_distance = distance


class NodeGNNSAdversarialAttack(NodeGNNSAttack):
    """
        a Node-based-attack class that tests different train epochs (Ktrain) and different tesst epochs (Ktest)
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    def __init__(self, args: ArgumentParser):
        super(NodeGNNSAdversarialAttack, self).__init__(args, start_to_file='NodeAdversarialAttack',
                                                        print_answer=Print.NO)

    # overriding
    def setModelWrapper(self, gnn_type: GNN_TYPE):
        """
            Sets a ModelWrapper object adversarially trained, according to the requested Ktrain
            
            Parameters
            ----------
            gnn_type: GNN_TYPE - the type of the gnn
                                 more information at classes.basic_classes.GNN_TYPE
        """
        dataset = self.getDataset()
        self.model_wrapper = AdversarialModelWrapper(node_model=True, gnn_type=gnn_type, num_layers=self.num_layers,
                                                     dataset=dataset, patience=self.patience, device=self.device,
                                                     seed=self.seed)
        print(f'######################## LOADING ADVERSARIAL MODEL {self.model_wrapper.model.name} ' +
              '########################')
        self.model_wrapper.train(dataset=dataset, attack=self)

    def setIdx(self, idx: int):
        """
            sets the idx
            
            Parameters
            ----------
            idx: int
        """
        self.idx = idx


class NodeGNNSMultipleAttack(NodeGNNSAttack):
    """
        a Node-based-attack class that tests different sizes for the number of attacker nodes
        
        Parameters
        ----------
        args: ArgumentParser - command line inputs
    """
    def __init__(self, args: ArgumentParser):
        super(NodeGNNSMultipleAttack, self).__init__(args=args, start_to_file='MultipleAttack', print_answer=Print.YES)
        self.list_of_attackers = np.arange(1, 6, 1).tolist()

    # a must-create
    def saveResults(self, defence: torch.Tensor, attributes: torch.Tensor):
        """
            information at the generic base class oneGNNSAttack
        """
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        num_of_attackers_string = [str(num_of_attackers) for num_of_attackers in self.list_of_attackers]
        header = ['', 'clean'] + num_of_attackers_string
        defence_df = pd.DataFrame(defence.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def attackPerGNN(self) -> Tuple[torch.Tensor]:
        """
            executes the requested attack for the requested number of attackers on a specific gnn_type
        """
        defence = torch.zeros(len(self.list_of_attackers) + 1).to(self.device)
        attributes = torch.zeros(len(self.list_of_attackers) + 1).to(self.device)
        for attackers_idx, num_of_attackers in enumerate(self.list_of_attackers):
            self.setDefaultNumOfAttackers(num_of_attackers)
            if num_of_attackers == 1:
                tmp_defence, tmp_attributes = self.attackPerApproachWrapper(approach=NodeApproach.SINGLE)
            else:
                tmp_defence, tmp_attributes = self.attackPerApproachWrapper(approach=NodeApproach.MULTIPLE_ATTACKERS)
            defence[attackers_idx + 1] = tmp_defence
            attributes[attackers_idx + 1] = tmp_attributes

        defence[0] = self.model_wrapper.clean
        attributes[0] = 0
        return defence.unsqueeze(0), attributes.unsqueeze(0)
