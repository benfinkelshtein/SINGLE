from model_functions.graph_model import ModelWrapper, AdversarialModelWrapper
from dataset_functions.graph_dataset import GraphDataset
from node_attack.attackSet import attackSet
from classes.basic_classes import Print, DatasetType
from helpers.fileNamer import fileNamer
from classes.basic_classes import GNN_TYPE
from classes.approach_classes import Approach
from edge_attack.edgeAttackSet import edgeAttackSet
from node_attack.attackSet import printAttackHeader

import torch
import numpy as np
import random
import pandas as pd


class oneGNNAttack(object):
    def __init__(self, args, start_to_file, print_answer):
        self.start_to_file = start_to_file
        self.print_answer = print_answer
        self.end_to_file = '.csv'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed = args.seed

        self.mode = args.attMode
        dataset = GraphDataset(args.dataset, device)
        self.dataset = dataset

        self.singleGNN = args.singleGNN
        if args.singleGNN is None:
            self.gnn_types = args.attMode.getGNN_TYPES()
        else:
            self.gnn_types = [args.singleGNN]

        self.num_layers = args.num_layers if args.num_layers is not None else 2
        self.patience = args.patience

        self.attack_epochs = args.attEpochs if args.attEpochs is not None else 20
        self.lr = args.lr
        if args.l_inf is None and dataset.type is DatasetType.CONTINUOUS:
            self.l_inf = 0.1
        else:
            self.l_inf = args.l_inf
        self.targeted = args.targeted

        self.max_distance = args.distance

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.device = device

        self.approaches = args.attMode.getApproaches()
        print(f'######################## STARTING ATTACK ########################')
        self.print_args(args)

        # use set functions
        self.setFileName(dataset, args)

        # *PARTLY* checking correctness of the inputs
        self.checkDistanceFlag(args)

    def checkDistanceFlag(self, args):
        if args.distance is not None:
            quit("This attack doesn't requires the distance flag")

    def setModelWrapper(self, gnn_type):
        self.model_wrapper = ModelWrapper(node_model=self.mode.isNodeModel(), gnn_type=gnn_type,
                                          num_layers=self.num_layers, dataset=self.dataset, patience=self.patience,
                                          device=self.device, seed=self.seed)
        print(f'######################## LOADING MODEL {self.model_wrapper.model.name} ########################')
        self.model_wrapper.train(self.dataset)

    def print_args(self, args):
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print()

    def attackPerGNN(self, add_clean=True):
        increment = 1 if add_clean else 0

        results = torch.zeros(len(self.approaches) + increment).to(self.device)
        for approach_idx, approach in enumerate(self.approaches):
            tmp_result = self.attackOneApproach(approach)
            results[approach_idx + increment] = tmp_result

        if add_clean:
            results[0] = self.model_wrapper.clean
        return results.unsqueeze(0)

    def setFileName(self, dataset, args):
        if self.singleGNN is None:
            self.file_name = fileNamer(dataset_name=dataset.name, l_inf=args.l_inf, num_layers=args.num_layers,
                                       seed=args.seed, targeted=args.targeted, attack_epochs=args.attEpochs,
                                       start=self.start_to_file, end=self.end_to_file)
        else:
            self.file_name = fileNamer(dataset_name=dataset.name, model_name=args.singleGNN.string(),
                                       l_inf=args.l_inf, num_layers=args.num_layers, seed=args.seed,
                                       targeted=args.targeted, attack_epochs=args.attEpochs, start=self.start_to_file,
                                       end=self.end_to_file)

    def extendLog(self, log_start, log_end):
        if self.mode.isAdversarial():
            log = 'Adv Epoch: {:02d}, '.format(self.idx) + log_start + log_end
        elif self.mode.isDistance():
            log = log_start + 'Distance: {:02d}, '.format(self.current_distance) + log_end
        else:
            log = log_start + log_end
        return log

    def setModel(self, model):
        self.model_wrapper.setModel(model)

    def saveResults(self, results):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def attackOneApproach(self, approach):
        raise NotImplementedError


class NodeGNNSAttack(oneGNNAttack):
    def __init__(self, args, start_to_file=None, print_answer=None):
        start_to_file = 'NodeAttack' if start_to_file is None else start_to_file
        print_answer = Print.YES if print_answer is None else print_answer
        super(NodeGNNSAttack, self).__init__(args=args, start_to_file=start_to_file, print_answer=print_answer)

    # a must-create
    def saveResults(self, results):
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        approaches = Approach.convertApprochesListToStringList(self.approaches)
        header = ['', 'clean'] + approaches
        defence_df = pd.DataFrame(results.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    def run(self):
        results = []
        for gnn_type in self.gnn_types:
            self.setModelWrapper(gnn_type)
            tmp_result = self.attackPerGNN()
            results.append(tmp_result)

        results = torch.cat(results).to(self.device)
        self.saveResults(results)

    def attackOneApproach(self, approach):
        results, _, _ = attackSet(self, approach=approach, print_answer=self.print_answer, trainset=False)
        return results[0]


class EdgeGNNSAttack(NodeGNNSAttack):
    def __init__(self, args):
        super(EdgeGNNSAttack, self).__init__(args=args, start_to_file='EdgeAttack', print_answer=Print.YES)

    # overriding
    def attackOneApproach(self, approach):
        if self.print_answer is Print.YES:
            print_flag = True
        else:
            print_flag = False
        return edgeAttackSet(self, approach=approach, print_flag=print_flag)


class NodeGNNSLinfAttack(NodeGNNSAttack):
    def __init__(self, args):
        super(NodeGNNSLinfAttack, self).__init__(args=args, start_to_file='NodeLinfAttack', print_answer=Print.YES)
        self.l_infs = np.arange(0.1, 1.1, 0.1).tolist()
        self.checkL_infFlag(self.dataset)

    # a must-create / overriding Node
    def run(self):
        results = []
        for gnn_type in self.gnn_types:
            self.setModelWrapper(gnn_type)
            tmp_result = self.attackLinfs()
            results.append(tmp_result)

        results = torch.cat(results).to(self.device)
        self.saveResults(results)

    def saveResults(self, results):
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        l_infs_string = [str(l_inf) for l_inf in self.l_infs]
        header = [''] + l_infs_string
        defence_df = pd.DataFrame(results.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def attackLinfs(self):
        l_inf_results = []
        for l_inf in self.l_infs:
            self.setLinf(l_inf)
            tmp_l_inf_result = self.attackPerGNN(add_clean=False)
            l_inf_results.append(tmp_l_inf_result)

        return torch.cat(l_inf_results, dim=1).to(self.device)

    def setFileName(self, dataset, args):
        if self.singleGNN is None:
            self.file_name = fileNamer(dataset_name=dataset.name, num_layers=args.num_layers, seed=args.seed,
                                       targeted=args.targeted, attack_epochs=args.attEpochs,
                                       start=self.start_to_file, end=self.end_to_file)
        else:
            self.file_name = fileNamer(dataset_name=dataset.name, model_name=args.singleGNN.string(),
                                       num_layers=args.num_layers, seed=args.seed, targeted=args.targeted,
                                       attack_epochs=args.attEpochs, start=self.start_to_file, end=self.end_to_file)

    def checkL_infFlag(self, dataset):
        if dataset.type is DatasetType.DISCRETE:
            quit("L_inf attack isn't suitable for discrete datasets")

    # creating
    def setLinf(self, l_inf):
        self.l_inf = l_inf


class NodeGNNSDistanceAttack(NodeGNNSAttack):
    def __init__(self, args):
        super(NodeGNNSDistanceAttack, self).__init__(args=args, start_to_file='NodeDistanceAttack',
                                                     print_answer=Print.YES)
        self.num_layers = self.max_distance

    # a must-create / overriding Node
    def saveResults(self, results):
        gnns = GNN_TYPE.convertGNN_TYPEListToStringList(self.gnn_types)
        distance_string = [str(distance) for distance in range(1, self.max_distance + 1)]
        header = [''] + distance_string
        defence_df = pd.DataFrame(results.to('cpu').numpy())
        defence_df.insert(0, " ", gnns)
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def attackPerGNN(self):
        results = torch.zeros(self.max_distance).to(self.device)
        for distance in range(1, self.max_distance + 1):
            self.setCurrentDistance(distance)
            tmp_result = self.attackOneApproach(self.approaches[0])
            results[distance - 1] = tmp_result
        return results.unsqueeze(0)

    def checkDistanceFlag(self, args):
        if args.distance is None:
            quit("This attack requires the distance flag")

    # creating
    def setCurrentDistance(self, distance):
        self.current_distance = distance


class NodeGNNSAdversarialAttack(NodeGNNSAttack):
    def __init__(self, args):
        super(NodeGNNSAdversarialAttack, self).__init__(args, start_to_file='NodeAdversarialAttack',
                                                        print_answer=Print.NO)
        self.Ktrain = self.attack_epochs
        self.Ktests = [1] + list(range(10, 110, 10))
        self.approach = self.approaches[0]

    # a must-create / overriding Node
    def saveResults(self, results):
        test_string = [str(k) for k in self.Ktests]
        header = ['', 'clean'] + test_string
        defence_df = pd.DataFrame(results.to('cpu').numpy())
        defence_df.insert(0, " ", str(self.Ktrain))
        defence_df.to_csv(self.file_name, float_format='%.3f', header=header, index=False, na_rep='')

    # overriding
    def run(self):
        self.setModelWrapper(gnn_type=self.gnn_types[0])
        tmp_mode = self.mode
        self.mode = self.mode.getModeNode()

        results = torch.zeros(len(self.Ktests) + 1).to(self.device)
        for idx, Ktest in enumerate(self.Ktests):
            print('######################## Attacking Adversarial Model with Ktrain: {:02d},'.format(self.Ktrain) +
                  ' Ktest: {:02d} ########################'.format(Ktest), flush=True)
            tmp_result = self.attackOneApproachAndSetAttackEpochs(approach=self.approach, Ktest=Ktest)
            results[idx + 1] = tmp_result

        results[0] = self.model_wrapper.clean
        self.mode = tmp_mode
        self.saveResults(results.unsqueeze(0))

    def setModelWrapper(self, gnn_type):
        if self.Ktrain != 0:
            self.model_wrapper = AdversarialModelWrapper(node_model=True, gnn_type=gnn_type, num_layers=self.num_layers,
                                                         dataset=self.dataset, patience=self.patience,
                                                         device=self.device, seed=self.seed)
        else:
            self.model_wrapper = ModelWrapper(node_model=True, gnn_type=gnn_type, num_layers=self.num_layers,
                                              dataset=self.dataset, patience=self.patience,
                                              device=self.device, seed=self.seed)
        printAttackHeader(attack=self, approach=self.approach)
        print('######################## Creating/Loading an Adversarial Model with Ktrain: {:02d}'.format(self.Ktrain) +
              ' ########################', flush=True)
        self.model_wrapper.train(dataset=self.dataset, attack=self)

    # creating
    def attackOneApproachAndSetAttackEpochs(self, approach, Ktest):
        self.setAttackEpochs(Ktest)
        results, _, _ = attackSet(self, approach=approach, print_answer=self.print_answer, trainset=False)
        return results[0]

    def setAttackEpochs(self, K):
        self.attack_epochs = K

    def setIdx(self, idx):
        self.idx = idx
