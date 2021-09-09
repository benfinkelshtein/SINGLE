from classes.basic_classes import GNN_TYPE, DataSet
from classes.attack_class import AttackMode

from argparse import ArgumentParser
from torch.cuda import set_device


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--attMode", dest="attMode", default=AttackMode.NODE, type=AttackMode.from_string,
                        choices=list(AttackMode), required=False)
    parser.add_argument("--dataset", dest="dataset", default=DataSet.PUBMED, type=DataSet.from_string,
                        choices=list(DataSet), required=False)

    parser.add_argument('--singleGNN', dest="singleGNN", type=GNN_TYPE.from_string, choices=list(GNN_TYPE),
                        required=False)

    parser.add_argument("--num_layers", dest="num_layers", default=2, type=int, required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)

    parser.add_argument("--continuous_epochs", dest="continuous_epochs", default=20, type=int, required=False)
    parser.add_argument("--lr", dest="lr", type=float, default=0.1, required=False)

    parser.add_argument("--l_inf", dest="l_inf", type=float, default=None, required=False)
    parser.add_argument("--l_0", dest="l_0", type=float, default=None, required=False)
    parser.add_argument('--targeted', dest="targeted", action='store_true', required=False)

    parser.add_argument("--distance", dest='distance', type=int, required=False)

    parser.add_argument("--seed", dest="seed", type=int, default=0, required=False)

    parser.add_argument('--gpu', type=int, required=False)

    args = parser.parse_args()
    if args.gpu is not None:
        set_device(args.gpu)

    attack = args.attMode.getAttack()
    attack(args).run()
