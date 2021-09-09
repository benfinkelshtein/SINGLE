# Single-Node Attack For Fooling Graph Neural Networks

This repository is the official implementation of Single-Node Attack For Fooling Graph Neural Networks. 

## Requirements
This project is based on PyTorch 1.6.0 and the PyTorch Geometric library.

First, install PyTorch from the official website: https://pytorch.org/.
Then install PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
(PyTorch Geometric must be installed according to the instructions there).
Eventually, run the following to verify that all dependencies are satisfied:

```setup
pip install -r requirements.txt
```

To download the Twitter dataset:

```twitter 
wget https://www.dropbox.com/s/wmlfy463dqs07hu/twitter-dataset.tar.gz
tar -xvzf twitter-dataset.tar.gz
mv twitter-dataset/data/* ./datasets/twitter
```

## Attacking

You can choose one of the 5 attacks as detailed in our paper:

1. **SINGLE NODE**
attack will produce a 2d matrix of SINGLE approaches such as (hops, GradChoice, Topology...) as a function of the available nets (GCN, GIN, GAT, SAGE, SGC, Robust GCN...)

2. **SINGLE EDGE**
attack will produce a 2d matrix of EDGE approaches such as (SINGLE, GradChoice...) as a function of the basic available nets (GCN, GIN, GAT, SAGE, SGC)

3. **NODE_LINF**
attack will produce a 2d matrix of `L_inf` values as a function of the available nets, only for the basic SINGLE approach

4. **NODE_L0**
attack will produce a 2d matrix of `L_0` values as a function of the available nets, only for the basic SINGLE approach

5. **DISTANCE**
attack will produce a 2d matrix of distance from the victim node as a function of the available nets, only for the basic SINGLE approach

6. **ADVERSARIAL**
attack will produce a 2d matrix of SINGLE approaches such as (hops, GradChoice, Topology...) as a function of the available nets (GCN, GIN, GAT, SAGE, SGC...), for a model which is trained adversarialy on the basic SINGLE approach

7. **MULTIPLE**
attack will produce a 2d matrix of the number of attackers as a function of the available nets, only for the basic SINGLE approach


The available input arguments are:

* `--attMode`: Name of the attack Mode as described above

* `--dataset`: Name of the dataset, all caps

* `--singleGNN`: name of the wanted GNN (only in the case that you want results for ONE GNN)

* `--num_layers`: number of layers in the GNN

* `--patience`: the patience of the basic training (not the adversarial training)

* `--attEpochs`: number of attack epochs per victim node / number of `Ktrain`

* `--lr`: the learning rate

* `--l_inf`: `L_inf` value, only for datasets that are represented in tf-idf (and not in many-hot-vec)

* `--l_0`: The limit on the ratio of attributes used in the attack

* `--targeted`: a bool flag that changes the attack to a targeted attack

* `--distance` (ONLY FOR THE DISTANCE ATTACK): the maximum distance

* `--seed`: a seed for reproducability

Note: Every combination of attack mode and GNN is available, except for the combination of Edge attacks+Robust GNNs

## Citation
If you want to cite this work, please use this bibtex entry:
```
@article{finkelshtein2020single,
  title={Single-Node Attack for Fooling Graph Neural Networks},
  author={Finkelshtein, Ben and Baskin, Chaim and Zheltonovzhskii, Evgenii and Alon, Uri},
  journal={arXiv preprint arXiv:2011.03574},
  year={2020}
}
```
