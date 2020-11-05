# Single-Node Attack For Fooling Graph Neural Networks

This repository is the official implementation of Single-Node Attack For Fooling Graph Neural Networks. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download the Twitter dataset:

```twitter 
wget https://www.dropbox.com/s/wmlfy463dqs07hu/twitter-dataset.tar.gz
tar -xvzf twitter-dataset.tar.gz
mv twitter-dataset/data/* GNNattack/datasets/twitter
```

## Attacking

You can choose one of the 5 attacks as detailed in our paper:

1. SINGLE (AKA, NODE)
SINGLE attack will produce a 2d matrix of SINGLE approaches such as (hops, GradChoice, Topology...) as a function of the available nets (GCN, GIN, GAT, SAGE)

2. EDGE
EDGE attack will produce a 2d matrix of EDGE approaches such as (EdgeGrad, GlobalEdgeGrad, RANDOM) as a function of the available nets (GCN, GIN, GAT, SAGE)

3. NODE_LINF
NODE_LINF attack will produce a 2d matrix of L_inf values {0.1-1.1} as a function of the available nets (GCN, GIN, GAT, SAGE), only for the basic SINGLE approach

4. DISTANCE
DISTANCE attack will produce a 2d matrix of distance from the victim node as a function of the available nets (GCN, GIN, GAT, SAGE), only for the basic SINGLE approach

5. ADVERSARIAL
ADVERSARIAL attack will produce a 1d vector of Ktest results for Ktrain=attEpochs, only for the basic SINGLE approach and for the GCN net


The available input arguments are:

* --attMode: Name of the attack Mode as described above

* --dataset: Name of the dataset, all caps

* --singleGNN: name of the wanted GNN (only in the case that you want results for ONE GNN)

* --num_layers: number of layers in the GNN

* --patience: the patience of the basic training (not the adversarial training)

* --attEpochs: number of attack epochs per victim node / number of Ktrain

* --lr: learn rate

* --l_inf: L_inf value, only for datasets that are represented in tf/idf (and not in many-hot-vec)

* --targeted: a bool flag that changes the attack to a targeted attack

* --distance (ONLY FOR THE DISTANCE ATTACK): the maximum distance

* --seed: a seed for reproducability
