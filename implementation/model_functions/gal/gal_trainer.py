"""
Implementation of the paper:
'Information Obfuscation of Graph Neural Networks'
by Peiyuan Liao, Han Zhao, Keyulu Xu, Tommi Jaakkola, Geoffrey Gordon, Stefanie Jegelka, Ruslan Salakhutdinov
Copyright (C) 2021
"""

import torch
import torch_geometric
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, train_test_split_edges)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score


def create_gal_optimizer(model, lr=0.01, lambda_reg=0.5):
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=0),
        dict(params=model.conv2.parameters(), weight_decay=0),
        dict(params=model.attr.parameters(), weight_decay=0)
    ], lr=lr)

    optimizer_attack = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=5e-4),
        dict(params=model.attk.parameters(), weight_decay=5e-4),
    ], lr=lr * lambda_reg)

    optimizer_fine_tune = torch.optim.Adam([
        dict(params=model.attr.parameters(), weight_decay=5e-4),
    ], lr=lr)

    return optimizer, optimizer_attack, optimizer_fine_tune


def galTrainer(model, data: torch_geometric.data.Data):
    """
        trains the model according to the required epochs/patience

        Parameters
        ----------
        model: Model
        data: torch_geometric.data.Data

        Returns
        -------
        model: Model
        model_log: str
        test_accuracy: torch.Tensor
    """

    # according to best results reported in GAL paper
    if model.dataset_name == "CITESEER":
        lambda_param = 0.75
        use_ws_loss = False
    elif model.dataset_name == "CORA":  # default param - nor reported in the paper
        lambda_param = 0.05
        use_ws_loss = True
    elif model.dataset_name == "PUBMED":
        lambda_param = 0.5
        use_ws_loss = False
    else:
        lambda_param = 0.05
        use_ws_loss = True

    # Train/validation/test
    data = train_test_split_edges(data)
    optimizer, optimizer_attack, optimizer_fine_tune = create_gal_optimizer(model=model, lambda_reg=lambda_param)

    train_epochs = 250
    fine_tune_epochs = 800

    switch = True
    for epoch in range(1, train_epochs + 1):

        train_acc = train(model=model, optimizer=optimizer, optimizer_attack=optimizer_attack,
                         data=data, switch=switch, use_ws_loss=use_ws_loss)
        switch = not switch
        # start of changes XXXXX
        log_template = 'Regular Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        val_acc, test_acc = test(model, data)
        print(log_template.format(epoch, train_acc, val_acc, test_acc), flush=True)

    print(flush=True)
    # end of changes XXXXX

    best_val_acc = test_acc = 0
    for epoch in range(1, fine_tune_epochs + 1):
        train_attr(model=model, optimizer_attr=optimizer_fine_tune, data=data)
        train_acc, val_acc, tmp_test_acc = test_attr(model=model, data=data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Finetune Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, tmp_test_acc))
    print(flush=True)
    model_log = 'Basic Model - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}' \
        .format(train_acc, best_val_acc, test_acc)
    return model, model_log, test_accuracy


def _get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float()
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


# training the current model
def train(model, optimizer: torch.optim, optimizer_attack: torch.optim, data: torch_geometric.data.Data,
          switch: bool = True, use_ws_loss: bool = True):
    """
        trains the model for one epoch

        Parameters
        ----------
        model: Model
        optimizer: torch.optim
        optimizer_attack: torch.optim
        data: torch_geometric.data.Data
        switch: bool
        use_ws_loss: bool
    """

    model.train()

    labels = data.y.to(model.device)
    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=pos_edge_index.size(1))

    link_logits, attr_prediction, attack_prediction, _ = model(pos_edge_index, neg_edge_index)
    link_labels = _get_link_labels(pos_edge_index, neg_edge_index).to(link_logits.device)

    # same from here to the end
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

    # loss 2
    if use_ws_loss:  # wasserstein distance VS total variation
        one_hot = torch.cuda.FloatTensor(attack_prediction.size(0), attack_prediction.size(1)).zero_()
        mask = one_hot.scatter_(1, labels.view(-1, 1), 1)

        nonzero = mask * attack_prediction
        avg = torch.mean(nonzero, dim=0)
        loss2 = torch.abs(torch.max(avg) - torch.min(avg))
    else:
        loss2 = F.nll_loss(attack_prediction, labels)

    link_logits = link_logits.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()

    train_acc = roc_auc_score(link_labels, link_logits)

    if switch:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        optimizer_attack.zero_grad()
        loss2.backward()
        optimizer_attack.step()

        for p in model.attk.parameters():
            p.data.clamp_(-1, 1)

    model.eval()
    return train_acc


# testing the current model
@torch.no_grad()
def test(model, data: torch_geometric.data.Data) -> torch.Tensor:
    """
        tests the model according to the train/val/test masks

        Parameters
        ----------
        model: Model
        data: torch_geometric.data.Data

        Returns
        -------
        accuracies: torch.Tensor - 3d-tensor that includes
                                    1st-d - the train accuracy
                                    2nd-d - the val accuracy
                                    3rd-d - the test accuracy
    """
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        neg_edge_index = neg_edge_index.to(pos_edge_index.device)
        link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index)[0])
        link_labels = _get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs


def train_attr(model, optimizer_attr: torch.optim, data: torch_geometric.data.Data):
    model.train()
    optimizer_attr.zero_grad()

    labels = data.y.to(model.device)
    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=pos_edge_index.size(1))

    F.nll_loss(model(pos_edge_index, neg_edge_index)[1][data.train_mask], labels[data.train_mask]).backward()
    optimizer_attr.step()
    model.eval()


@torch.no_grad()
def test_attr(model, data):
    model.eval()
    accs = []
    m = ['train_mask', 'val_mask', 'test_mask']
    i = 0
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):

        if m[i] == 'train_mask':
            x, pos_edge_index = data.x, data.train_pos_edge_index

            _edge_index, _ = remove_self_loops(pos_edge_index)
            pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                               num_nodes=x.size(0))

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                num_neg_samples=pos_edge_index.size(1))
        else:
            pos_edge_index, neg_edge_index = [
                index for _, index in data("{}_pos_edge_index".format(m[i].split("_")[0]),
                                           "{}_neg_edge_index".format(m[i].split("_")[0]))
            ]
        neg_edge_index = neg_edge_index.to(pos_edge_index.device)
        _, logits, _, _ = model(pos_edge_index, neg_edge_index)

        pred = logits[mask].max(1)[1]

        macro = f1_score((data.y[mask]).cpu().numpy(), pred.cpu().numpy(), average='macro')
        accs.append(macro)

        i += 1
    return accs
