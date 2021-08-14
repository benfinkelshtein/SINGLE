from classes.basic_classes import DatasetType

from typing import List, Dict
import torch
import torch.nn.functional as F
import torch_geometric


def createLogTemplate(attack, dataset):
    """
        a helper function which creates a log to print based on the attack

        Returns
        -------
        log: str
    """
    log_start = 'Attack: {:03d}'
    log_end = ', Epoch {:03d}'
    if dataset.type is DatasetType.DISCRETE:
        log_end += ', #Att: {:03d}'
    log_end += ', Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    log = attack.extendLog(log_start, log_end)
    return log


def setRequiresGrad(model, malicious_nodes: torch.Tensor) -> List[Dict]:
    """
        a helper which turns off the grad for the net layers and
        turns on the grad for the malicious nodes attributes

        Parameters
        -------
        model: Model
        malicious_nodes: torch.Tensor

        Returns
        -------
        optimization_params: List[Dict]
    """
    # zeroing requires grad
    for layer in model.layers:
        for p in layer.parameters():
            p.detach()
            p.requires_grad = False
    for row in model.node_attribute_list:
        row.detach()
        row.requires_grad = False

    # specifying adversarial parameters and constructing list
    if malicious_nodes.shape[0] == 1:
        malicious_node_indexes = [malicious_nodes.item()]
    else:
        malicious_node_indexes = malicious_nodes.tolist()
    malicious_row_list = [model.node_attribute_list[idx] for idx in malicious_node_indexes]
    for row in malicious_row_list:
        row.requires_grad = True

    return [dict(params=malicious_row_list)]


def train(model, targeted: bool, attacked_nodes: torch.Tensor, y_targets: torch.Tensor, optimizer: torch.optim):
    """
        trains the attack for one epoch

        Parameters
        -------
        model: Model
        targeted: bool
        attacked_nodes: torch.Tensor
        y_targets: torch.Tensor - the target labels of the attack
        optimizer: torch.optim
    """
    model.train()
    optimizer.zero_grad()

    attacked_nodes = [attacked_nodes.item()]
    model_output = model()[attacked_nodes]

    if torch.sum(model_output - model_output[:y_targets.shape[0], y_targets]) == 0:
        model.eval()
        model_output = model()[attacked_nodes]

    loss = F.nll_loss(model_output, y_targets)
    loss = loss if targeted else -loss
    loss.backward()

    optimizer.step()

    model.eval()


# a function which test the model with all masks and tests the attack
# returns the accuracies of the test AKA attack_results
@torch.no_grad()
def test(data: torch_geometric.data.Data, model, targeted: bool, attacked_nodes: torch.Tensor,
         y_targets: torch.Tensor) -> torch.Tensor:
    """
        tests the model according to the train/val/test masks and attack mask

        Parameters
        ----------
        data: torch_geometric.data.Data
        model: Model
        targeted: bool
        attacked_nodes: torch.Tensor
        y_targets: torch.Tensor - the target labels of the attack

        Returns
        -------
        accuracies: : torch.Tensor - train, val, test, misclassified/attack success (True, False)
    """
    model.eval()
    logits, accuracies = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(acc)

    model_res = logits[attacked_nodes]

    # edge case where a model in train mode is mistaken
    model.train()
    train_model_res = model()[attacked_nodes]
    y_targets_acc = model_res2targets_acc(targeted=targeted, y_targets=y_targets, model_res=train_model_res)
    model.eval()
    if y_targets_acc:
        accuracies.append(y_targets_acc)
        return accuracies

    y_targets_acc = model_res2targets_acc(targeted=targeted, y_targets=y_targets, model_res=model_res)
    accuracies.append(y_targets_acc)
    return accuracies


@torch.no_grad()
def model_res2targets_acc(targeted: bool, y_targets: torch.Tensor, model_res: torch.Tensor) -> float:
    """
        converts the probabilities of the attacked node to a bool of attack success/fail

        Parameters
        ----------
        targeted: bool
        y_targets: torch.Tensor - the target labels of the attack
        model_res: torch.Tensor - model result for the attacked node

        Returns
        -------
        y_targets_acc: : float - attack success (True, False)
    """
    pred_val, pred = model_res.max(1)

    # edge case where more than one of the classes has the same prob
    diff_prob_mat = (model_res.T - pred_val).T
    same_prob_vec = (diff_prob_mat == 0).sum(1)
    edge_case_vec = torch.logical_and(same_prob_vec > 1,
                                      diff_prob_mat[range(0, y_targets.shape[0]), y_targets] == 0)
    if targeted:
        pred[edge_case_vec] = y_targets[edge_case_vec]
    else:
        for node_idx in range(model_res.shape[0]):
            for class_idx in range(model_res.shape[1]):
                if model_res[node_idx, class_idx] == pred_val[node_idx] and class_idx != y_targets[node_idx]:
                    pred[node_idx] = class_idx
    # end of edge case

    if y_targets.shape[0] == 1:
        y_targets_acc = (pred == y_targets)
        if not targeted:
            y_targets_acc = torch.logical_not(y_targets_acc)
    else:
        y_targets_acc = torch.sum(pred == y_targets).type(torch.FloatTensor) / y_targets.shape[0]
        if not targeted:
            y_targets_acc = 1 - y_targets_acc
    y_targets_acc = y_targets_acc.item()
    return y_targets_acc


@torch.no_grad()
def flipUpBestNewAttributes(model, model0, malicious_nodes: torch.Tensor, num_attributes_left: torch.Tensor)\
        -> torch.Tensor:
    """
        tests the model according to the train/val/test masks and attack mask

        Parameters
        ----------
        model: Model - post-attack model
        model0: Model - pre-attack model
        malicious_nodes: torch.Tensor - the attacker/malicious node
        num_attributes_left: torch.Tensor -  a torch tensor vector with ones where the attribute is not flipped

        Returns
        -------
        num_attributes_left: torch.Tensor -  a torch tensor vector with ones where the attribute is not flipped
    """
    for malicious_idx, malicious_node in enumerate(malicious_nodes):
        row = model.node_attribute_list[malicious_node][0]
        row0 = model0.node_attribute_list[malicious_node][0]

        # exclude attributes which are already used and attributes with negative gradient
        zero_mask = torch.logical_or(row < row0, row0 == 1)
        diff = row - row0
        diff[zero_mask] = 0

        # find best gradient indexes
        row = row0.clone().detach()
        max_diff = diff.max()
        flip_indexes = (diff == max_diff).nonzero(as_tuple=True)[0]

        # check if attribute limit exceeds
        if num_attributes_left[malicious_idx] < flip_indexes.shape[0]:
            flip_indexes = flip_indexes[:num_attributes_left[malicious_idx]]

        # flip
        if max_diff != 0:
            row[flip_indexes] = 1
            num_attributes_left[malicious_idx] -= flip_indexes.shape[0]

        # save flipped gradients
        model.setNodesAttributes(idx_node=malicious_node, values=row)
    return num_attributes_left


# embed the attribute row and limits the inf norm, only for continuous datasets
@torch.no_grad()
def embedRowContinuous(model, malicious_node, model0, l_inf, l_0):
    row0 = model0.node_attribute_list[malicious_node][0]
    row = model.node_attribute_list[malicious_node][0]
    final_row = row0.clone()
    k = int(l_0 * row0.shape[0])

    # limiting number of attributes
    row[row < 0] = 0
    abs_diff = (row - row0).abs()
    largest_diff_indices = torch.topk(abs_diff, k=k)[1]
    final_row[largest_diff_indices] = row[largest_diff_indices]

    # limiting the amplitude of attributes
    upper_bound = row0 + l_inf
    lower_bound = row0 - l_inf

    final_row[final_row > upper_bound] = upper_bound[final_row > upper_bound]
    final_row[final_row < lower_bound] = lower_bound[final_row < lower_bound]
    final_row[final_row < 0] = 0

    model.setNodesAttributes(idx_node=malicious_node, values=final_row)
