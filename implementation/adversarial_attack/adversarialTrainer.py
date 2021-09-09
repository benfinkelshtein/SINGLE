from node_attack.attackSet import attackSet
from node_attack.attackTrainerHelpers import test
from classes.basic_classes import Print
from classes.approach_classes import Approach, NodeApproach

import torch
import torch_geometric
import torch.nn.functional as F
import copy
from typing import Tuple


# a trainer function for our adversarial model
# the function find harmful attributes
# and learns to classify them correctly
def adversarialTrainer(attack):
    """
        trains the model adversarial (the model learns to classify correctly harmful feature matrices)
        
        Parameters
        ----------
        attack: oneGNNAttack
        
        Returns
        -------
        model: Model
        model_log: str
        test_accuracy: torch.Tensor
    """

    model = attack.model_wrapper.model  # important note: this is a fresh, untrained model!
    data = attack.getDataset().data

    patience_counter, best_val_accuracy = 0, 0
    adversarial_model_train_epochs = 200
    log_template = 'Adversarial Model - Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Attack: {:.4f}'

    model.attack = True
    # train in an adversarial way
    for epoch in range(0, adversarial_model_train_epochs):
        tmp_attack = copy.deepcopy(attack)
        tmp_attack.setIdx(epoch + 1)
        attacked_x, attacked_nodes, y_targets = \
            getTheMostHarmfulInput(attack=tmp_attack, approach=NodeApproach.TOPOLOGY)

        train(model=attack.model_wrapper.model, optimizer=attack.model_wrapper.optimizer, data=data,
              attacked_nodes=attacked_nodes, attacked_x=attacked_x)
        train_results = test(data=data, model=attack.model_wrapper.model, targeted=attack.targeted,
                             attacked_nodes=attacked_nodes, y_targets=y_targets)
        print(log_template.format(epoch + 1, *train_results))

        # patience
        val_acc = train_results[1]
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= attack.patience:
            break

    attack.model_wrapper.model.attack = False
    print()
    model_log = 'Adversarial Model - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Attack: {:.4f}'\
        .format(*train_results)
    return attack.model_wrapper.model, model_log, train_results[2]


def getTheMostHarmfulInput(attack, approach: Approach) -> Tuple[torch.Tensor]:
    """
        attacks the model and extract the attacked feature matrix
        
        Parameters
        ----------
        attack: oneGNNAttack
        approach: torch_geometric.data.Data
        
        Returns
        -------
        attacked_nodes: torch.Tensor - the victim nodes
        attacked_x: torch.Tensor - the feature matrices after the attack
        y_targets: torch.Tensor - the target labels of the attack
    """
    attack.print_answer = Print.NO
    _, attacked_nodes, y_targets = attackSet(attack=attack, approach=approach, trainset=True)
    attacked_x = attack.model_wrapper.model.getInput().clone().detach()
    return attacked_x, attacked_nodes, y_targets


def train(model, optimizer: torch.optim, data: torch_geometric.data.Data, attacked_nodes: torch.Tensor,
          attacked_x: torch.Tensor, adv_scale: int = 1):
    """
        trains the model with both losses - clean and adversarial, for one epoch
        
        Parameters
        ----------
        model: Model
        optimizer: torch.optim
        data: torch_geometric.data.Data
        attacked_nodes: torch.Tensor - the victim nodes
        attacked_x: torch.Tensor - the feature matrices after the attack
        adv_scale: int - the lambda scale hyperparameter between the two losses
    """
    model.train()
    optimizer.zero_grad()

    basic_loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    adv_loss = F.nll_loss(model(attacked_x)[attacked_nodes], data.y[attacked_nodes])
    loss = basic_loss + adv_scale * adv_loss

    loss.backward()
    optimizer.step()

    model.eval()
