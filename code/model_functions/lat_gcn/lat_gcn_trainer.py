import torch
import torch_geometric

import torch.nn.functional as F

def constructOptimizer(model):
    """
        sets an optimizer for the Model object
    """
    for layer in model.layers:
        list_dict_param += [dict(params=layer.parameters(), weight_decay=5e-4)]
    optimizer = torch.optim.Adam(list_dict_param, lr=0.01)
    return optimizer


def latgcnTrainer(model, optimizer: torch.optim, data: torch_geometric.data.Data, patience: int,
     gamma=0.1, epsilon=0.1):
    """
        trains the model according to the required epochs/patience

        Parameters
        ----------
        model: Model
        data: torch_geometric.data.Data
        patience:

        Returns
        -------
        model: Model
        model_log: str
        test_accuracy: torch.Tensor
    """
    perturbation_epochs = 20
    train_epochs = 200
    # train_epochs = 12

    patience = 30

    patience_counter = 0
    best_val_accuracy = test_accuracy = 0
    for epoch in range(1, train_epochs+1):

        best_perturbation = train_perturbation(model, data, epsilon,
            perturbation_epochs, patience).detach()

        train(model, optimizer, data, best_perturbation, gamma)

        train_accuracy, val_acc, tmp_test_acc = test(model, data, best_perturbation)

        # Our logging
        log_template = 'Regular Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            test_accuracy = tmp_test_acc
            # patience
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
        print(log_template.format(epoch, train_accuracy, best_val_accuracy, test_accuracy), flush=True)

    model_log = 'Basic Model - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'\
        .format(train_accuracy, best_val_accuracy, test_accuracy)

    return model, model_log, test_accuracy


def cut_perturbation(perturbation, epsilon):
    with torch.no_grad():
        row_norm = torch.norm(perturbation, dim=1, p=2)
        bad_rows = row_norm > epsilon

        corrected_rows = epsilon * perturbation / row_norm[:, None]

        perturbation[bad_rows, :] = corrected_rows[bad_rows, :]
    return perturbation

def train_perturbation(model, data, epsilon, epochs, patience):
    model.eval()

    perturbation = torch.rand(model.get_perturbation_shape(),
        requires_grad=True, device=model.device) # in [0,1]
    perturbation = 2 * (perturbation - 0.5) # in [-1,1]
    perturbation = epsilon * perturbation # in [-eps, eps]
    perturbation = cut_perturbation(perturbation, epsilon).detach().requires_grad_()

    perturbation = perturbation.detach().requires_grad_()

    optimizer = torch.optim.Adam([perturbation], lr=0.01)

    patience_counter = 0
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        _, R = model.forward(perturbation=perturbation, grad_perturbation=True)

        loss = R

        loss = -1 * loss
        loss.backward()

        optimizer.step()

        perturbation = cut_perturbation(perturbation, epsilon).detach().requires_grad_()


        if loss < best_loss:
            best_loss = loss
            # patience
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    print("\tPerturbation loss: {} (best loss: {})".format(loss, best_loss), flush=True)
    model.train()
    return perturbation





# training the current model
def train(model, optimizer: torch.optim, data: torch_geometric.data.Data, perturbation, gamma):
    """
        trains the model for one epoch

        Parameters
        ----------
        model: Model
        optimizer: torch.optim
        data: torch_geometric.data.Data
    """

    model.train()
    optimizer.zero_grad()

    y_hat, R = model.forward(perturbation=perturbation, grad_perturbation=False)

    # look here - what do we do. accuracy goes up, then down"
    y_hat = y_hat[model.data.train_mask]
    loss = F.nll_loss(y_hat, data.y[model.data.train_mask]) + gamma * R
    # loss = F.nll_loss(y_hat, data.y) + gamma * R

    loss.backward()

    optimizer.step()

    model.eval()

# testing the current model
@torch.no_grad()
def test(model, data: torch_geometric.data.Data, perturbation) -> torch.Tensor:
    model.eval()
    accuracies = []
    logits = model.forward(perturbation=None, grad_perturbation=False)
    # logits, _ = model.forward(perturbation=perturbation, grad_perturbation=False)

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]

        accuracy = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(accuracy)
    return accuracies