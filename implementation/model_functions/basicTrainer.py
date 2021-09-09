import torch
import torch_geometric
import torch.nn.functional as F


def basicTrainer(model, optimizer: torch.optim, data: torch_geometric.data.Data, patience: int):
    """
        trains the model according to the required epochs/patience

        Parameters
        ----------
        model: Model
        optimizer: torch.optim
        data: torch_geometric.data.Data
        patience: int

        Returns
        -------
        model: Model
        model_log: str
        test_accuracy: torch.Tensor
    """
    best_val_accuracy = test_accuracy = 0
    model_train_epochs = 200
    log_template = 'Regular Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    for epoch in range(0, model_train_epochs):
        train(model, optimizer, data)
        train_accuracy, val_acc, tmp_test_acc = test(model, data)
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
    print()
    model_log = 'Basic Model - Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'\
        .format(train_accuracy, best_val_accuracy, test_accuracy)
    return model, model_log, test_accuracy


# training the current model
def train(model, optimizer: torch.optim, data: torch_geometric.data.Data):
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
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

    model.eval()


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
    logits, accuracies = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        accuracy = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(accuracy)
    return accuracies
