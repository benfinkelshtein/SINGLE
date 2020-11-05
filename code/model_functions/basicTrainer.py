import torch
import torch.nn.functional as F


# a trainer for our basic (non-adversarial, not yet attacked) models
def basicTrainer(model, optimizer, data, patience):
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
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


# testing the current model
@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accuracies = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        accuracy = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(accuracy)
    return accuracies
