from classes.basic_classes import Print, DatasetType

import torch
import torch.nn.functional as F
import copy


# a trainer function that attacks our model by changing the input attribute in 2 steps
# step 1 - attack the model with all attributes
# for continuous datasets stop when:
# the model attack successfully AND
# embed the attributes
# attack successfully again with embedded attributes
# for discrete datasets stop when the model attack successfully
# step 2 - attack the model with minimal attributes - go to findMinimalAttributes for more information
def attackTrainer(attack, approach, model, print_answer, attacked_nodes, y_targets, malicious_nodes, node_num,
                  attack_epochs, lr):
    dataset = attack.dataset
    data = dataset.data

    log_template = createLogTemplate(attack=attack, first_step=True)
    # changing the parameters which require grads
    optimizer_params, malicious_row_list = setRequiresGrad(model=model, malicious_nodes=malicious_nodes)

    # adversarial optimizer
    optimizer = torch.optim.Adam(params=optimizer_params, lr=lr)

    # setting up discrete gradient flag
    multiple_malicious_nodes = False
    if approach.isMultipleMaliciousNodes() and malicious_nodes.shape[0] > 1:
        multiple_malicious_nodes = True

    # step 1 - attack the model with all attributes
    model0 = copy.deepcopy(model)
    x0 = model0.getInput().clone().detach()
    for epoch in range(0, attack_epochs):
        train(model=model, targeted=attack.targeted, attacked_nodes=attacked_nodes, y_targets=y_targets,
              optimizer=optimizer)
        results = test(data=data, model=model, targeted=attack.targeted, attacked_nodes=attacked_nodes,
                       y_targets=y_targets)

        # print each epoch
        if print_answer is Print.YES:
            print(log_template.format(node_num, epoch + 1, *results[:-1]), flush=True, end='')
        if results[3]:
            embeded_model = copy.deepcopy(model)
            if dataset.type is DatasetType.CONTINUOUS:
                embedRow(model=embeded_model, malicious_nodes=malicious_nodes, x0=x0, l_inf=attack.l_inf)
                results = test(data=data, model=embeded_model, targeted=attack.targeted,
                               attacked_nodes=attacked_nodes, y_targets=y_targets)
                if results[3]:
                    break
            if dataset.type == DatasetType.DISCRETE:
                break
        if epoch != attack_epochs - 1 and print_answer is not Print.NO:
            print()

    # print when done with the first step
    if print_answer is Print.YES:
        end_log_template = ', First Attack Success: {}'
        print(end_log_template.format(results[-1]), flush=True)
    attack_results = torch.tensor([[results[3]]])

    # step 2 - attack the model with minimal attributes
    if attack_results[0][0] and not multiple_malicious_nodes and not dataset.skip_attributes:
        if print_answer is Print.YES:
            print('Limiting Attributes:')
        attack_results = \
            findMinimalAttributes(attack=attack, embeded_model=embeded_model, print_answer=print_answer,
                                  node_num=node_num, attacked_nodes=attacked_nodes, y_targets=y_targets,
                                  malicious_nodes=malicious_nodes, model0=model0)
    else:
        if print_answer is Print.YES:
            print()
        attack_results = torch.cat((attack_results, torch.tensor([[data.x.shape[1]]])), dim=1)
    return attack_results.type(torch.FloatTensor)


# a helper function which creates a log to print
def createLogTemplate(attack, first_step):
    log_start = 'Attack: {:03d}, '
    if first_step:
        log_end = 'Epoch'
    else:
        log_end = '#Att'
    log_end += ': {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    log = attack.extendLog(log_start, log_end)
    return log


# a helper which turns off the grad for the net layers and turns on the grad for the malicious nodes attributes
def setRequiresGrad(model, malicious_nodes):
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

    optimizer_params = [dict(params=malicious_row_list)]
    return optimizer_params, malicious_row_list


# a function which trains the model with the attacked node loss
def train(model, targeted, attacked_nodes, y_targets, optimizer):
    model.train()
    optimizer.zero_grad()

    attacked_nodes = [attacked_nodes.item()]
    model_output = model()[attacked_nodes]

    loss = F.nll_loss(model_output, y_targets)
    loss = loss if targeted else -loss
    loss.backward()

    optimizer.step()


# a function which test the model with all masks and tests the attack
# returns the accuracies of the test AKA attack_results
@torch.no_grad()
def test(data, model, targeted, attacked_nodes, y_targets):
    model.eval()
    logits, accuracies = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accuracies.append(acc)

    pred = logits[attacked_nodes].max(1)[1]

    if y_targets.shape[0] == 1:
        y_targets_acc = (pred == y_targets)
        if not targeted:
            y_targets_acc = torch.logical_not(y_targets_acc)
    else:
        y_targets_acc = torch.sum(pred == y_targets).type(torch.FloatTensor) / y_targets.shape[0]
        if not targeted:
            y_targets_acc = 1 - y_targets_acc
    y_targets_acc = y_targets_acc.item()
    accuracies.append(y_targets_acc)
    return accuracies


# embed the attribute row and limits the inf norm, only for continuous datasets
def embedRow(model, malicious_nodes, x0, l_inf):
    for malicious_node in malicious_nodes:
        row0 = x0[malicious_node]
        row = model.node_attribute_list[malicious_node][0]

        upper_bound = row0 + l_inf
        lower_bound = row0 - l_inf

        row[row > upper_bound] = upper_bound[row > upper_bound]
        row[row < lower_bound] = lower_bound[row < lower_bound]
        row[row < 0] = 0

        model.setNodesAttributes(idx_node=malicious_node, values=row)


# a function which finds the minimal embeded attributes for a successful attack
# sorts the attributes in a descending order by the side of their distance (from their starting position before step 1)
# iteratively - adds an attribute and test whether we attack successfully (now with our newly added attribute)
def findMinimalAttributes(attack, embeded_model, print_answer, node_num, attacked_nodes, y_targets, malicious_nodes,
                          model0):
    dataset = attack.dataset
    data = dataset.data
    x0 = model0.getInput().clone().detach()

    semi_embedded_model = copy.deepcopy(model0)
    attribute_first_step_change = torch.abs((embeded_model.getInput() - x0)[malicious_nodes])
    _, sorted_attributes_indices = torch.sort(attribute_first_step_change, dim=1, descending=True)
    sorted_attributes_indices = sorted_attributes_indices[0]

    for malicious_node in malicious_nodes:
        semi_embedded_row = x0[malicious_node]

        for attribute_num, attribute_index in enumerate(sorted_attributes_indices):
            if dataset.type is DatasetType.CONTINUOUS:
                semi_embedded_row_attribute_value = embeded_model.getInput()[malicious_node][attribute_index]
            if dataset.type is DatasetType.DISCRETE:
                semi_embedded_row_attribute_value = 1 - semi_embedded_row[attribute_index]

            semi_embedded_model.setNodesAttribute(idx_node=malicious_node, idx_attribute=attribute_index,
                                                  value=semi_embedded_row_attribute_value)
            results = test(data=data, model=semi_embedded_model, targeted=attack.targeted,
                           attacked_nodes=attacked_nodes, y_targets=y_targets)

            if results[3]:
                tmp_l_inf = 1.0 if dataset.type is DatasetType.DISCRETE else attack.l_inf
                try:
                    test_validate_linf(x0=x0, new_x=semi_embedded_model.getInput(),
                                       malicious_node_num=malicious_nodes.shape[0], l_inf=tmp_l_inf)
                except AssertionError:
                    x_diff = x0 - semi_embedded_model.getInput()
                    indices_of_changed_nodes = x_diff.sum(dim=-1).nonzero(as_tuple=True)
                    print(f"Assert failed, #Nodes Changed:{len(indices_of_changed_nodes)},"
                          f"Biggest Absolute Change:{x_diff.abs().max().item()}")
                break

    # print when done with the second step
    if print_answer is Print.YES:
        embed_log_template = createLogTemplate(attack=attack, first_step=False)
        end_embed_log_template = embed_log_template + ', Second Attack Success: {}\n'
        print(end_embed_log_template.format(node_num, attribute_num + 1, *results), flush=True)
    attack_results = torch.tensor([[results[3], attribute_num + 1]])
    return attack_results


# a test function that checks the number of changed nodes and the limit on the inf norm
@torch.no_grad()
def test_validate_linf(x0, new_x, malicious_node_num, l_inf):
    x_diff = x0 - new_x
    indices_of_changed_nodes = x_diff.sum(dim=-1).nonzero(as_tuple=True)
    assert len(indices_of_changed_nodes) <= malicious_node_num
    assert len(indices_of_changed_nodes) >= 1
    max_changed_attribute = x_diff.abs().max()
    assert max_changed_attribute.item() <= l_inf + 1e-2
