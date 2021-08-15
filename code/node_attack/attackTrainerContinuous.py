from node_attack.attackTrainerHelpers import (createLogTemplate, setRequiresGrad, train, test, embedRowContinuous)
from classes.basic_classes import Print
from node_attack.attackTrainerTests import test_discrete, test_continuous

import torch
import copy


def attackTrainerContinuous(attack, attacked_nodes: torch.Tensor, y_targets: torch.Tensor,
                            malicious_nodes: torch.Tensor, node_num: int) -> torch.Tensor:
    """
        a trainer function that attacks our model by changing the input attributes
        a successful attack is when we attack successfully AND embed the attributes

        Parameters
        ----------
        attack: oneGNNAttack
        attacked_nodes: torch.Tensor - the victim nodes
        y_targets: torch.Tensor - the target labels of the attack
        malicious_nodes: torch.Tensor - the attacker/malicious node
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)

        Returns
        -------
        attack_results: torch.Tensor - 2d-tensor that includes
                                       1st-col - the defence
                                       2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
    """
    # initialize
    model = attack.model_wrapper.model
    continuous_epochs = attack.continuous_epochs
    lr = attack.lr
    print_answer = attack.print_answer
    dataset = attack.getDataset()
    data = dataset.data

    num_attributes = data.x.shape[1]
    l_0_max_attributes_per_malicious = int(num_attributes * attack.l_0)
    l_0_max_attributes = l_0_max_attributes_per_malicious * malicious_nodes.shape[0]
    max_attributes = num_attributes * malicious_nodes.shape[0]

    log_template = createLogTemplate(attack=attack, dataset=dataset)

    # changing the parameters which require grads and setting adversarial optimizer
    optimizer_params = setRequiresGrad(model=model, malicious_nodes=malicious_nodes)
    optimizer = torch.optim.Adam(params=optimizer_params, lr=lr)

    # find best_attributes
    model0 = copy.deepcopy(model)
    previous_embeded_model = None
    for epoch in range(0, continuous_epochs):
        # train
        train(model=model, targeted=attack.targeted, attacked_nodes=attacked_nodes, y_targets=y_targets,
              optimizer=optimizer)
        is_zero_grad = model.is_zero_grad()

        # test correctness
        if not is_zero_grad:
            changed_attributes = (model.getInput() != model0.getInput())[malicious_nodes].sum().item()
            test_discrete(model=model, model0=model0, malicious_nodes=malicious_nodes, attacked_nodes=attacked_nodes,
                          changed_attributes=changed_attributes, max_attributes=max_attributes)
        else:
            changed_attributes = 0

        # test
        results = test(data=data, model=model, targeted=attack.targeted, attacked_nodes=attacked_nodes,
                       y_targets=y_targets)

        # breaks
        if is_zero_grad:
            if print_answer is Print.YES:
                print(log_template.format(node_num, epoch + 1, *results[:-1]), flush=True, end='')
            break

        if results[3]:
            # embed
            embeded_model = copy.deepcopy(model)
            for malicious_idx, malicious_node in enumerate(malicious_nodes):
                embedRowContinuous(model=embeded_model, malicious_node=malicious_node, model0=model0,
                                   l_inf=attack.l_inf, l_0=attack.l_0)

            # test correctness
            changed_attributes = (embeded_model.getInput() != model0.getInput())[malicious_nodes].sum().item()
            test_continuous(model=embeded_model, model0=model0, malicious_nodes=malicious_nodes,
                            attacked_nodes=attacked_nodes, changed_attributes=changed_attributes,
                            max_attributes=l_0_max_attributes, l_inf=attack.l_inf)
            # test
            results = test(data=data, model=embeded_model, targeted=attack.targeted, attacked_nodes=attacked_nodes,
                           y_targets=y_targets)
            if results[3]:
                if print_answer is Print.YES:
                    print(log_template.format(node_num, epoch + 1, *results[:-1]), flush=True, end='')
                break

            if previous_embeded_model is not None:
                if torch.norm(embeded_model.getInput() - previous_embeded_model.getInput(), p='fro') == 0:
                    if print_answer is Print.YES:
                        print(log_template.format(node_num, epoch + 1, *results[:-1]), flush=True, end='')
                    break
            previous_embeded_model = copy.deepcopy(embeded_model)
        
        # prints
        if print_answer is Print.YES:
            print(log_template.format(node_num, epoch + 1, *results[:-1]), flush=True, end='')
        if epoch != continuous_epochs - 1 and print_answer is not Print.NO:
            print()

    if print_answer is Print.YES:
        final_log = ''
        if results[3]:
            attr_percent = changed_attributes / (num_attributes * malicious_nodes.shape[0])
            final_log += ', l_0 used: {:.4f}'.format(attr_percent)
        final_log += ', Attack Success: {}'.format(results[-1])
        print(final_log + '\n', flush=True)
    if not results[3]:
        changed_attributes = max_attributes

    if attack.mode.isAdversarial():
        if results[3]:
            attack.setModel(embeded_model)
        else:
            attack.setModel(model0)
    return torch.tensor([[results[3], changed_attributes]]).type(torch.long)
