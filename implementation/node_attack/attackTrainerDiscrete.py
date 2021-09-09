from node_attack.attackTrainerHelpers import createLogTemplate, setRequiresGrad, train, test, flipUpBestNewAttributes
from classes.basic_classes import Print
from node_attack.attackTrainerTests import test_discrete

import torch
import copy


def attackTrainerDiscrete(attack, attacked_nodes: torch.Tensor, y_targets: torch.Tensor, malicious_nodes: torch.Tensor,
                          node_num: int, discrete_stop_after_1iter: bool) -> torch.Tensor:
    """
        a trainer function that attacks our model by changing the input attribute for a limited number of attributes
        1.attack the model with i attributes
        2.backprop
        3.add the attribute with the largest gradient as the i+1 attribute

        Parameters
        ----------
        attack: oneGNNAttack
        attacked_nodes: torch.Tensor - the victim nodes
        y_targets: torch.Tensor - the target labels of the attack
        malicious_nodes: torch.Tensor - the attacker/malicious node
        node_num: int - the index of the attacked/victim node (out of the train/val/test-set)
        discrete_stop_after_1iter: bool - whether or not to stop the discrete after 1 iteration
                                          this is a specific flag for the GRAD_CHOICE Approach

        Returns
        -------
        attack_results: torch.Tensor - 2d-tensor that includes
                                       1st-col - the defence
                                       2nd-col - the number of attributes used
        if the number of attributes is 0 the node is misclassified to begin with
    """
    # initialize
    model = attack.model_wrapper.model
    lr = attack.lr
    print_answer = attack.print_answer
    dataset = attack.getDataset()
    data = dataset.data

    num_attributes = data.x.shape[1]
    l_0_max_attributes_per_malicious = int(num_attributes * attack.l_0)
    limited_max_attributes = l_0_max_attributes_per_malicious * malicious_nodes.shape[0]

    changed_attributes_all_malicious, epoch = 0, 0
    log_template = createLogTemplate(attack=attack, dataset=dataset)

    # changing the parameters which require grads and setting adversarial optimizer
    optimizer_params = setRequiresGrad(model=model, malicious_nodes=malicious_nodes)
    optimizer = torch.optim.Adam(params=optimizer_params, lr=lr)
    optimizer.zero_grad()

    # zero attributes
    with torch.no_grad():
        changed_attributes = 0
        for malicious_node in malicious_nodes:
            changed_attributes += model.node_attribute_list[malicious_node][0].sum().item()
            model.setNodesAttributes(idx_node=malicious_node, values=torch.zeros(num_attributes))

    # flip the attribute with the largest gradient
    model0 = copy.deepcopy(model)
    changed_attributes, prev_changed_attributes = 0, 0
    num_attributes_left = l_0_max_attributes_per_malicious * torch.ones_like(malicious_nodes).to(attack.device)
    while True:
        epoch += 1
        prev_model = copy.deepcopy(model)
        # train
        train(model=model, targeted=attack.targeted, attacked_nodes=attacked_nodes, y_targets=y_targets,
              optimizer=optimizer)
        is_zero_grad = model.is_zero_grad()

        # test correctness
        if not is_zero_grad:
            num_attributes_left = flipUpBestNewAttributes(model=model, model0=prev_model, malicious_nodes=malicious_nodes,
                                                          num_attributes_left=num_attributes_left)
            changed_attributes = limited_max_attributes - num_attributes_left.sum().item()

            test_discrete(model=model, model0=model0, malicious_nodes=malicious_nodes, attacked_nodes=attacked_nodes,
                          changed_attributes=changed_attributes, max_attributes=limited_max_attributes)
        else:
            changed_attributes = 0

        # test
        results = test(data=data, model=model, targeted=attack.targeted, attacked_nodes=attacked_nodes,
                       y_targets=y_targets)

        # prints
        if print_answer is not Print.NO and epoch != 1:
            print()
        if print_answer is Print.YES:
            print(log_template.format(node_num, epoch, changed_attributes, *results[:-1]), flush=True, end='')
        # breaks
        if is_zero_grad:
            break
        if results[3] or changed_attributes == limited_max_attributes or changed_attributes == prev_changed_attributes:
            break
        prev_changed_attributes = changed_attributes
        if discrete_stop_after_1iter:
            break

    if print_answer is Print.YES:
        final_log = ''
        if results[3]:
            attr_percent = changed_attributes / (num_attributes * malicious_nodes.shape[0])
            final_log += ', l_0 used: {:.4f}'.format(attr_percent)
        final_log += ', Attack Success: {}'.format(results[-1])
        print(final_log + '\n', flush=True)
    if changed_attributes > limited_max_attributes:
        return torch.tensor([[results[3], limited_max_attributes]]).type(torch.long)
    else:
        return torch.tensor([[results[3], changed_attributes]]).type(torch.long)
