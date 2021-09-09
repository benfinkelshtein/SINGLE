import torch


@torch.no_grad()
def test_continuous(model, model0, malicious_nodes: torch.Tensor, attacked_nodes: torch.Tensor,
                    changed_attributes: int, max_attributes: int, l_inf: float) -> torch.Tensor:
    """
        the main test function for continuous datasets
        "#perturbed nodes < #malicious nodes"
        "#perturbed nodes is zero"
        "Attributes from non-malicious nodes are changed"
        "The change of attributes is inconsistent"
        "Allowed number of attributes has been exceeded"
        "L_inf limit is broken"
        
        Parameters
        ----------
        model: Model - post-attack model
        model0: Model - pre-attack model
        malicious_nodes: torch.Tensor
        attacked_nodes: torch.Tensor
        changed_attributes: int - the number of changed attributes according to the post and pre attack models
        max_attributes: int - the total allowed number of attributes
        l_inf: float - the limit on the l_inf value
    """
    max_abs_diff = test_discrete(model, model0, malicious_nodes, attacked_nodes, changed_attributes, max_attributes)

    assert max_abs_diff <= l_inf + 1e-2, "L_inf limit is broken"


@torch.no_grad()
def test_discrete(model, model0, malicious_nodes: torch.Tensor, attacked_nodes: torch.Tensor,
                  changed_attributes: int, max_attributes: int) -> float:
    """
        the main tests for discrete datasets:
        "#perturbed nodes < #malicious nodes"
        "#perturbed nodes is zero"
        "Attributes from non-malicious nodes are changed"
        "The change of attributes is inconsistent"
        "Allowed number of attributes has been exceeded"
        
        Parameters
        ----------
        model: Model - post-attack model
        model0: Model - pre-attack model
        malicious_nodes: torch.Tensor
        attacked_nodes: torch.Tensor
        changed_attributes: int - the number of changed attributes according to the post and pre attack models
        max_attributes: int - the total allowed number of attributes
        
        Returns
        -------
        x_abs_diff.max().item(): float - maximum change in an attribute
    """
    x0 = model0.getInput()
    x = model.getInput()
    x_diff = (x - x0)
    x_abs_diff = x_diff.abs()
    change_in_nodes = (x_abs_diff != 0).sum(dim=-1)

    indices_of_changed_nodes = change_in_nodes.nonzero(as_tuple=True)[0].tolist()
    changed_attributes_test = change_in_nodes.sum()

    assert len(indices_of_changed_nodes) <= malicious_nodes.shape[0], "#perturbed nodes < #malicious nodes"
    # An edge case where:
    # the log_softmax has a zero in it (as a result of huge differences in the values of the softmax).
    # Therefore, no perturbation of a feature would help
    attacked_nodes_output = model()[attacked_nodes]
    zeros_per_row = torch.all(attacked_nodes_output != 0, dim=1)
    rows_with_zeros = (zeros_per_row == 0).sum()
    assert len(indices_of_changed_nodes) >= 1 or rows_with_zeros == attacked_nodes.numel(), "#perturbed nodes is zero"
    set_of_changed_nodes = set(indices_of_changed_nodes)
    set_of_malicious_nodes = set(malicious_nodes.tolist())
    assert set_of_changed_nodes.issubset(set_of_malicious_nodes) or rows_with_zeros == attacked_nodes.numel(),\
        "Attributes from non-malicious nodes are changed"

    assert changed_attributes_test.item() == changed_attributes, "The change of attributes is inconsistent"
    assert changed_attributes <= max_attributes, "Allowed number of attributes has been exceeded"

    return x_abs_diff.max().item()
