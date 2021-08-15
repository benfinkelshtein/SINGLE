"""This file contains the Projected Gradient Descent attack as proposed in:
Kaidi Xu, Hongge Chen, Sijia Liu, Pin Yu Chen, Tsui Wei Weng, Mingyi Hong, and Xue Lin.Topology attack and defense
for graph neural networks: An optimization perspective. IJCAI International Joint Conference on Artificial
Intelligence, 2019-Augus:3961–3967, 2019. ISSN10450823. doi: 10.24963/ijcai.2019/550.

The Subsequent code build upon the implementation https://github.com/DSE-MSU/DeepRobust (under MIT License). We did
not intent to unify the code style, programming paradigms, etc. with the rest of the code base.

"""

import numpy as np
import torch
from torch.nn import functional as F

from rgnn.models import DenseGCN


class PGD(object):
    """L_0 norm Projected Gradient Descent (PGD) attack as proposed in:
    Kaidi Xu, Hongge Chen, Sijia Liu, Pin Yu Chen, Tsui Wei Weng, Mingyi Hong, and Xue Lin.Topology attack and defense
    for graph neural networks: An optimization perspective. IJCAI International Joint Conference on Artificial
    Intelligence, 2019-Augus:3961–3967, 2019. ISSN10450823. doi: 10.24963/ijcai.2019/550.

    Parameters
    ----------
    X : torch.Tensor
        [n, d] feature matrix.
    adj : torch.sparse.FloatTensor
        [n, n] sparse adjacency matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked [?].
    model : DenseGCN
        Model to be attacked.
    epochs : int, optional
        Number of epochs to attack the adjacency matrix, by default 200.
    loss_type : str, optional
        'CW' for Carlini and Wagner or 'CE' for cross entropy, by default 'CE'.
    """

    def __init__(self,
                 X: torch.Tensor,
                 adj: torch.sparse.FloatTensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: DenseGCN,
                 epochs: int = 200,
                 epsilon: float = 1e-5,
                 loss_type: str = 'CE',
                 **kwargs):
        assert adj.device == X.device, 'The device of the features and adjacency matrix must match'
        self.X = X
        self.adj = adj
        if self.adj.is_sparse:
            self.adj = self.adj.to_dense()
        self.labels = labels
        self.idx_attack = idx_attack
        self.model = model
        self.epochs = epochs
        self.epsilon = epsilon
        self.loss_type = loss_type

        self.n = self.X.shape[0]
        self.device = X.device
        self.n_perturbations = 0

        self.attr_adversary = self.X  # Only the adjacency matrix will be perturbed
        self.adj_adversary = None

    def attack(self, n_perturbations: int, **kwargs):
        """Perform attack (`n_perturbations` is increasing as it was a greedy attack).

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        self.n_perturbations += n_perturbations

        self.complementary = None
        self.adj_changes = torch.zeros(int(self.n * (self.n - 1) / 2), dtype=torch.float, device=self.device)
        self.adj_changes.requires_grad = True

        self.model.eval()
        for t in range(self.epochs):
            modified_adj = self.get_modified_adj()
            output = self.model(self.X, modified_adj)
            loss = self._loss(output)
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection()

        self.random_sample()
        self.adj_adversary = self.get_modified_adj().detach().to_sparse()

    def random_sample(self):
        K = 20
        best_loss = float('-Inf')
        with torch.no_grad():
            while best_loss == float('-Inf'):
                s = self.adj_changes.cpu().detach().numpy()
                for i in range(K):
                    sampled = np.random.binomial(1, s)

                    if sampled.sum() > self.n_perturbations:
                        continue
                    self.adj_changes.data.copy_(torch.tensor(sampled))
                    modified_adj = self.get_modified_adj()
                    output = self.model(self.X, modified_adj)
                    loss = self._loss(output)
                    if best_loss < loss:
                        best_loss = loss
                        best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output):
        output = output[self.idx_attack]
        labels = self.labels[self.idx_attack]
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            eye = torch.eye(labels.max() + 1, device=self.device)
            onehot = eye[labels]
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - output[np.arange(len(output)), best_second_class]
            loss = -torch.clamp(margin, min=0).mean()
        return loss

    def projection(self):
        if torch.clamp(self.adj_changes, 0, 1).sum() > self.n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = PGD.bisection(left, right, self.adj_changes, self.n_perturbations, self.epsilon)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def get_modified_adj(self):
        if self.complementary is None:
            self.complementary = torch.ones_like(self.adj) - torch.eye(self.n, device=self.device) - 2 * self.adj

        m = torch.zeros_like(self.adj)
        tril_indices = torch.tril_indices(row=self.n, col=self.n, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + self.adj

        return modified_adj

    @staticmethod
    def bisection(a: float, b: float, adj_changes: torch.Tensor, n_perturbations: int, epsilon: float):
        def func(x):
            return torch.clamp(adj_changes - x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b - a) >= epsilon):
            miu = (a + b) / 2
            if (func(miu) == 0.0):
                break
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
        return miu
