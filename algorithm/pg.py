from algorithm.base import Optimizer
from backend import xp
from utils import prox_l1

class ProximalGradient(Optimizer):
    def __init__(self, model, lr, lam):
        super().__init__(model, lr)
        self.lam = lam

    def step(self, X, y):
        self.model.grad(X, y)

        for p in self.model.parameters():
            p2 = p.data - self.lr * p.grad
            p.data = prox_l1(p2, self.lam * self.lr)
