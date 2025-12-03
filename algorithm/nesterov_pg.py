from algorithm.base import Optimizer
from backend import xp


class NesterovProximalGradient(Optimizer):
    def __init__(self, model, lr, lam):
        super().__init__(model, lr)
        self.lam = lam
        self.k = 0
        self.prev = [xp.copy(p.data) for p in model.parameters()]

    def step(self, X, y):
        self.model.grad(X, y)

        params = list(self.model.parameters())

        for idx, p in enumerate(params):
            p2 = p.data - self.lr * p.grad
            p3 = prox_l1(p2, self.lam * self.lr)

            p.data = p3 + (self.k / (self.k + 3)) * (p3 - self.prev[idx])
            self.prev[idx] = xp.copy(p3)
        self.k += 1


def prox_l1(v, lam):
    return xp.sign(v) * xp.maximum(xp.abs(v) - lam, 0)
