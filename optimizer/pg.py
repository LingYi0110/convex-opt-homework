from optimizer.base import Optimizer
from backend import xp
from utils import prox_l1

class ProximalGradient(Optimizer):
    def __init__(self, model, lr, nesterov=False):
        super().__init__(model, lr)
        self.lam = model.lam
        self.nesterov = nesterov

        self._k = 0
        self._prev = [xp.copy(p.data) for p in model.parameters()]

    def step(self, X, y):
        self.model.grad(X, y)

        if self.nesterov:
            params = list(self.model.parameters())

            for idx, p in enumerate(params):
                p2 = p.data - self.lr * p.grad
                p3 = self.model.prox(p2, self.lam * self.lr)

                p.data = p3 + (self._k / (self._k + 3)) * (p3 - self._prev[idx])
                self._prev[idx] = xp.copy(p3)
            self._k += 1
        else:
            for p in self.model.parameters():
                p2 = p.data - self.lr * p.grad
                p.data = self.model.prox(p2, self.lam * self.lr)
