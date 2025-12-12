from optimizer.base import Optimizer
from backend import xp
from utils import prox_l1

class ProximalGradient(Optimizer):
    def __init__(self, model, lr, nesterov=False):
        super().__init__(model, lr)
        self.lam = model.lam
        self.nesterov = nesterov # FISTA算法

        self._k = 1
        self._prev = [xp.copy(model.weight.data), xp.copy(model.weight.data)]

    def step(self, X, y):
        w = self.model.weight

        if self.nesterov:
            y_k = self._prev[1] + (self._k - 2) / (self._k + 1) * (self._prev[1] - self._prev[0])
            w.data = y_k
            self.model.grad(X, y)

            self._prev[0] = xp.copy(self._prev[1])
            self._prev[1] = self.model.prox(y_k - self.lr * w.grad, self.lam * self.lr)
            w.data = self._prev[1]
            self._k += 1
        else:
            self.model.grad(X, y)
            w_2 = w.data - self.lr * w.grad
            w.data = self.model.prox(w_2, self.lam * self.lr)

