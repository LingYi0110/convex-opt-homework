from .base import Optimizer
from backend import xp


class BFGS(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model, lr)
        self.I = xp.eye(len(self.model.weight.data))
        self.hk = self.I

    def step(self, X, y):
        gk = self.model.grad(X, y)

        n = len(gk)
        if self.hk is None:
            self.hk = xp.eye(n)

        dk = - self.hk @ gk
        sk = self.lr * dk

        self.model.weight.data += sk
        gk1 = self.model.grad(X, y)

        yk = gk1 - gk
        if sk.T @ yk <= 1e-12:
            self.hk = self.I
            return
        rho_k = 1 / (sk.T @ yk)

        self.hk = (self.I - rho_k * xp.outer(sk, yk)) @ self.hk @ (self.I - rho_k * xp.outer(yk, sk)) + rho_k * xp.outer(sk, sk)