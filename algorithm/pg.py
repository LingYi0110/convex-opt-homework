from algorithm.base import Optimizer
from backend import xp


class ProximalGradient(Optimizer):
    def __init__(self, model, lr, lam):
        super().__init__(model, lr)
        self.lam = lam

    def step(self, X, y):
        self.model.grad(X, y)

        for p in self.model.parameters():
            p2 = p.data - self.lr * p.grad
            p.data = prox_l1(p2, self.lam * self.lr)

def prox_l1(v, lam):
    # 推导看报告吧，这里不写了
    # for循环对每个分量操作太慢了，不如下面
    return xp.sign(v) * xp.maximum(xp.abs(v) - lam, 0)