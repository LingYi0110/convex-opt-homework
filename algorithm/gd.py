from algorithm.base import Optimizer
from backend import xp

class GradientDescent(Optimizer):
    def __init__(self, model, lr, momentum=0):
        super().__init__(model, lr)
        self.momentum = momentum
        self.velocity = None
        if momentum > 0:
            self.velocity = [xp.zeros_like(p.data) for p in model.parameters()]

    def step(self, X, y):
        self.model.grad(X, y)
        # 最原始的梯度下降法
        # 小batch更新就是SDG了
        if self.momentum == 0:
            for p in self.model.parameters():
                p.data -= self.lr * p.grad
        else:
            for v, p in zip(self.velocity, self.model.parameters()):
                v = v * self.momentum + p.grad
                p.data -= self.lr * v