from optimizer.base import Optimizer
from backend import xp

class GradientDescent(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model, lr)


    def step(self, X, y):
        self.model.grad(X, y)
        # 最原始的梯度下降法
        # 小batch更新就是SDG了

        self.model.weight.data -= self.lr * self.model.weight.grad
