from algorithm.base import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model, lr)

    def step(self, X, y):
        self.model.grad(X, y)
        # 最原始的梯度下降法
        for p in self.model.parameters():
            p.data -= self.lr * p.grad