from model.base import BaseModel, Parameter
from utils import *
from backend import xp

class LASSO(BaseModel):
    def __init__(self, input_dim: int, lam: float, sub_grad: str = 'off'):
        super().__init__()
        self.weight = Parameter(xp.zeros(input_dim))
        self.lam_l1 = lam
        self.sub_grad = sub_grad

    def forward(self, X):
        return X @ self.weight.data

    def loss(self, X, y):
        # 最小化这一部分，怎么看都像是一个MSE的损失函数加上一个L1正则
        # 不过也能理解，其实就是既要能使得预测值和真实值接近，又要使得权重尽力变小
        # 最后的结果应该就是选择重要的特征
        residual = self.forward(X) - y
        f = 0.5 * l2_norm(residual)**2
        g = self.lam_l1 * l1_norm(self.weight.data)
        return f + g

    def l1_subgrad(self):
        if self.sub_grad == 'off':
            return xp.zeros_like(self.weight.data)
        elif self.sub_grad == 'zero':
            return self.lam_l1 * xp.sign(self.weight.data)
        elif self.sub_grad == 'random':
            g = xp.sign(self.weight.data)
            zero = (self.weight.data == 0)
            g[zero] = xp.random.uniform(-1.0, 1.0, size=zero.sum().item())
            return self.lam_l1 * g
        else:
            raise NotImplementedError(f'Not Supported SubGradient Mode:{self.sub_grad}')

    def grad(self, X, y):
        # 没有自动求导 :(
        residual = self.forward(X) - y
        self.weight.grad = X.T @ residual + self.l1_subgrad()