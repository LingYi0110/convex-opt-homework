from scheduler.base import Scheduler
from backend import xp
import math


class StepLR(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, X, y):
        #每step_size次就会衰减一次
        if self.last_epoch % self.step_size == 0 and self.last_epoch != 0:
            self.optimizer.lr *= self.gamma


class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_t = self.optimizer.lr # 取初始lr作为起始点

    def step(self, X, y):
        # lr计算
        # 公式来源于 https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        ratio = (1 + math.cos(math.pi * (self.last_epoch + 1) / self.T_max)) / (1 + math.cos(math.pi * self.last_epoch / self.T_max))
        eta_next = self.eta_min + (self.eta_t - self.eta_min) * ratio

        self.optimizer.lr = eta_next

        self.eta_t = eta_next


class BarzilaiBorwein(Scheduler):
    def __init__(self, optimizer, lr_type='BB1', c1=1e-4, alpha=1, decay=0.5, memory_size=10, lr_min=1e-10, lr_max=1e4):
        super().__init__(optimizer)
        self.lr_type = lr_type
        self.c1 = c1
        self.decay = decay
        self.memory_size = memory_size
        self.alpha = alpha
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.model = self.optimizer.model
        self._max_val = float('-inf')

    def step(self, X, y):
        weight = self.optimizer.model.weight.data
        grad = self.optimizer.model.weight.grad

        if self.model.loss(X, y) > self._max_val and self.last_step <= self.memory_size:
            self._max_val = self.model.loss(X, y)

        # 非单调Armijo条件
        sk = xp.copy(weight)
        yk = xp.copy(grad)

        self.model.grad(X, y)
        f1 = self._max_val - self.c1 * self.alpha * grad.T @ grad

        while True:
            self.model.grad(X, y)
            weight -= self.alpha * grad
            f2 = self.model.loss(X, y)
            if f2 >= f1:
                weight[...] = sk
                self.alpha *= self.decay # 回退
                self.model.grad_zero()
                continue
            else:
                sk = weight - sk
                yk = grad - yk
                weight += self.alpha * grad
                self.model.grad_zero()
                break

        # 计算下一步的BB步长
        if self.lr_type == 'BB1':
            self.alpha = (sk.T @ yk) / (yk.T @ yk)
        elif self.lr_type == 'BB2':
            self.alpha = (sk.T @ sk) / (sk.T @ yk)

        # 对步长进行限制
        self.alpha = max(min(self.alpha, self.lr_max), self.lr_min)

        self.optimizer.lr = self.alpha
