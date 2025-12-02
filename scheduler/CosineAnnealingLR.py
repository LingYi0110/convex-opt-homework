from scheduler.base import Scheduler
import math


# 余弦退火调度器
# 公式来源于 https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_t = self.optimizer.lr # 取初始lr作为起始点

    def step(self):
        # lr计算
        ratio = (1 + math.cos(math.pi * (self.last_epoch + 1) / self.T_max)) / (1 + math.cos(math.pi * self.last_epoch / self.T_max))
        eta_next = self.eta_min + (self.eta_t - self.eta_min) * ratio

        self.optimizer.lr = eta_next

        self.eta_t = eta_next
        self.last_epoch += 1
