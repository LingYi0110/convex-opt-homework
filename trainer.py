import importlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
from backend import xp, set_backend
from dataset import LibSVMDataset, DataLoader
from model.lasso import LASSO
from model.logistic import Logistic

from tensorboardX import SummaryWriter
from tqdm import tqdm

def _import_class(type_str: str):
    module_name, class_name = type_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 设置环境
        self._setup_environment()
        # 加载数据和模型
        self._setup_dataset()
        self._setup_model()
        # 设置优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()

    def _setup_environment(self):
        backend_name = self.config.get('trainer', {}).get('backend', 'numpy')
        set_backend(backend_name)

        seed = int(self.config.get('random_seed', 42))
        xp.random.seed(seed)

        self.name = self.config['experiment_name']
        self.log_dir = Path(self.config['log_dir'])
        self.epochs = int(self.config['epochs'])

    def _setup_dataset(self):
        config = self.config['dataset']

        if config['precision'] == 'float32':
            dtype = np.float32
        else:
            dtype = np.float64

        self.dataset = LibSVMDataset(path=config['path'], dtype=dtype)
        self.dataloader = DataLoader(self.dataset, batch_size=config['batch_size'])

    def _setup_model(self):
        # 反正不是通用框架，直接这样定了
        input_dim = self.dataset.X.shape[1]

        if 'lasso_model' in self.config:
            config = self.config['lasso_model']
            lam = config["lam"]
            sub_gradient = config.get("sub_gradient", "off")
            self.model = LASSO(input_dim=input_dim, lam=lam, sub_gradient=sub_gradient)
        elif 'logistic_model' in self.config:
            config = self.config['logistic_model']
            lam = config["lam"]
            norm = config["norm"]
            sub_gradient = config.get("sub_gradient", "off")
            self.model = Logistic(input_dim=input_dim, lam=lam, norm=norm, sub_gradient=sub_gradient)
        else:
            raise ValueError("No supported model found in config")

    def _setup_optimizer(self):
        config = dict(self.config["optimizer"])
        type_str = config.pop("type") # 把type删除，防止传参出错
        optimizer_class = _import_class(type_str)
        self.optimizer = optimizer_class(self.model, **config)

    def _setup_scheduler(self):
        config = self.config.get("scheduler")
        if not config:
            self.scheduler = None # lr保持不变
            return
        config = dict(config)
        type_str = config.pop("type")
        scheduler_class = _import_class(type_str)
        self.scheduler = scheduler_class(self.optimizer, **config)

    def train(self):
        writer = SummaryWriter(log_dir=self.log_dir / self.name)

        global_step = 0
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch} / {self.epochs}", leave=True)
            total_loss = 0

            for X, y in pbar:
                loss = self.model.loss(X, y)

                total_loss += loss.item()
                global_step += 1
                self.optimizer.step(X, y)
                pbar.set_postfix(loss=float(loss), lr=float(self.optimizer.lr))
                writer.add_scalar('train/step_loss', float(loss), global_step)

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = total_loss / len(self.dataloader)

            writer.add_scalar('train/loss_epoch', avg_loss, epoch)
            writer.add_scalar('train/lr', self.optimizer.lr, epoch)
            #print(f"Epoch {epoch} | loss = {avg_loss:.6f}, lr = {self.optimizer.lr}\n")

        writer.close()