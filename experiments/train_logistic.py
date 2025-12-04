from scipy._lib.cobyqa.subsolvers import optim

from backend import set_backend, xp
from dataset import LibSVMDataset, DataLoader
from model.logistic import Logistic
from algorithm.pg import ProximalGradient
from algorithm.nesterov_pg import NesterovProximalGradient
from scheduler.StepLR import StepLR
from scheduler.CosineAnnealingLR import CosineAnnealingLR
from algorithm.gd import GradientDescent
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter


set_backend('numpy')

xp.random.seed(42)

name = 'german.numer_scale'
lr = 1e-2
lam = 1e-2
epochs = 1000
batch_size = None

dataset = LibSVMDataset('../datasets/classification/german.numer_scale.txt', dtype=np.float32)
dataloader = DataLoader(dataset, batch_size=batch_size)

feature_dim = dataset.X.shape[1]


writer = SummaryWriter(log_dir=f'../runs/{name}')
model = Logistic(feature_dim, lam, norm='l1', subgrad='off')

optimizer = NesterovProximalGradient(model, lr, lam)
#optimizer = GradientDescent(model, lr=lr)
#scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

global_step = 0
for epoch in range(1, epochs + 1):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} / {epochs}", leave=False)
    total_loss = 0
    for X, y in pbar:
        loss = model.loss(X, y)

        total_loss += loss.item()
        global_step += 1
        optimizer.step(X, y)
        pbar.set_postfix(loss=float(loss))
        writer.add_scalar('train/step_loss', float(loss), global_step)
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('train/loss_epoch', avg_loss, epoch)
    writer.add_scalar('train/lr', optimizer.lr, epoch)
    print(f"Epoch {epoch} | loss = {avg_loss:.6f}, lr = {optimizer.lr}")

writer.close()
print(model.weight.data)