from backend import set_backend, xp
from dataset import LibSVMDataset, DataLoader
from model.lasso import LASSO
from algorithm.pg import ProximalGradient
from algorithm.nesterov_pg import NesterovProximalGradient
from scheduler.StepLR import StepLR
from scheduler.CosineAnnealingLR import CosineAnnealingLR
from algorithm.sgd import StochasticGradientDescent
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter


set_backend('numpy')

name = 'YearPredictionMSD'
lr = 1e-6
lam = 1e-3
epochs = 1000
batch_size = 64

dataset = LibSVMDataset('../datasets/abalone_scale.txt', dtype=np.float32)
dataloader = DataLoader(dataset, batch_size=batch_size)

feature_dim = dataset.X.shape[1]


writer = SummaryWriter(log_dir=f'../runs/{name}')
model = LASSO(feature_dim, lam, sub_grad='zero')
#optimizer = NesterovProximalGradient(model, lr, lam)
optimizer = StochasticGradientDescent(model, lr=lr)
#scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
#scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-10)

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
    #scheduler.step()
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('train/loss_epoch', avg_loss, epoch)
    writer.add_scalar('train/lr', optimizer.lr, epoch)
    print(f"Epoch {epoch} | loss = {avg_loss:.6f}, lr = {optimizer.lr}")

writer.close()
print(model.weight.data)