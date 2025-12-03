from backend import set_backend, xp
from dataset import LibSVMDataset, DataLoader
from model.lasso import LASSO
from algorithm.pg import ProximalGradient
from algorithm.nesterov_pg import NesterovProximalGradient
from scheduler.StepLR import StepLR
from scheduler.CosineAnnealingLR import CosineAnnealingLR
from algorithm.gd import GradientDescent

set_backend('numpy')

dataset = LibSVMDataset('../datasets/mg_scale.txt')
dataloader = DataLoader(dataset)

feature_dim = dataset.X.shape[1]

lr = 1e-4
lam = 1e-1
epochs = 1000

model = LASSO(feature_dim, lam, sub_grad='random')
#optimizer = NesterovProximalGradient(model, lr, lam)
optimizer = GradientDescent(model, lr)
#scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

for epoch in range(1, epochs + 1):
    total_loss = 0
    for X, y in dataloader:
        loss = model.loss(X, y)

        total_loss += loss
        optimizer.step(X, y)
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f}, lr = {scheduler.get_lr()}")

print(model.weight.data)