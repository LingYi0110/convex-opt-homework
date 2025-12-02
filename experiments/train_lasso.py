from backend import set_backend, xp
from dataset import LibSVMDataset, DataLoader
from model.lasso import LASSO
from algorithm.pg import ProximalGradient


set_backend('numpy')

dataset = LibSVMDataset('../datasets/mg_scale.txt')
dataloader = DataLoader(dataset)

feature_dim = dataset.X.shape[1]

lr = 1e-3
lam = 1e-3
epochs = 10000

model = LASSO(feature_dim, lam)
optimizer = ProximalGradient(model, lr, lam)

for epoch in range(1, epochs + 1):
    total_loss = 0
    for X, y in dataloader:
        loss = model.loss(X, y)

        total_loss += loss
        optimizer.step(X, y)
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch:03d} | loss = {avg_loss:.6f}")
print(model.weight.data)