from backend import xp, set_backend
from dataset import LibSVMDataset, DataLoader

set_backend('numpy')

dataset = LibSVMDataset('../datasets/abalone_scale.txt')
loader = DataLoader(dataset, 16)

for batch in loader:
    print(batch)