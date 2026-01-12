import json, random, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torchvision import datasets, transforms
from tqdm import trange
from model import SimpleCNN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
ART = ROOT/"artifacts"
ART.mkdir(exist_ok=True)

SEED=1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)
tfm_train = transforms.Compose([transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.Normalize(MEAN,STD)])
tfm_eval = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

tr = datasets.CIFAR10(root=DATA, train=True, download=True, transform=tfm_train)
te = datasets.CIFAR10(root=DATA, train=False, download=True, transform=tfm_eval)
full = ConcatDataset([tr, datasets.CIFAR10(root=DATA, train=False, transform=tfm_eval)])
N=len(full); sizes=[int(0.8*N), int(0.1*N)]; sizes.append(N-sum(sizes))
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full, sizes, generator=g)

BATCH=64
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

device = torch.device("cpu")
model = SimpleCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def run(loader, train=False):
    model.train(train)
    total=0; correct=0; loss_sum=0.0
    for x,y in loader:
        x,y=x.to(device), y.to(device)
        if train: opt.zero_grad()
        out = model(x)
        loss = crit(out,y)
        if train: loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1); correct += int((pred==y).sum()); total += x.size(0)
    return loss_sum/total, correct/total

best_acc=0.0
EPOCHS=3
for ep in trange(EPOCHS, desc="epochs"):
    tr_loss,tr_acc = run(train_loader, train=True)
    va_loss,va_acc = run(val_loader, train=False)
    if va_acc>best_acc:
        best_acc=va_acc
        torch.save(model.state_dict(), ART/"best_state.pt")

model.load_state_dict(torch.load(ART/"best_state.pt", map_location=device))
te_loss, te_acc = run(test_loader, train=False)
print({"val_best": float(best_acc), "test_acc": float(te_acc)})

model.eval()
scripted=torch.jit.script(model)
scripted.save(str(ART/"scripted_model.pt"))

meta = {"mean":MEAN, "std":STD, "labels": list(range(10))}
with open(ART/"meta.json","w") as f: json.dump(meta,f, indent=2)
print("Saved:", str(ART/"scripted_model.pt"))