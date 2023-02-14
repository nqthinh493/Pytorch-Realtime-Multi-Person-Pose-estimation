import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

log_name = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f'./Results/logs/{log_name}')
x = torch.arange(-5, 100000, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())
z = -6 * x + 0.1 * torch.randn(x.size())
model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss1 = criterion(y1, y)
        loss2 = criterion(y1, z)
        writer.add_scalar("Loss/train", loss1, epoch)
        writer.add_scalar("Loss/train", loss2, epoch)
        optimizer.zero_grad()
        (loss1+loss2).backward()
        optimizer.step()

train_model(10)
writer.flush()
writer.close()

