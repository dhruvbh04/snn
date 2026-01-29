# -*- coding: utf-8 -*-

pip install snntorch

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""##Data Loading"""

batch_size = 128
data_path = '/tmp/data/mnist'

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

"""## Network Architecture"""

num_steps = 25
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #Define Network
        self.fc1 = nn.Linear(784, 512)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(512, 10)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net().to(device)

"""##Training"""

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 5
loss_hist = []

for epoch in range(num_epochs):
    train_batch = iter(train_loader)
    total_epoch_loss = 0
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        net.train()
        spk_rec, _ = net(data)

        # Loss
        loss_val = loss_fn(spk_rec.sum(0), targets)

        # Gradient calculation
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())
        total_epoch_loss += loss_val.item()
    avg_loss = total_epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

"""##Loss Curve"""

fig = plt.figure(figsize=(10,5))
plt.plot(loss_hist)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

"""##Raster Plot"""

data, targets = next(iter(test_loader))
data = data.to(device)
# Forward pass
net.eval()
spk_rec, _ = net(data)
idx = 0
fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
splt.raster(spk_rec[:, idx, :], ax, s=1.5, c="black")

plt.title(f"Output Layer Spike Raster (Target Label: {targets[idx]})")
plt.xlabel("Time step")
plt.ylabel("Neuron Index")
plt.show()

def measure_accuracy(model, dataloader):
  with torch.no_grad():
    model.eval()
    running_length = 0
    running_accuracy = 0

    for data, targets in iter(dataloader):
      data = data.to(device)
      targets = targets.to(device)

      # Forward-pass
      spk_rec, _ = model(data)
      spike_count = spk_rec.sum(0)
      _, max_spike = spike_count.max(1)

      # correct classes for one batch
      num_correct = (max_spike == targets).sum()

      # total accuracy
      running_length += len(targets)
      running_accuracy += num_correct

    accuracy = (running_accuracy / running_length)

    return accuracy.item()

print(f"Test set accuracy: {measure_accuracy(net, test_loader)}")