#!/usr/bin/env python3
import numpy as np
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim

class ChessValueDataset(Dataset):
  def __init__(self, fname, device="cuda"):
    dat = np.load(fname)
    self.X = dat['arr_0']
    self.Y = dat['arr_1']
    print("loaded", self.X.shape, self.Y.shape)
    self.device = device

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
      x = torch.tensor(self.X[idx], dtype=torch.bfloat16, device=self.device)
      y = torch.tensor(self.Y[idx], dtype=torch.long, device=self.device)
      return x, y


class Net(nn.Module):
  """
  Classification convolutional neural network for image like features.
  """
  def __init__(self, num_inputs=5, num_outputs = 1):
    """
    Convolutional layers (Conv2D) help detect these patterns by learning filters that highlight features like
    lines, corners, and more abstract representations in deeper layers.
    As the spatial size shrinks with stride, the number of filters grows based on convnet size

    NOTE: Conv2D initializes biases to float32, and so does Linear with its weights,
      which means the input vector needs to have the same dtype

    padding=1 ensures the convnet doesn't shrink the dimensions.
    stride=N reduces spatial dimensions (width x height) for downsampling by a factor of N dimensions
    num_filters = number of filters
    """
    super(Net, self).__init__() # inherit from parent

    self.a1 = nn.Conv2d(num_inputs, 16, kernel_size=3, padding=1)
    self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
    self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

    # spatial dimensions = 8/2 x 8/2
    self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

    # spatial dimensions = 8/2^2 x 8/2^2
    self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
    self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
    self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

    # spatial dimensions = 8/2^3 x 8/2^3
    self.d1 = nn.Conv2d(128, 128, kernel_size=1)
    self.d2 = nn.Conv2d(128, 128, kernel_size=1)
    self.d3 = nn.Conv2d(128, 128, kernel_size=1)

    self.relu = nn.ReLU()
    self.tanh = nn.tanh()

    self.last = nn.Linear(128, num_outputs)

  def forward(self, x):
    # 4x4
    # 2x2
    # 1x128
    x = x.view(-1, 128)
    x = self.relu(self.a1(x))
    x = self.relu(self.a2(x))
    x = self.relu(self.a3(x))

    x = self.relu(self.b1(x))
    x = self.relu(self.b2(x))
    x = self.relu(self.b3(x))

    x = self.relu(self.c1(x))
    x = self.relu(self.c2(x))
    x = self.relu(self.c3(x))

    x = self.relu(self.d1(x))
    x = self.relu(self.d2(x))
    x = self.relu(self.d3(x))

    x = self.last(x)

    # value output
    return F.tanh(x)

if __name__ == "__main__":

  device = "cpu"
  if torch.cuda.is_available():
      device = "cuda":

  fname_in = pathlib.Path(__file__) / "processed" / "dataset_5M.npz"
  DIR_OUT = pathlib.Path(__file__).parent / "nets"
  DIR_OUT.mkdir(exist_ok=True, parents=True)

  chess_dataset = ChessValueDataset(fname_in, device)
  train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle=True)
  model = Net()
  optimizer = optim.Adam(model.parameters())
  floss = nn.MSELoss()
  model.to(device)

  model.train()

  for epoch in range(100):
    all_loss = 0
    num_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      target = target.unsqueeze(-1)

      #print(data.shape, target.shape)
      optimizer.zero_grad()
      output = model(data)
      #print(output.shape)

      loss = floss(output, target)
      loss.backward()
      optimizer.step()
      
      all_loss += loss.item()
      num_loss += 1

    print("%3d: %f" % (epoch, all_loss/num_loss))
    torch.save(model.state_dict(), str(DIR_OUT / "value.pth"))

