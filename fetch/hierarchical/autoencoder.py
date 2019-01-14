import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        size_encoded = self.encoder(torch.zeros(input_size)).shape()







function autoencoder:initialize()
  local pool_layer1 = nn.SpatialMaxPooling(2, 2, 2, 2)
  local pool_layer2 = nn.SpatialMaxPooling(2, 2, 2, 2)

  self.net = nn.Sequential()
  self.net:add(nn.SpatialConvolution(3, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(pool_layer1)
  self.net:add(nn.SpatialConvolution(12, 24, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(pool_layer2)
  self.net:add(nn.Reshape(24 * 14 * 14))
  self.net:add(nn.Linear(24 * 14 * 14, 1568))
  self.net:add(nn.Linear(1568, 24 * 14 * 14))
  self.net:add(nn.Reshape(24, 14, 14))
  self.net:add(nn.SpatialConvolution(24, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxUnpooling(pool_layer2))
  self.net:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 0, 0))
  self.net:add(nn.ReLU())
  self.net:add(nn.SpatialMaxUnpooling(pool_layer1))
  self.net:add(nn.SpatialConvolution(12, 3, 3, 3, 1, 1, 0, 0))

  self.net = self.net:cuda()
end