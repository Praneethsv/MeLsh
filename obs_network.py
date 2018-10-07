import torch as t
import numpy as np
import torch.nn as nn



class obs_network(nn.Module):

    def __init__(self, input_shape, ob):

        super(obs_network, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4, padding="VALID"),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 16, kernel_size=4, stride=2, padding="VALID"),
                                  nn.ReLU())

        conv_out_size = self._get_conv_size(input_shape)

        self.fc = nn.Sequential(nn.Linear(conv_out_size, 64),
                                nn.ReLU())

    def _get_conv_size(self, shape):

        o = self.conv(t.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out_flatten = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out_flatten)