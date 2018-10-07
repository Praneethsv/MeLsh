import torch as t
import numpy as np
import torch.nn as nn

class SubPolicyNet(nn.Module):

    def __init__(self, obs, action_space, hid_size, num_hidden_layers, num_sub_policies, gaussian_fixed_var=True):

        super(SubPolicyNet, self).__init__()
        self.obs = obs
        self.action_space = action_space
        self.num_hidden_layers = num_hidden_layers
        self.num_sub_policies = num_sub_policies
        self.gaussian_fixed_var = gaussian_fixed_var
        self.hid_size = hid_size

        obz =
        last_out = obz
        self.densenet = nn.Sequential(nn.Linear(last_out, self.hid_size),
                                      nn.Tanh())

    def forward(self, x):
        pass