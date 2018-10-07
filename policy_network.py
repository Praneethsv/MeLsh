import torch as t
import numpy as np
import torch.nn as nn
from running_mean_std import RunningMeanStd

class PolicyNet(nn.Module):

    def __init__(self, obs, action_space, hid_size, num_hidden_layers, num_sub_policies, gaussian_fixed_var=True):

        super(PolicyNet, self).__init__()
        self.obs = obs
        self.action_space = action_space
        self.num_hidden_layers = num_hidden_layers
        self.num_sub_policies = num_sub_policies
        self.gaussian_fixed_var = gaussian_fixed_var
        self.hid_size = hid_size
        self.ob_rms = RunningMeanStd(shape=(self.obs.get_shape()[1],))
        obz = np.clip((self.obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = t.FloatTensor(obz)
        self.hiddenlayer = []
        self.lin = (nn.Linear(last_out, self.hid_size),nn.Tanh())
        #self.hiddenlayer = nn.Tanh(nn.Linear(last_out, self.hid_size))
        for layer in range(self.num_hidden_layers):
            last_out = self.hiddenlayer.append(self.lin)
        print(last_out)


    def forward(self, x):
        pass


def parallel_variance(avg_a, count_a, var_a, avg_b, count_b, var_b):
    delta = avg_b - avg_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    return M2 / (count_a + count_b - 1)