# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:42:09 2024

@author: Murad
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

relu = torch.nn.ReLU()
x = torch.arange(0, 3 * torch.pi, 0.1)
y = relu(torch.log(x - (0 - 1))) #+ relu(torch.log(x - (4 - 1))) + relu(torch.log(x - (6 - 1)))

plt.plot(x.numpy(), y.numpy())
plt.show()
