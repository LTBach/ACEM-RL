import os
import numpy as np
from tqdm import tqdm

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = T.cuda.is_available()

DEVICE = T.device("cuda:0" if USE_CUDA else "cpu")

if USE_CUDA:
    FloatTensor = T.cuda.FloatTensor
else:
    FloatTensor = T.FloatTensor
    
class Memory(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = T.zeros(max_size, state_dim, device=DEVICE)
        self.actions = T.zeros(max_size, action_dim, device=DEVICE)
        self.next_states = T.zeros(max_size, state_dim, device=DEVICE)
        self.rewards = T.zeros(max_size, 1, device=DEVICE)
        self.dones = T.zeros(max_size, 1, device=DEVICE)


    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = FloatTensor(state)
        self.actions[self.ptr] = FloatTensor(action)
        self.next_states[self.ptr] = FloatTensor(next_state)
        self.rewards[self.ptr] = FloatTensor([reward])
        self.dones[self.ptr] = FloatTensor([done])

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (self.states[ind],
                self.actions[ind],
                self.next_states[ind],
                self.rewards[ind],
                self.dones[ind])
