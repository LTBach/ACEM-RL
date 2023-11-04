import os
import numpy as np
from tqdm import tqdm

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = T.cuda.is_available()

DEVICE = T.device('cuda:0' if USE_CUDA else 'cpu')

if USE_CUDA:
    FloatTensor = T.cuda.FloatTensor
else:
    FloatTensor = T.FloatTensor

class Memory(object):

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.states = T.zeros(self.memory_size, self.state_dim).to(DEVICE)
        self.actions = T.zeros(self.memory_size, self.action_dim).to(DEVICE)
        self.n_states = T.zeros(self.memory_size, self.state_dim).to(DEVICE)
        self.rewards = T.zeros(self.memory_size, 1).to(DEVICE)
        self.dones = T.zeros(self.memory_size, 1).to(DEVICE)

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos
    
    def get_pos(self):
        return self.pos
    
        # Expects tuples of (state, next_state, action, reward, done)

    def add(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos] = FloatTensor(state)
        self.n_states[self.pos] = FloatTensor(n_state)
        self.actions[self.pos] = FloatTensor(action)
        self.rewards[self.pos] = FloatTensor([reward])
        self.dones[self.pos] = FloatTensor([done])

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        upper_bound = self.memory_size if self.full else self.pos
        batch_inds = T.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))
        
        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])
    
    def get_reward(self, start_pos, end_pos):

        tmp = 0
        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):
                tmp += self.rewards[i]
        else:
            for i in range(start_pos, self.memory_size):
                tmp += self.rewards[i]

            for i in range(end_pos):
                tmp += self.rewards[i]

        return tmp
    
    def repeat(self, start_pos, end_pos):

        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.rewards[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0

        else:
            for i in range(start_pos, self.memory_size):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()
            
            self.pos += 1
            if self.pos == self.memory_size:
                self.full = True
                self.pos = 0

            for i in range(end_pos):
                
                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.actions[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

            self.pos += 1
            if self.pos == self.memory_size:
                self.full = True
                self.pos = 0
