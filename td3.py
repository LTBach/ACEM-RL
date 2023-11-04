import os
import numpy as np
from tqdm import tqdm

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import Actor, CriticTD3

USE_CUDA = T.cuda.is_available()

DEVICE = T.device('cuda:0' if USE_CUDA else 'cpu')

if USE_CUDA:
    FloatTensor = T.cuda.FloatTensor
else:
    FloatTensor = T.FloatTensor

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, memory, args):

        # actor
        self.actor = Actor(state_dim, action_dim, max_action,
                           layer_norm=args.layer_norm)
        self.actor_target = Actor(
            state_dim, action_dim, max_action, layer_norm=args.layer_norm)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)
        
        # critic
        self.critic = CriticTD3(state_dim, action_dim,
                                layer_norm=args.layer_norm)
        self.critic_target = CriticTD3(
            state_dim, action_dim, layer_norm=args.layer_norm)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # cuda
        if USE_CUDA:
            self.actor = self.actor.to(DEVICE)
            self.actor_target = self.actor_target.to(DEVICE)
            self.critic = self.critic.to(DEVICE)
            self.critic_target = self.critic_target.to(DEVICE)

        # misc
        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = memory

        # hyper-parameters
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq

    def select_action(self, state, noise=None):
        state = FloatTensor(
            state.reshape(-1, self.state_dim))
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise is not None:
            action += noise.sample()

        return np.clip(action, -self.max_action, self.max_action)
    
    def train(self, iterations):

        for it in tqdm(range(iterations)):
            
            # Sample replay buffer
            x, y, u, r ,d = self.memory.sample(self.batch_size)
            state = FloatTensor(x)
            next_state = FloatTensor(y)
            action = FloatTensor(u)
            reward = FloatTensor(r)
            done = FloatTensor(1 - d)

            # Select action according to policy add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state) + FloatTensor(noise)
            next_action = next_action.clampe(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with T.no_grad():
                target_Q1, target_Q2 = self.critic_target(
                    next_state, next_action)
                target_Q = T.min(target_Q1, target_Q2)
                target_Q = reward + (done * self.discount * target_Q)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.criterion(current_Q2, target_Q) + \
                self.criterion(current_Q2, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                Q1, Q2 = self.critic(state, self.actor(state))
                actor_loss = -Q1.mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.date))

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
                
    def load(self, filename):
        self.actor.load_model(filename, "actor")
        self.critic.load_model(filename, "critic")
    
    def save(self, output):
        self.actor.save_model(output, "actor")
        self.critic.save_model(output, "critic")
