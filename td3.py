import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import gym.spaces

from models import Actor, CriticTD3
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from utils import *

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
        self.actor_t = Actor(
            state_dim, action_dim, max_action, layer_norm=args.layer_norm)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)
        
        # critic
        self.critic = CriticTD3(state_dim, action_dim,
                                layer_norm=args.layer_norm)
        self.critic_t = CriticTD3(
            state_dim, action_dim, layer_norm=args.layer_norm)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # cuda
        if USE_CUDA:
            self.actor = self.actor.to(DEVICE)
            self.actor_t = self.actor_t.to(DEVICE)
            self.critic = self.critic.to(DEVICE)
            self.critic_t = self.critic_t.to(DEVICE)

        # environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # misc
        self.steps = 0 
        self.criterion = nn.MSELoss()
        self.memory = memory

        # noise
        self.noise_clip = args.noise_clip
        self.policy_noise = args.policy_noise

        # hyper-parameters
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.policy_freq = args.policy_freq

    def select_action(self, state, noise=None):
        state = FloatTensor(
            state.reshape(-1, self.state_dim))
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise is not None:
            action += noise.sample()

        return np.clip(action, -self.max_action, self.max_action)
    
    def train(self, iterations=1):

        for it in range(iterations):
            
            # Sample replay buffer
            state, next_state, action, reward , done \
                = self.memory.sample(self.batch_size)

            # Select action according to policy add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            next_action = self.actor_t(next_state) + FloatTensor(noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with T.no_grad():
                target_Q1, target_Q2 = self.critic_t(
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
            if self.steps % self.policy_freq == 0:

                # Compute actor loss
                Q1, Q2 = self.critic(state, self.actor(state))
                actor_loss = -Q1.mean()

                # Update the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
                
            self.steps += 1
                
    def load(self, filename):
        self.actor.load_model(filename, "actor")
        self.actor_t.load_model(filename, "actor_t")
        self.critic.load_model(filename, "critic")
        self.critic_t.load_model(filename, "critic_t")
    
    def save(self, output):
        self.actor.save_model(output, "actor")
        self.actor_t.save_model(output, "actor_t")
        self.critic.save_model(output, "critic")
        self.critic_t.save_model(output, "critic_t")

def evaluate(agent, env, n_episodes=1):
    """
    Evaluate actor on a given number of runs
    """

    scores = []

    for _ in range(n_episodes):

        score = 0
        obs, _ = deepcopy(env.reset())
        truncated = False
        done = False

        while not truncated and not done:

            # get action
            action = agent.select_action(obs)

            # pass to environment
            n_obs, reward, done, truncated, _ = env.step(action)

            # update score and observation
            score += reward
            obs = n_obs

        scores.append(score)

    return np.mean(scores)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Enviroment parameters
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--env', default='HalfCheetah-v4', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # Deep parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--eval_and_save_per', default=100, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(os.path.join(args.output, "parameters.txt"), "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key} = {value}\n")
    
    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # agent
    agent = TD3(state_dim, action_dim, max_action, memory, args)

    # action noise
    if args.ou_noise:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)
    else:
        a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

    # training
    random = True
    steps = 0
    df = pd.DataFrame(columns=["steps", "score"])

    while steps < args.max_steps:
        
        score = 0
        obs, _ = env.reset()
        truncated = False
        done = False

        while not done and not truncated:

            if steps > args.start_steps:

                # get action
                action = agent.select_action(obs, a_noise)
            
            else:

                # get action
                action = env.action_space.sample()

            # pass action to env
            n_obs, reward, done, truncated, _ = env.step(action)
            done_bool = not done and not truncated

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))

            # if not warm up stage then update agent
            if steps > args.start_steps:
                agent.train()

            if steps % args.eval_and_save_per == 0:
                # eval
                eval_score = evaluate(agent, env)

                res = {"steps": steps,
                    "score": eval_score}
                df = df._append(res, ignore_index=True)
                print(res)

                # save log
                df.to_pickle(os.path.join(args.output, "log.pkl"))

                # save agent
                agent.save(args.output)

            # update score, steps and observation
            score += reward
            steps += 1
            obs = n_obs

        


