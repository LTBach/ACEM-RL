import os
import time
import math
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

from ES import sepCEM
from models import RLNN
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from utils import *

USE_CUDA = T.cuda.is_available()

DEVICE = T.device('cuda:0' if USE_CUDA else 'cpu')

if USE_CUDA:
    FloatTensor = T.cuda.FloatTensor
else:
    FloatTensor = T.FloatTensor

def evaluate(actor, env, n_episodes=1):
    """
    Evaluate actor on a given number of runs
    """

    scores = []
    
    for _ in range(n_episodes):

        score = 0
        obs, _ = env.reset()
        truncated = False
        done = False

        while not truncated and not done:

            # get action
            obs = FloatTensor(obs.reshape(-1))
            action = actor(obs).cpu().detach().numpy()

            # pass to environment
            n_obs, reward, done, truncated, _ = env.step(action)

            # update score and observation
            score += reward
            obs = n_obs
        
        scores.append(score)
    
    return np.mean(scores)



def find_target_action(action_dim, state, critic, sigma_init, damp, damp_limit, 
                       pop_size, antithetic, parents, elitism, use_td3=True, iterations=10):
    es = sepCEM(action_dim, sigma_init=sigma_init, damp=damp, damp_limit=damp_limit, 
                pop_size=pop_size, antithetic=antithetic, parents=parents, elitism=elitism)
    
    target_action = None
    Q_value = None

    for it in range(iterations):
        es_params = es.ask(pop_size)
        actions = to_tensor(deepcopy(es_params))

        # Evaluate action using critic
        if use_td3:
            with T.no_grad():
                critic_1, critic_2 = critic(state.repeat(actions.shape[0], 1), actions)
                fitness= T.max(critic_1, critic_2)

        else:
            with T.no_grad():
                fitness = critic_t(state.repeat(actions.shape[0], 1), actions)

        # Update es
        elite, elite_score = es.tell(to_numpy(T.squeeze(fitness)), es_params)

        if target_action is None:
            target_action = elite
            Q_value = elite_score
        
        elif Q_value <= elite_score:
            target_action = elite
            Q_value = elite_score
            
    return target_action, Q_value

class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        # es stuff
        self.sigma_init = args.sigma_init
        self.damp = args.damp
        self.damp_limit = args.damp_limit
        self.pop_size = args.pop_size
        self.antithetic = not args.pop_size % 2
        self.parents = args.pop_size // 2
        self.elitism = args.elitism

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.use_td3 = args.use_td3
        self.mask = args.mask

    def forward(self, x):
        
        if not self.layer_norm:
            x = F.tanh(self.l1(x))
            x = F.tanh(self.l2(x))
            x = self.max_action * F.tanh(self.l3(x))

        else:
            x = F.tanh(self.n1(self.l1(x)))
            x = F.tanh(self.n2(self.l2(x)))
            x = self.max_action * F.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t, iterations=10):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        es_es = [sepCEM(self.action_dim, sigma_init=self.sigma_init, damp=self.damp, 
                        damp_limit=self.damp_limit, pop_size=self.pop_size, 
                        antithetic=self.antithetic, parents=self.parents, 
                        elitism=self.elitism) for state in states]

        tar_actions = np.zeros((states.shape[0], self.action_dim))
        Q_values = np.zeros(states.shape[0])

        for _ in range(iterations):
            es_params = [es.ask(es.pop_size) for es in es_es]
            actions = to_tensor(deepcopy(es_params))
            states_for_es = states.unsqueeze(1).repeat(1, 10, 1)

            # Evaluate action using critic
            if self.use_td3:
                with T.no_grad():
                    critic_1, critic_2 = critic(states_for_es, actions)
                    fitness= T.min(critic_1, critic_2)

            else:
                with T.no_grad():
                    fitness = critic_t(states_for_es, actions)

            # Update es
            for idx, es in enumerate(es_es):
                es.tell(to_numpy(T.squeeze(fitness[idx])), es_params[idx])

                if not tar_actions.all():
                    tar_actions[idx] = es.elite
                    Q_values[idx] = es.elite_score
                
                elif Q_values[idx] <= es.elite_score:
                    tar_actions[idx] = es.elite
                    Q_values[idx] = es.elite_score
        
        actor_actions = self(states) 
        if self.mask:
            if self.use_td3:
                with T.no_grad():
                    Q_values_actor_1, Q_values_actor_2 = critic(states, actor_actions)
                    Q_values_actor = T.min(Q_values_actor_1, Q_values_actor_2)
            else:
                with T.no_grad():
                    Q_values_actor = critic(states, actor_actions)
            
            mask = (to_tensor(Q_values).reshape(-1, 1) > Q_values_actor).repeat(1, self.action_dim)

            # Compute actor loss
            actor_loss = nn.MSELoss()(actor_actions*mask, to_tensor(tar_actions)*mask)
        
        else:
            # Compute actor loss
            actor_loss = nn.MSELoss()(actor_actions, to_tensor(tar_actions))

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
#         nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        
        self.layer_norm = args.layer_norm
        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount

    def forward(self, x, u):
        
        if not self.layer_norm:
            x = F.leaky_relu(self.l1(T.cat([x, u], -1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(T.cat([x, u], -1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)
        
        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with T.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
            
class CriticTD3(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, max_action)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):
        
        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(T.cat([x, u], -1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.leaky_relu(self.n1(self.l1(T.cat([x, u], -1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(T.cat([x, u], -1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(T.cat([x, u], -1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add cliped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
        
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-self.max_action, self.max_action)

        # Q target = reward + discount * min_i(Q_i(next_state, pi(next_state)))
        with T.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = T.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) \
            + nn.MSELoss()(current_Q2, target_Q)
        
        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--mask', dest='mask', action='store_true')

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
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
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
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--eval_and_save_per', default=100, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key} = {value}\n")
    
    # enviroment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    actor = Actor(state_dim, action_dim, max_action, args)
    actor_t = Actor(state_dim, action_dim, max_action, args)
    actor_t.load_state_dict(actor.state_dict())

    # action noise
    if not args.ou_noise:
        a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)
    else:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)
    
    if USE_CUDA:
        critic.to(DEVICE)
        critic_t.to(DEVICE)
        actor.to(DEVICE)
        actor_t.to(DEVICE)
    
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
                obs = FloatTensor(obs.reshape(-1))
                action = actor(obs).cpu().detach().numpy()
                if a_noise is not None:
                    action += a_noise.sample()
                action = np.clip(action, -max_action, max_action)
                print('action_____:', action)
            else:

                # get action 
                action = env.action_space.sample()

            # pass action to env
            n_obs, reward, done, truncated, _ = env.step(action)
            done_bool = not done and not truncated

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))

            # if not warm up stage then update actor, critic
            if steps > args.start_steps:
                # critic update
                critic.update(memory, args.batch_size, actor_t, critic_t)

                # actor update
                if steps % args.policy_freq == 0:
                    actor.update(memory, args.batch_size, critic, actor_t)
            
            # save stuff
            if steps % args.eval_and_save_per == 0:

                # eval
                eval_score = evaluate(actor, env)
            
                res = {"steps": steps,
                       "score": eval_score}
                df = df._append(res, ignore_index=True)
                print(res)
                
                # save log
                df.to_pickle(os.path.join(args.output, "log.pkl"))

                # save actor
                actor.save_model(args.output, "actor")

                # save critic
                critic.save_model(args.output, "critic")
                

            # update score, steps and obsevation
            score += reward
            steps += 1
            obs = n_obs
