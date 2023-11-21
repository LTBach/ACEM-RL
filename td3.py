import os
import gym
from copy import deepcopy
import argparse

import pandas as pd
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory
from utils import *
from models import RLNN

USE_CUDA = T.cuda.is_available()

DEVICE = T.device("cuda:0" if USE_CUDA else "cpu")

if USE_CUDA:
    FloatTensor = T.cuda.FloatTensor
else:
    FloatTensor = T.FloatTensor

class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        if layer_norm:
            self.n1 = nn.LayerNorm(256)
            self.n2 = nn.LayerNorm(256)
        self.layer_norm = layer_norm

    def forward(self, x):
        
        if not self.layer_norm:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.max_action * F.tanh(self.l3(x))

        else:
            x = F.relu(self.n1(self.l1(x)))
            x = F.relu(self.n2(self.l2(x)))
            x = self.max_action * F.tanh(self.l3(x))

        return x
        

class Critic(RLNN):
    
    def __init__(self, state_dim, action_dim, max_action, layer_norm=False):
        super(Critic, self).__init__(state_dim, action_dim, max_action)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(256)
            self.n2 = nn.LayerNorm(256)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        if layer_norm:
            self.n4 = nn.LayerNorm(256)
            self.n5 = nn.LayerNorm(256)
        self.layer_norm = layer_norm

    def forward(self, x, u):
        
        if not self.layer_norm:
            x1 = F.relu(self.l1(T.cat([x, u], -1)))
            x1 = F.relu(self.l2(x1))
            x1 = self.l3(x1)

            x2 = F.relu(self.l4(T.cat([x, u], -1)))
            x2 = F.relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x1 = F.relu(self.n1(self.l1(T.cat([x, u], -1))))
            x1 = F.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

            x2 = F.relu(self.n4(self.l4(T.cat([x, u], -1))))
            x2 = F.relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2
    
    def Q1(self, x, u):
        if not self.layer_norm:
            x1 = F.relu(self.l1(T.cat([x, u], -1)))
            x1 = F.relu(self.l2(x1))
            x1 = self.l3(x1)
        
        else:
            x1 = F.relu(self.n1(self.l1(T.cat([x, u], -1))))
            x1 = F.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)
            
        return x1
    
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args):

        # actor
        self.actor = Actor(state_dim, action_dim, max_action,
                           layer_norm=args.layer_norm).to(DEVICE)
        self.actor_t = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)
        
        # critic
        self.critic = Critic(state_dim, action_dim, max_action,
                                layer_norm=args.layer_norm).to(DEVICE)
        self.critic_t = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # misc
        self.total_it = 0 
        self.criterion = nn.MSELoss()

        # noise
        self.noise_clip = args.noise_clip
        self.policy_noise = args.policy_noise

        # hyper-parameters
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.policy_freq = args.policy_freq


    def select_action(self, state):
        state = FloatTensor(state.reshape(1, -1)).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, memory, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, done = memory.sample(batch_size)

        with T.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                T.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_t(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_t(next_state, next_action)
            target_Q = T.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, output):
        self.actor.save(output, "actor")
        T.save(self.actor_optimizer.state_dict(), output + "/actor_optimizer")
        
        self.critic.save(output, "critic")
        T.save(self.critic_optimizer.state_dict(), output + "/critic_optimizer")
        
#         T.save(self.critic.state_dict(), filename + "_critic")
#         T.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

#         T.save(self.actor.state_dict(), filename + "_actor")
#         T.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.actor.load(filename, "actor")
        self.actor_optimizer.load_state_dict(T.load(filename + "/actor_optimizer"))
        self.actor_t = deepcopy(self.actor)
        
        self.critic.load(filename, "critic")
        self.critic_optimizer.load_state_dict(T.load(filename + "/critic_optimizer"))
        self.critic_t = deepcopy(self.critic)
#         self.critic.load_state_dict(T.load(filename + "_critic"))
#         self.critic_optimizer.load_state_dict(T.load(filename + "_critic_optimizer"))
#         self.critic_t = deepcopy(self.critic)

#         self.actor.load_state_dict(T.load(filename + "_actor"))
#         self.actor_optimizer.load_state_dict(T.load(filename + "_actor_optimizer"))
#         self.actor_t = deepcopy(self.actor)
        
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(seed=seed+100)
        done = False
        truncated = False
        while not done and not truncated:
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Enviroment parameters
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--env', default='HalfCheetah-v4', type=str)
    parser.add_argument('--start_timesteps', default=25e3, type=int)

    # Deep parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
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

    # Training parameters
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_timesteps', default=1e6, type=int)
    parser.add_argument('--mem_size', default=1e6, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--eval_freq', default=5e3, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    
    args.output = get_output_folder(args.output, args.env)
    with open(os.path.join(args.output, "parameters.txt"), "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key} = {value}\n")
    
    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.action_space.seed(args.seed)
    T.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    
    ############# Load
    
    
    memory = Memory(state_dim, action_dim)
    
    policy = TD3(state_dim, action_dim, max_action, args)



    # Start environment
    state, _ = env.reset(seed=args.seed)
    done = False
    truncated = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    df = pd.DataFrame(columns=["total_timesteps", "eval_score"])

    # Evaluate untrained policy
    res = {"total_timesteps": 0,
           "eval_score:": eval_policy(policy, args.env, args.seed)}
        
    df = df._append(res, ignore_index=True)
    
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t >= args.start_timesteps:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.gauss_sigma, size=action_dim)
            ).clip(-max_action, max_action)
        else:
            action = env.action_space.sample()

        # Perform action
        next_state, reward, done, truncated, _ = env.step(action) 
        done_bool = float(done) or float(truncated)

        # Store data in replay buffer
        memory.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(memory, args.batch_size)

        if done or truncated: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, _ = env.reset(seed=args.seed)
            done = False
            truncated = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0 or t + 1 == args.max_timesteps:
            res = {"total_timesteps": t + 1,
                   "eval_score:": eval_policy(policy, args.env, args.seed)}

            df = df._append(res, ignore_index=True)
            df.to_pickle(args.output + "/log.pkl")
            policy.save(args.output)
#             evaluations.append(eval_policy(policy, args.env, args.seed))
#             np.save(f"./results/{file_name}", evaluations)
#             if args.save_model: policy.save(f"./models/{file_name}")
