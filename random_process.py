
import numpy as np


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhnlenbeck process
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

class GaussianNoise:
    """
    Simple Gaussian noise
    """

    def __init__(self, action_dim, sigma=0.2):
        self.action_dim = action_dim
        self.sigma = sigma

    def sample(self):
        s = np.random.normal(scale=self.sigma, size=self.action_dim)
        return s
