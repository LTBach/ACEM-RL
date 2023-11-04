import numpy as np


class Optimizer(object):

    def __init__(self, epsilon=1e-08):
        self.epsilon = epsilon

    def step(self, grad):
        raise NotImplementedError
    

class BasicSGD(Optimizer):
    """
    Standard gradient descent
    """

    def __init__(self, step_size):
        super(BasicSGD, self).__init__()
        self.step_size = step_size

    def step(self, grad):
        step = -self.step_size * grad
        return step
    

class SGD(Optimizer):
    """
    Gradient descent with momentum
    """

    def __init__(self, step_size, momentum=0.9):
        super(SGD, self).__init__()
        self.step_size = step_size

    def step(self, grad):
        if not hasattr(self, "v"):
            self.v = np.zeros(grad.shape[0], dtype=np.float32)
        self.v = self.momentum * self.v + (1. - self.momentum) * grad
        step = -self.step_size * self.v
        return step
    
class Adam(Optimizer):
    """
    Adam optimizer
    """

    def __init__(self, step_size, beta_1=0.99, beta_2=0.999):
        super(Adam, self).__init__()
        self.step_size = step_size
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 0

    def step(self, grad):

        if not hasattr(self, "m"):
            self.m = np.zeros(grad.shape[0], dtype=np.float32)
        if not hasattr(self, "v"):
            self.v = np.zeros(grad.shape[0], dtype=np.float32)
        
        self.t += 1
        a = self.step_size * np.sqrt(1 - self.bata_2 ** 
                                     self.t) / (1 - self.beta_1 ** self.t)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.b = self.beta_2 * self.v + (1 - self.beta_2) * (grad * grad)
        step = -a * self.m / (np.sqrt.v + self.epsilon) 

        return step
