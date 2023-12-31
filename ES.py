import numpy as np
from copy import deepcopy

from Optimizers import Adam, BasicSGD

def compute_ranks(x):
    """
    Returns ranks in [0, len(x)]
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y

def compute_weight_decay(weight_decay, model_param_list):
    model_param_list = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_list * model_param_list, axis=1)

class VES(object):
    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0,
                 rank_fitness=True):
    
        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init

        # optimization stuff
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 ==0),  "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic:
            epsilon_half = np.random.randn(self.pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half])
        
        else:
            epsilon = np.random.randn(self.pop_size, self.num_params)
        
        return self.mu + epsilon * self.sigma
    
    def tell(self, scores, solutions):
        """
        Updates the ditribution
        """
        assert(len(scores) == 
               self.pop_size), "Inconsistent reward_table size reported."
        
        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay
    
        epsilon = (solutions - self.mu) / self.sigma
        grad = -1/(self.sigma * self.pop_size) * np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def get_distrib_params(self):
        """
        Returns the parameters of the distribution
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)
    

class GES(object):
    """
    Guided Evolution Strategies
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=10**-2,
                 alpha=0.5,
                 beta=2,
                 k=1,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0,
                 rank_fitness=False):

        # misc
        self.num_params = num_params
        self.first_interation = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.U = np.ones((self.num_params, k))

        # optimization stuff
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)
        self.beta = beta
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 ==0),  "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameterss
        """
        if self.antithetic:
            epsilon_half = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size // 2, self.num_params)
            epsilon_half += np.sqrt((1 - self.alpha) / self.k) * \
                np.random.randn(self.pop_size // 2, self.k) @ self.U.T
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size, self.num_params)
            epsilon += np.sqrt(1 - self.alpha) * \
                np.random.randn(self.pop_size, self.num_params) @ self.U.T

        return self.mu + epsilon * self.sigma

    def tell(self, scores, solutions):
        """
        Updates the distribution
        """
        assert(len(scores) ==
               self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -self.beta/(self.sigma * self.pop_size) * \
            np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def add(self, params, grads, fitness):
        """
        Adds new "gradient" to U
        """
        if params is not None:
            self.mu = params
        grads = grads / np.linalg.norm(grads)
        self.U[:, -1] = grads

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)

class sepCEM(object):
    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds

    def tell(self, scores, solutions):
        """
        Updates the distribution
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)


        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights @ (
            z * z) + self.damp * np.ones(self.num_params)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)
