import os
import math
import pdb
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
     

class QLogEI():
    '''
        A class to perform Bayesian optimization using qLogEI
        It should be asynchronous, allowing 'ask' and 'tell' to be called in any order
    '''


    def __init__(self, dim=8, n_init=50, lb=-4., ub=4., seed=0, candidate_buffer_size=12):
        self.device = torch.device("cpu") # TODO torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        self.dim = dim
        self.bounds = torch.tensor([[lb]*self.dim, [ub]*self.dim], dtype=torch.float64)
        self.n_init = n_init

        batch_size = 1
        self.seed = seed
        torch.manual_seed(self.seed)

        self.X_ei = None
        self.Y_ei = None

        self.candidate_buffer_size = candidate_buffer_size
        self.candidate_buffer = []

        self.init_xs = self.get_initial_points()

        # while len(Y_ei) < 1000:
        #     train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
        #     # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        #     model = SingleTaskGP(X_ei, train_Y)
        #     mll = ExactMarginalLogLikelihood(model.likelihood, model)
        #     fit_gpytorch_mll(mll)

        #     # Create a batch
        #     ei = qExpectedImprovement(model, train_Y.max())
        #     candidate, acq_value = optimize_acqf(
        #                             ei,
        #                             bounds=torch.stack(
        #                                 [
        #                                     torch.zeros(dim, dtype=dtype, device=device),
        #                                     torch.ones(dim, dtype=dtype, device=device),
        #                                 ]
        #                             ),
        #                             q=batch_size,
        #                             num_restarts=10,
        #                             raw_samples=512,
        #                         )
        #     print("Candidate: ", candidate)
        #     Y_next = torch.tensor(
        #                             [eval_objective(x) for x in candidate], dtype=dtype, device=device
        #                         ).unsqueeze(-1)

        #     # Append data
        #     X_ei = torch.cat((X_ei, candidate), axis=0)
        #     Y_ei = torch.cat((Y_ei, Y_next), axis=0)

            

    def get_initial_points(self):
        '''
        Get initial points for the optimization
        Return a list for easily working in a non-pytorch environment
        '''
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        X_init = sobol.draw(n=self.n_init).to(dtype=self.dtype, device=self.device)
        X_init = self._unnormalize(X_init)
        return X_init.tolist()
    
    def _unnormalize(self, x):
        return unnormalize(x, self.bounds)
    
    def _normalize(self, x):
        return normalize(x, self.bounds)
    
    def tell(self, x: list, y: float, minimize=True):
        '''
        Tell the optimizer the result of the evaluation in a sequential manner
        '''
        if minimize:
            y = -y
        x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device).unsqueeze(0)
        assert len(x_tensor.shape) == 2, "x should be a 2D tensor"
        x_tensor = self._normalize(x_tensor)
        if self.X_ei is None:
            self.X_ei = x_tensor
        else:
            self.X_ei = torch.cat((self.X_ei, x_tensor), axis=0)

        y_tensor = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1).unsqueeze(-1)
        if self.Y_ei is None:
            self.Y_ei = y_tensor
        else:
            self.Y_ei = torch.cat((self.Y_ei, y_tensor), axis=0)

        # Update the model
        if len(self.Y_ei) >= self.n_init/3:
            self.train_Y = (self.Y_ei - self.Y_ei.mean()) / self.Y_ei.std()
            # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            self.model = SingleTaskGP(self.X_ei, self.train_Y)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)

            self.candidate_buffer = []

        # Print current status
        print(f"{len(self.X_ei)}) Best value: {self.Y_ei.max().item():.2e}")


    def ask(self):
        '''
        Ask the optimizer for the next point to evaluate
        '''
        # If there are initial points, return them
        if self.init_xs:
            return self.init_xs.pop()

        if len(self.candidate_buffer) == 0:
            # New data has been added or the buffer is empty
            # Create a batch
            ei = qLogExpectedImprovement(self.model, self.train_Y.max())
            # print("Optimizing acquisition function")
            candidate, acq_value = optimize_acqf(
                                    ei,
                                    bounds=torch.stack(
                                        [
                                            torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                                            torch.ones(self.dim, dtype=self.dtype, device=self.device),
                                        ]
                                    ),
                                    q=self.candidate_buffer_size,
                                    num_restarts=10,
                                    raw_samples=512,
                                    sequential=True
                                )
            # print("Candidate: ", candidate)
            candidate = self._unnormalize(candidate)
            candidate_list =  candidate.tolist()
            acq_value = acq_value.tolist()

            self.candidate_buffer = [a for _, a in sorted(zip(acq_value, candidate_list), reverse=True)]


        suggestion = self.candidate_buffer.pop()
        print(f"Suggestion: {suggestion}")
        return suggestion



def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    '''botorch assumes a maximization problem'''
    y = -np.sum(np.square(x))
    return y   


def eval_objective(x):
    """
    Compute the 8D Ackley function.

    Parameters:
    x : array-like
        Input array with shape (8,).

    Returns:
    float
        The value of the Ackley function at the input.
    """
    # Ensure that x is a numpy array
    x = np.asarray(x)
    assert x.shape == (8,), "Input must be a vector of length 8"

    # Constants for the Ackley function
    a = 20
    b = 0.2
    c = 2 * np.pi

    # Compute the sum of squares
    sum_sq_term = np.sum(x**2)
    sum_sq_term = np.sqrt(sum_sq_term / 8)

    # Compute the cosine sum term
    cos_term = np.sum(np.cos(c * x)) / 8

    # Final Ackley function computation
    result = -a * np.exp(-b * sum_sq_term) - np.exp(cos_term) + a + np.e
    
    return -result




    
def test_loop():
    qlogei = QLogEI()

    X_ei = qlogei.get_initial_points()
    Y_ei = [eval_objective(x) for x in X_ei]
    [qlogei.tell(x, y, minimize=False) for x, y in zip(X_ei, Y_ei)]

    for i in range(1000):
        x = qlogei.ask()
        y = eval_objective(x)
        qlogei.tell(x, y)
        print(f"Best value: {qlogei.Y_ei.max().item():.2e}")
        print(f"Best point: {qlogei._unnormalize(qlogei.X_ei[qlogei.Y_ei.argmax()])}")
        print("")


def test_loop():
    import nevergrad as ng
    qlogei = QLogEI()
    param = ng.p.Array(shape=(8,), lower=-4, upper=4)
    bboptimizer = ng.optimizers.NGOpt(parametrization=param, budget=5000, num_workers=1)
    param_dict = {}


    

    

    X_ei = qlogei.get_initial_points()
    Y_ei = [eval_objective(x) for x in X_ei]
    [qlogei.tell(x, y) for x, y in zip(X_ei, Y_ei)]

    for _ in range(50):
        param = bboptimizer.ask()
        x = param.value
        y = eval_objective(x)
        bboptimizer.tell(param, -y)


    best_ei = -np.inf
    best_cma = -np.inf

    for i in range(1000):
        # BO
        x = qlogei.ask()
        y = eval_objective(x)
        qlogei.tell(x, y)
        best_ei = max(best_ei, y)
        print("[BO] Best value: ", best_ei)

        # CMA
        param = bboptimizer.ask()
        x = param.value
        y = eval_objective(x)
        bboptimizer.tell(param, -y)
        best_cma = max(best_cma, y)
        print("[CMA] Best value: ", best_cma)


def test_loop():
    qlogei = QLogEI()

    # X_ei = qlogei.get_initial_points()
    # Y_ei = [eval_objective(x) for x in X_ei]
    # [qlogei.tell(x, y, minimize=False) for x, y in zip(X_ei, Y_ei)]

    for i in range(1000):
        print(f"Step {i}")
        x = qlogei.ask()
        y = eval_objective(x)
        qlogei.tell(x, y, minimize=False)
        print(f"Best value: {qlogei.Y_ei.max().item():.2e}")
        print(f"Best point: {qlogei._unnormalize(qlogei.X_ei[qlogei.Y_ei.argmax()])}")
        print("")

if __name__ == "__main__": 
    test_loop()