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
from modular_legs.sim.evolution.vae.qlogei import QLogEI


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    dtype,
    device,
    n_candidates=None,  # Number of candidates for Thompson sampling
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():  # We don't need gradients when using TS
        X_next = thompson_sampling(X_cand, num_samples=batch_size)


    return X_next

class TuRBO(QLogEI):
    '''
        A class to perform Bayesian optimization using qLogEI
        It should be asynchronous, allowing 'ask' and 'tell' to be called in any order
    '''


    def __init__(self, dim=8, n_init=20, lb=-4., ub=4., seed=0, candidate_buffer_size=12):
        super(TuRBO, self).__init__(dim, n_init, lb, ub, seed, candidate_buffer_size)

        self.batch_size = 1 # Sequential optimization is fine?
        self.state = TurboState(dim, batch_size=self.batch_size, best_value=-float("inf"))

        self.X_turbo = None
        self.Y_turbo = None



            

    # def get_initial_points(self):
    #     '''
    #     Get initial points for the optimization
    #     Return a list for easily working in a non-pytorch environment
    #     '''
    #     sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
    #     X_init = sobol.draw(n=self.n_init).to(dtype=self.dtype, device=self.device)
    #     return X_init.tolist()
    
    # def _unnormalize(self, x):
    #     return unnormalize(x, self.bounds)
    
    # def _normalize(self, x):
    #     return normalize(x, self.bounds)

    def _reset(self):
        self.X_turbo = None
        self.Y_turbo = None
        self.state = TurboState(self.dim, batch_size=self.batch_size, best_value=-float("inf"))
        self.init_xs = self.get_initial_points()

    def ask(self):
        '''
        Ask the optimizer for the next point to evaluate
        '''

        if self.state.restart_triggered:
            print("Restarting the optimizer!! ")
            self._reset()

        # If there are initial points, return them
        if self.init_xs:
            print("Returning initial points")
            return self.init_xs.pop()

        # if len(self.candidate_buffer) == 0:
        # New data has been added or the buffer is empty
        # Create a batch
        # ei = qLogExpectedImprovement(self.model, self.train_Y.max())
        # print("Optimizing acquisition function")
        # candidate, acq_value = optimize_acqf(
        #                         ei,
        #                         bounds=torch.stack(
        #                             [
        #                                 torch.zeros(self.dim, dtype=self.dtype, device=self.device),
        #                                 torch.ones(self.dim, dtype=self.dtype, device=self.device),
        #                             ]
        #                         ),
        #                         q=self.candidate_buffer_size,
        #                         num_restarts=10,
        #                         raw_samples=512,
        #                         sequential=True
        #                     )

        

        
        with gpytorch.settings.max_cholesky_size(float("inf")):
            candidate = generate_batch(
                state=self.state,
                model=self.model,
                X=self.X_turbo,
                Y=self.train_Y,
                batch_size=self.batch_size,
                n_candidates=min(5000, max(2000, 200 * self.dim)),
                acqf="ts",
                dtype=self.dtype,
                device=self.device,
            )

        candidate = self._unnormalize(candidate)
        suggestion =  candidate[0].tolist()
        # acq_value = acq_value.tolist()

        # self.candidate_buffer = [a for _, a in sorted(zip(acq_value, candidate_list), reverse=True)]


        # suggestion = self.candidate_buffer.pop()
        print(f"Suggestion: {suggestion}")
        return suggestion
    
    def tell(self, x: list, y: float, minimize=True):
        '''
        Tell the optimizer the result of the evaluation in a sequential manner
        '''
        if minimize:
            y = -y
        x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device).unsqueeze(0)
        assert len(x_tensor.shape) == 2, "x should be a 2D tensor"
        x_tensor = self._normalize(x_tensor)

        # Append data
        if self.X_turbo is None:
            self.X_turbo = x_tensor
        else:
            self.X_turbo = torch.cat((self.X_turbo, x_tensor), axis=0)
        y_tensor = torch.tensor(y, dtype=self.dtype, device=self.device).unsqueeze(-1).unsqueeze(-1)
        if self.Y_turbo is None:
            self.Y_turbo = y_tensor
        else:
            self.Y_turbo = torch.cat((self.Y_turbo, y_tensor), axis=0)

        # Update state
        # Note: make sure that this Y should comes from the X that generated by optimizing the acquisition function
        if len(self.Y_turbo) > self.n_init:
            self.state = update_state(state=self.state, Y_next=y_tensor)


        # Update the model
        if len(self.Y_turbo) >= self.n_init/3:
            self.train_Y = (self.Y_turbo - self.Y_turbo.mean()) / self.Y_turbo.std()
            # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0)
                )
            )

            self.model = SingleTaskGP(self.X_turbo, self.train_Y, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

            with gpytorch.settings.max_cholesky_size(float("inf")):
                # Fit the model
                fit_gpytorch_mll(mll)

            # self.candidate_buffer = []

        # Print current status
        print(f"{len(self.X_turbo)}) Best value: {self.Y_turbo.max().item():.2e}")


    



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
    import nevergrad as ng
    qlogei = TuRBO()
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
    qlogei = TuRBO()

    # X_ei = qlogei.get_initial_points()
    # Y_ei = [eval_objective(x) for x in X_ei]
    # [qlogei.tell(x, y, minimize=False) for x, y in zip(X_ei, Y_ei)]

    for i in range(1000):
        print(f"Step {i}")
        x = qlogei.ask()
        y = eval_objective(x)
        qlogei.tell(x, y, minimize=False)
        print(f"Best value: {qlogei.Y_turbo.max().item():.2e}")
        print(f"Best point: {qlogei._unnormalize(qlogei.X_turbo[qlogei.Y_turbo.argmax()])}")
        print("")

if __name__ == "__main__": 
    test_loop()