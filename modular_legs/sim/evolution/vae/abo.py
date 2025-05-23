
import os
import pdb
import pickle
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # disable GPU
from trieste.objectives import ScaledBranin
from trieste.space import Box
from trieste.data import Dataset
import gpflow
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
# these imports will be used later for optimization
from trieste.acquisition import LocalPenalization
from trieste.acquisition.rule import (
    AsynchronousGreedy,
    EfficientGlobalOptimization,
)
from trieste.ask_tell_optimization import AskTellOptimizer

class AsynchronousBO:

    def __init__(self, dim=8, n_init=3, lb=-4., ub=4., load_gp=None, log_dir=None, likelihood_variance=1e-7):

        self.dim = dim
        self.n_init = n_init
        self.load_gp = load_gp
        self.log_dir = log_dir
        self.likelihood_variance = likelihood_variance
        os.makedirs(self.log_dir, exist_ok=True)
        self.search_space = Box([lb]*dim, [ub]*dim)
        self.pending_points = []

        if self.load_gp is None:
            self.initial_query_points = self.search_space.sample(n_init)
            self.initial_point_queue = self.initial_query_points.numpy().tolist()
            self.gp_inited = False
        else:
            # Load the GP from a checkpoint
            print(f"Loading GP from {self.load_gp}")
            with open(self.load_gp, 'rb') as f:
                loaded_state = pickle.load(f)
            self.async_bo = AskTellOptimizer.from_record(loaded_state, self.search_space)
            self.gp_inited = True
            # Load the pending points from the loaded state
            self.initial_query_points = self.async_bo.acquisition_state.pending_points
            self.initial_point_queue = self.initial_query_points.numpy().tolist()
            while len(self.initial_point_queue) < n_init:
                # If the loaded pending points has less than n_init points, ask for more
                self.initial_point_queue.append(self.ask())
            while len(self.initial_point_queue) > n_init:
                # If the loaded pending points has more than n_init points, pop the last one
                # Points in pending_points will be popped in the later ask
                self.pending_points.append(self.initial_point_queue.pop())
            

        self.num_told = 0

    def _save_gp(self, name=None):
        if name is None:
            with open(os.path.join(self.log_dir, "last_gp_model.pkl"), 'wb') as f:
                state = self.async_bo.to_record()
                pickle.dump(state, f)
            if self.num_told % 10 == 0:
                with open(os.path.join(self.log_dir, f"gp_model_{self.num_told}.pkl"), 'wb') as f:
                    state = self.async_bo.to_record()
                    pickle.dump(state, f)
        else:
            with open(os.path.join(self.log_dir, name), 'wb') as f:
                state = self.async_bo.to_record()
                pickle.dump(state, f)

    def _init_gp(self):
        
        initial_data = Dataset(
            query_points=self.initial_query_points,
            observations=self.initial_observations,
        )

        gpflow_model = build_gpr(initial_data, self.search_space, likelihood_variance=self.likelihood_variance)
        self.model = GaussianProcessRegression(gpflow_model)

        local_penalization_acq = LocalPenalization(self.search_space, num_samples=8000)
        local_penalization_rule = AsynchronousGreedy(builder=local_penalization_acq)  # type: ignore

        self.async_bo = AskTellOptimizer(
            self.search_space, initial_data, self.model, local_penalization_rule
        )



        self.gp_inited = True

    def ask_init(self):
        # The first n_init points should be tested syncghronously!
        return self.initial_point_queue
    
    def tell_init(self, ys_in: list, minimize=True):
        '''
        Tell the optimizer the result of the evaluation in a BATCH manner
        '''
        if not minimize:
            ys = [-y for y in ys_in]
        else:
            ys = ys_in.copy()

        ys = [[y] for y in ys]

        if not self.gp_inited:
            self.initial_observations = tf.constant(ys, dtype=tf.float64)
            self._init_gp()
        else:
            # A checkpint has been loaded
            for x, ys in zip(self.initial_point_queue, ys):
                self.tell(x, ys[0], minimize=minimize)

        self.num_told += self.n_init
        self._save_gp()


    def ask(self):
            
        assert self.gp_inited, "GP not initialized"

        if self.pending_points:
            raw_point = self.pending_points.pop()
        else:
            point = self.async_bo.ask()
            raw_point = point.numpy().tolist()[0]

        return raw_point



    def tell(self, x: list, y: float, minimize=True):
        '''
        Tell the optimizer the result of the evaluation in a sequential manner
        '''
        if not minimize:
            # trieste assumes minimization
            y = -y


        new_data = Dataset(
            query_points=tf.constant([x], dtype=tf.float64),
            observations=tf.constant([[y]], dtype=tf.float64),
        )

        self.async_bo.tell(new_data)

        self.num_told += 1
        self._save_gp()




if __name__ == "__main__":
    log_dir = "/test_log_dir"
    bo = AsynchronousBO(n_init=3, log_dir=log_dir)
    x = bo.ask_init()
    print("Got points: ", x)
    bo.tell_init([1, 2, 3])
    for _ in range (12):
        x1 = bo.ask()
        print("Got point: ", x1)
        bo.tell(x1, 1)
    [bo.ask() for _ in range(5)]
    
    print("Pending points: ", bo.async_bo.acquisition_state.pending_points)
    bo._save_gp() # save all the pending points for testing
    pdb.set_trace()


    load_file = os.path.join(log_dir, "last_gp_model.pkl")
    bo = AsynchronousBO(n_init=3, log_dir=log_dir, load_gp=load_file)
    pdb.set_trace()
    x = bo.ask_init()
    print("Got points: ", x)