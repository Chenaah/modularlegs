





import argparse
import copy
import datetime
from functools import partial
import os
import pdb
import pickle
import random
import re
import time
import numpy as np
from omegaconf import OmegaConf
import wandb
# import zmq
import socket
from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.evolution.mutation_meta import crossover, extend_random_design, mutate, random_gen
from modular_legs.sim.evolution.pose_optimizer import optimize_pose
from modular_legs.sim.evolution.run_server import start_server
from modular_legs.sim.evolution.utils import CSVLogger, gen_log_dir_5x1, gen_log_dir_asym, is_metapipeline_valid
from modular_legs.sim.robot_designer import RobotDesigner
from modular_legs.sim.robot_metadesigner import MetaDesignerAsym
from modular_legs.utils.files import load_cfg



def get_best_designs(d, n):
    sorted_items = sorted(d.items(), key=lambda item: item[1][1], reverse=True)
    keys_of_highest_y = [key for key, value in sorted_items[:n]]
    
    return keys_of_highest_y

def update_run_name(s):
    # Define a function to increment the matched number
    def increment(match):
        return f"{match.group(1)}{int(match.group(2)) + 1}{match.group(3)}"
    
    # Use regex to find and increment the second number
    updated_string = re.sub(r'(\D+\d+\D+)(\d+)(\D*)', increment, s)
    
    return updated_string

def are_ok(xs, self_collision_threshold, ave_speed_threshold):
    # Constraint function
    valid = []
    for design_pipeline in xs:
        valid.append(
            is_metapipeline_valid(design_pipeline, 
                                level=2, 
                                conf_dict={"self_collision_threshold": self_collision_threshold, "ave_speed_threshold": ave_speed_threshold}
                                )
            )

    not_empty = [x != [] for x in xs]

    return [a and b for a, b in zip(valid, not_empty)]


# pop_size = 18
# n_delete = 3
n_pool_a = 3
rate_pool_b = 0.5
# mutate_rate = 0.3
# n_gen = 100

max_n_modules = 6
max_design_length = (max_n_modules-1)*5+1
min_n_modules = 2
min_design_length = (min_n_modules-1)*5+1

ini_xs = []

ini_ys = []


def birth(pop_dict, g, max_depth=100, _depth=0):

    if _depth > max_depth:
        print("Max depth reached")
        return None

    mating_pool_a = get_best_designs(pop_dict, n_pool_a)
    mating_pool_b = get_best_designs(pop_dict, int(rate_pool_b*len(pop_dict)))
    design_a_id = np.random.choice(mating_pool_a)
    design_b_id = np.random.choice([b for b in mating_pool_b if b != design_a_id])
    design_a = pop_dict[design_a_id][0]
    design_b = pop_dict[design_b_id][0]

    
    rand = np.random.rand()
    if rand < 0.5:
        # crossover
        new_design = crossover(design_a, design_b, constraint_func=g, max_depth=max_depth, return_single=True)
    else:
        new_design = random.choice([design_a, design_b])
        
    if rand > 0.25:
        # mutate
        if len(new_design) >= max_design_length:
            mutate_type = np.random.choice(["mutate_limb", "delete_limb"])
        elif len(new_design) <= min_design_length:
            mutate_type = np.random.choice(["mutate_limb", "grow_limb"])
        else:
            mutate_type = None
        new_design = mutate(new_design, mutate_type=mutate_type, constraint_func=g, max_depth=max_depth)

    if new_design in [pop_dict[key][0] for key in pop_dict]:
        print("Design already exists!")
        return birth(pop_dict, g, max_depth=max_depth, _depth=_depth+1)

    return new_design


def gen(pop_dict, g, max_depth=100, _depth=0):
    
    add_pipline_length = np.random.randint(min_n_modules, max_n_modules+1)
    random_design = random_gen(add_pipline_length, g, max_depth)

    if random_design in [pop_dict[key][0] for key in pop_dict]:
        print("Design already exists!")
        return gen(pop_dict, g, max_depth=max_depth, _depth=_depth+1)

    return random_design


class Worker(object):
    def __init__(self, address, client_sock, worker_id, cuda=0):
        self.address = address
        self.client_sock = client_sock
        self.worker_id = worker_id
        self.port = address[1]
        self.cuda = cuda
        # self.context = context
        # self.socket = self.context.socket(zmq.REQ)
        # self.socket.connect(address)
        # self.poller = poller
        # self.poller.register(self.socket, zmq.POLLIN)
        self.train_conf = None
        self.design_pipeline = None
        self.run_name = None
        self.last_request_time = time.time()
        self.last_response_time = time.time()
    
    def send_request(self, request, design_pipeline, run_name, reset_server=True):
        if reset_server:
            # Reset the server
            conda_env = os.getenv('CONDA_DEFAULT_ENV', 'jax9400')
            start_server(self.port, cuda=self.cuda, conda=conda_env) #TODO: from config
            print("Servers started!")
            time.sleep(10)

            # Reset the socket
            # self.socket.close()
            # self.poller.unregister(self.socket)
            # self.socket = self.context.socket(zmq.REQ)
            # self.socket.connect(self.address)
            # self.poller.register(self.socket, zmq.POLLIN)

        self.train_conf = request
        self.design_pipeline = design_pipeline
        self.run_name = run_name
        # self.socket.send(pickle.dumps(request))
        self.client_sock.sendto(pickle.dumps(request), self.address)
        print(f"Sent request to {self.address} (Run {run_name})")

        self.last_request_time = time.time()

    def resend_request(self):
        self.send_request(self.train_conf, self.design_pipeline, self.run_name, reset_server=True)

    def get_fitness(self, fitness_type="mean"):
        response = self.socket.recv()
        ep_rew_mean_values = pickle.loads(response)
        if fitness_type == "mean":
            score = np.mean(ep_rew_mean_values)
        elif fitness_type == "tail":
            tail_length = max(1, int(len(ep_rew_mean_values) * 0.1))
            score = np.median(ep_rew_mean_values[-tail_length:]) # Median of the last 10% of the episodes
                                                                 # Use median to advoid simulation unstable
        return score

        
class AsyncGA(object):
    def __init__(self, cfg_name="evolution"):
        self.cfg_name = cfg_name

        # Configuration
        self.n_workers = 9
        self.timeout = 15*60 # seconds

        self.conf = load_cfg(name=self.cfg_name, alg="evolve")
        
        self_collision_threshold = self.conf.trainer.evolution.self_collision_threshold
        ave_speed_threshold = self.conf.trainer.evolution.ave_speed_threshold
        self.max_depth = self.conf.trainer.evolution.max_mutate_depth
        self.wandb_on = self.conf.trainer.wandb_on
        self.run_token = datetime.datetime.now().strftime("%m%d%H%M%S")
        self.evolve_log_dir = os.path.join(LEG_ROOT_DIR,
                                        "exp", 
                                        "sim_evolve", 
                                        f"{self.conf.agent.obs_version}-{self.conf.agent.reward_version}{self.run_token}"
                                        )
        os.makedirs(self.evolve_log_dir, exist_ok=True)

        # Setup the workers
        self.workers = []
        for i in range(self.n_workers):
            worker = Worker(address=("0.0.0.0", 5555+i), cuda=i%3+1)
            self.workers.append(worker)

        # Initialize the population
        self.g = partial(are_ok, self_collision_threshold=self_collision_threshold, ave_speed_threshold=ave_speed_threshold)
        self.pop_dict = {f"init{i}":(x,y) for i,(x,y) in enumerate(zip(ini_xs, ini_ys))} # i: design idx, the order of the design does not matter

        # Logging
        wandb.init(project="OctopusLite", 
                            name=f"[Evolution]{self.conf.agent.obs_version}-{self.conf.agent.reward_version}{self.run_token}", 
                            config=OmegaConf.to_container(self.conf), 
                            sync_tensorboard=True, 
                            notes=self.conf.trainer.notes, mode="online" if self.wandb_on else "disabled")
        self.csv_logger = CSVLogger(self.evolve_log_dir)
        
        self.best_fitness = -np.inf
        self.best_design_pipeline = None

    def _birth_and_send(self, worker, run_name):
        new_design = birth(self.pop_dict, self.g, max_depth=self.max_depth)
        if new_design is not None:
            conf = update_cfg_with_pipeline(self.conf, new_design, run_name, f"E{self.run_token}-{run_name}", self.evolve_log_dir)
            worker.send_request(conf, new_design, run_name, reset_server=True)

    def _gen_and_send(self, worker, run_name):
        new_design = gen(self.pop_dict, self.g, max_depth=self.max_depth)
        if new_design is not None:
            conf = update_cfg_with_pipeline(self.conf, new_design, run_name, f"E{self.run_token}-{run_name}", self.evolve_log_dir)
            worker.send_request(conf, new_design, run_name, reset_server=True)


    def _update_pop_dict(self, worker, score):
        self.pop_dict[worker.run_name] = (worker.design_pipeline, score)

        if score > self.best_fitness:
            self.best_fitness = score
            self.best_design_pipeline = worker.design_pipeline

        if self.wandb_on:
            wandb.log({"Fitness": score, 
                       "DesignPipeline": wandb.Histogram(worker.design_pipeline),
                       "BestFitness": self.best_fitness,
                       "PopSize": len(self.pop_dict)
                       })
        self.csv_logger.log(worker.run_name, worker.design_pipeline, score)



    def run(self):
        # Initialize the first batch of workers
        for i in range(self.n_workers):
            run_name = f"W{i}G{0}"
            self._gen_and_send(self.workers[i], run_name)
        
        while True:

            # Poll the sockets with a small timeout to allow frequent checking
            polled_sockets = dict(self.poller.poll(1000))

            for worker in self.workers:
                if worker.socket in polled_sockets and polled_sockets[worker.socket] == zmq.POLLIN:
                    # If we have a response, receive it and send a new request
                    score = worker.get_fitness()
                    print(f"Received response from {worker.address}: {score} (Run {worker.run_name})")
                    self._update_pop_dict(worker, score)

                    # run_name = f"{worker.run_name[:-1]}{int(worker.run_name[-1])+1}"
                    run_name = update_run_name(run_name)
                    if len(self.pop_dict) < 50:
                        self._gen_and_send(worker, run_name)
                    else:
                        self._birth_and_send(worker, run_name)

                elif time.time() - worker.last_request_time >= self.timeout:
                    # If individual timeout has occurred, resend the request
                    print(f"Timeout occurred for {worker.address}, resending request")
                    worker.resend_request()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, default='evolution')
    args = parser.parse_args()

    ga = AsyncGA(cfg_name=args.cfg)
    ga.run()