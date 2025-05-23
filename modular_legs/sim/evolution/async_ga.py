





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
import zmq
from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.evolution.mutation import crossover, mutate
from modular_legs.sim.evolution.run_server import start_server
from modular_legs.sim.evolution.utils import CSVLogger, gen_log_dir, is_pipeline_valid
from modular_legs.sim.robot_designer import RobotDesigner
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
            is_pipeline_valid(design_pipeline, 
                                level=2, 
                                conf_dict={"self_collision_threshold": self_collision_threshold, "ave_speed_threshold": ave_speed_threshold}
                                )
            )

    not_empty = [x != [] for x in xs]

    return [a and b for a, b in zip(valid, not_empty)]


def update_cfg_with_pipeline(cfg, design_pipeline, run_name, wandb_run_name, evolve_log_dir, init_pose_type="default"):
    '''
    Update the config with the design pipeline (TODO: input designer to save decoding time)
    '''
    designer = RobotDesigner(design_pipeline)
    designer.compile()
    if cfg.sim.terrain is not None:
        designer.set_terrain(cfg.sim.terrain)
    xml, robot_properties = designer.get_xml(), designer.robot_properties
    xml_file, yaml_file = designer.save(gen_log_dir(design_pipeline), render=False)

    num_joints = robot_properties["num_joints"]
    num_act = num_joints
    num_obs = 6 + num_joints*3 # for robust_proprioception

    if init_pose_type == "default":
        init_pos = robot_properties["stable_pos"][0].tolist()
        init_quat = robot_properties["stable_quat"][0].tolist()
    elif init_pose_type == "highest":
        highest_idx = np.argmax(robot_properties["stable_height"])
        init_pos = robot_properties["stable_pos"][highest_idx].tolist()
        init_quat = robot_properties["stable_quat"][highest_idx].tolist()


    conf = copy.deepcopy(cfg)
    conf.agent.num_act = num_act
    conf.agent.num_obs = num_obs
    conf.sim.asset_file = xml_file
    conf.sim.init_pos = init_pos
    conf.sim.init_quat = init_quat
    OmegaConf.update(conf, "trainer.evolution.design_pipeline", f"{[int(i) for i in design_pipeline]}") # for logging
    OmegaConf.update(conf, "trainer.evolution.run_name", wandb_run_name) # for logging

    train_log_dir = os.path.join(evolve_log_dir, run_name)
    conf.logging.data_dir = train_log_dir
    return conf


# pop_size = 18
# n_delete = 3
n_pool_a = 3
rate_pool_b = 0.5
# mutate_rate = 0.3
# n_gen = 100

max_n_modules = 6
max_design_length = (max_n_modules-1)*4
min_n_modules = 3
min_design_length = (min_n_modules-1)*4

ini_xs = [[0, 1, 0, 1, 0, 3, 0, 0, 1, 0, 1, 13, 0, 6, 1, 2], [0, 0, 1, 0, 0, 3, 0, 2, 1, 4, 1, 2, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 2, 1, 0, 2, 1, 1, 0, 5, 0, 0], [1, 0, 1, 7, 0, 1, 0, 2, 0, 0, 2, 1, 1, 2, 5, 7], [1, 1, 2, 12, 0, 0, 2, 0, 1, 0, 1, 3], [1, 1, 1, 2, 0, 0, 2, 1, 0, 4, 0, 1, 0, 6, 2, 0], [0, 1, 1, 1, 0, 0, 2, 1, 1, 3, 0, 9], [1, 0, 1, 10, 0, 1, 2, 0, 1, 4, 1, 6], [0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 2, 1], [0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 2, 2, 1, 2, 1, 6], [1, 1, 2, 0, 1, 0, 1, 10, 0, 2, 2, 0], [0, 1, 2, 0, 1, 2, 1, 7, 1, 0, 1, 8], [0, 0, 2, 1, 0, 2, 1, 1, 1, 1, 0, 10], [1, 0, 1, 7, 0, 2, 5, 0, 1, 3, 1, 9], [0, 1, 2, 1, 0, 3, 1, 2, 0, 5, 0, 0], [1, 0, 2, 9, 1, 0, 1, 10, 0, 1, 0, 0], [1, 1, 0, 9, 1, 2, 2, 0, 1, 0, 2, 1], [1, 0, 2, 6, 0, 1, 0, 1, 0, 4, 2, 1], [0, 1, 0, 2, 0, 0, 2, 0, 1, 2, 2, 7, 0, 1, 1, 0, 1, 2, 1, 5], [0, 0, 1, 0, 1, 3, 1, 5, 0, 2, 1, 2, 0, 1, 2, 0], [0, 1, 2, 1, 1, 3, 2, 2, 1, 0, 1, 14], [0, 1, 1, 0, 1, 3, 0, 14, 0, 0, 2, 2], [0, 1, 0, 1, 1, 3, 0, 1, 1, 4, 6, 3, 0, 5, 5, 0], [1, 0, 2, 11, 1, 2, 7, 8, 1, 2, 4, 3, 0, 0, 1, 2], [0, 1, 1, 2, 1, 2, 1, 13, 1, 0, 2, 9, 0, 5, 8, 1, 1, 4, 1, 2], [0, 1, 0, 2, 0, 0, 2, 2, 0, 5, 1, 0, 0, 1, 2, 2], [1, 1, 1, 13, 0, 1, 0, 1, 1, 4, 1, 8], [1, 0, 1, 3, 0, 1, 1, 1, 0, 0, 2, 1], [1, 1, 1, 14, 0, 1, 0, 0, 1, 4, 2, 5], [0, 0, 1, 1, 1, 2, 1, 6, 1, 4, 4, 1, 0, 4, 5, 0, 1, 1, 2, 10], [0, 1, 1, 2, 0, 2, 1, 2, 1, 5, 1, 0, 0, 4, 2, 1, 1, 2, 2, 13], [0, 1, 2, 1, 0, 0, 2, 0, 1, 4, 2, 6], [0, 0, 1, 0, 0, 1, 0, 0, 1, 4, 2, 7, 1, 1, 2, 0, 0, 0, 2, 0], [1, 1, 1, 2, 0, 2, 5, 2, 0, 4, 2, 1], [0, 1, 1, 2, 0, 2, 1, 1, 0, 2, 2, 0, 0, 0, 1, 1], [0, 1, 2, 1, 0, 3, 1, 2, 0, 5, 2, 1]]

ini_ys = [15.61551306, 25.80413686,  2.64955958, 29.15538669,  8.03729593,  0.23040906,
          11.60188343,  1.04745745, 32.62483575, 21.57420166,  0.94809349, 25.58622902,
          21.842258,   -0.10751495,  2.21257145, 20.09702084,  8.83079485,  5.67999607,
          12.54270258,  5.94099415, 13.21084102,  7.03647125,  7.99051496,  7.56764707,
          3.30959404, 71.26605299, -0.32883073, -0.91031607, 34.26561007, 20.6834816,
          12.81995246, 23.9294342,  40.4077244,  10.38192072, 33.26741582,  9.35627933]


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
        context = zmq.Context()
        self.poller = zmq.Poller()
        self.workers = []
        for i in range(self.n_workers):
            worker = Worker(address=f"tcp://localhost:{5555+i}", cuda=i%3+1, context=context, poller=self.poller)
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
            self._birth_and_send(self.workers[i], run_name)
        
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