'''
The main asynchronous VAE evolution script. 
'''

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
import socket
import select
from rich.progress import Progress
from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.evolution.async_ga_meta import Worker, are_ok, update_run_name
from modular_legs.sim.evolution.mutation_meta import crossover, extend_random_design, mutate, random_gen
from modular_legs.sim.evolution.run_server import start_server
from modular_legs.sim.evolution.utils import CSVLogger, gen_log_dir_5x1, is_metapipeline_valid, update_cfg_with_pipeline
from modular_legs.sim.evolution.vae.utils import pipe5x1_to_n_modules
from modular_legs.sim.evolution.vae.vae_trainer import VAETrainer, get_vae_cfg
from modular_legs.sim.robot_designer import RobotDesigner
# from modular_legs.sim.robot_metadesigner import MetaDesigner
from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_FINE, ROBOT_CFG_AIR1S
from modular_legs.utils.files import load_cfg, get_log_name



        
class AsyncVAE(object):
    def __init__(self, cfg_name="evolution"):
        self.cfg_name = cfg_name

        

        self.conf = load_cfg(name=self.cfg_name, alg="evolve")

        # Configuration
        self.n_workers = self.conf.trainer.evolution.num_servers
        # n_gpu = self.conf.trainer.evolution.num_gpus # Deprecated
        visiable_gpus = self.conf.trainer.evolution.visiable_gpus
        n_gpu = len(visiable_gpus)
        self.timeout = 120*60 # seconds
        
        self_collision_threshold = self.conf.trainer.evolution.self_collision_threshold
        ave_speed_threshold = self.conf.trainer.evolution.ave_speed_threshold
        self.max_depth = self.conf.trainer.evolution.max_mutate_depth
        self.wandb_on = self.conf.trainer.wandb_on
        # self.run_token = datetime.datetime.now().strftime("%m%d%H%M%S")
        self.master_run_name = get_log_name(self.conf).replace("?-", "evo-")
        self.evolve_log_dir = os.path.join(LEG_ROOT_DIR,
                                        "exp", 
                                        "sim_vae", 
                                        self.master_run_name
                                        )
        os.makedirs(self.evolve_log_dir, exist_ok=True)
        self.fitness_type = self.conf.trainer.evolution.fitness_type
        self.fitness_per_module = self.conf.trainer.evolution.fitness_per_module
        self.init_pose_type = self.conf.trainer.evolution.init_pose_type
        self.vae_checkpoint = self.conf.trainer.evolution.vae_checkpoint
        self.dataset_name = self.conf.trainer.evolution.dataset_name
        self.use_result_buffer = self.conf.trainer.evolution.use_result_buffer
        self.optimizer_type = self.conf.trainer.evolution.optimizer

        # Create the client socket (self)
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_sock.bind(("0.0.0.0", 5550))
        print(f"Client bound to port 5550")

        self.progress = Progress()
        self.progress.start()
        self.progress_tasks = []



        self.workers = []
        for i in range(self.n_workers):
            worker = Worker(address=("127.0.0.1", 5555+i), client_sock=self.client_sock, worker_id=i, cuda=visiable_gpus[i%n_gpu])
            self.workers.append(worker)

        # Initialize the population
        self.g = partial(are_ok, self_collision_threshold=self_collision_threshold, ave_speed_threshold=ave_speed_threshold)
        ini_xs = []
        ini_ys = []
        self.pop_dict = {f"init{i}":(x,y) for i,(x,y) in enumerate(zip(ini_xs, ini_ys))} # i: design idx, the order of the design does not matter


        # Logging
        # obs_name = ''.join([i[0] for i in self.conf.agent.obs_version.split("_")])
        # rwd_name = ''.join([i[0] for i in self.conf.agent.reward_version.split("_")])
        # if self.conf.sim.reset_terrain and self.conf.sim.reset_terrain_type is not None:
        #     add_info = ''.join([i[0] for i in self.conf.sim.reset_terrain_type.split("_")])
        #     if self.conf.sim.reset_terrain_params is not None:
        #         add_info += f"{self.conf.sim.reset_terrain_params[0]}"
        # else:
        #     add_info = "f"
        # run_name = f"[E{self.run_token}]{add_info}-{obs_name}-{rwd_name}"
        run = wandb.init(project="OctopusLite", 
                            name=self.master_run_name, 
                            config=OmegaConf.to_container(self.conf), 
                            sync_tensorboard=True, 
                            notes=self.conf.trainer.notes, mode="online" if self.wandb_on else "disabled"
                            )
        self.csv_logger = CSVLogger(self.evolve_log_dir)
        
        self.best_fitness = -np.inf
        self.best_design_pipeline = None

        # Setup the VAE
        cfg = get_vae_cfg()
        OmegaConf.update(cfg, "log_dir", os.path.join(self.evolve_log_dir, "VAE"))
        OmegaConf.update(cfg, "load_from_checkpoint", self.vae_checkpoint)
        OmegaConf.update(cfg, "dataset_path", os.path.join(LEG_ROOT_DIR, f"data/designs/{self.dataset_name}.pkl"))
        OmegaConf.update(cfg, "device", f"cuda:{visiable_gpus[0]}" if n_gpu > 0 else "cpu")
        cfg.seed = self.conf.trainer.seed
        
        self.vae_trainer = VAETrainer(cfg, self.n_workers, self.use_result_buffer, run, optimizer=self.optimizer_type)


    def _gen_and_send(self, worker, run_name):
        new_design = self.vae_trainer.get_new_design()

        # TODO: from config
        robot_cfg=ROBOT_CFG_AIR1S
        mesh_dict=MESH_DICT_FINE

        conf = update_cfg_with_pipeline(self.conf, 
                                        new_design, 
                                        robot_cfg,
                                        mesh_dict, 
                                        run_name=run_name, 
                                        wandb_run_name=f"E{self.master_run_name.split("-")[-1]}-{run_name}", 
                                        evolve_log_dir=self.evolve_log_dir,
                                        init_pose_type=self.init_pose_type
                                        )
        worker.send_request(conf, new_design, run_name, reset_server=True)


    def _update_pop_dict(self, worker, score):

        # Tell the blackbox optimizer the result
        self.vae_trainer.tell_result(worker.design_pipeline, score)

        # Update the population dictionary for logging
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

        # Train the VAE
        if self.vae_checkpoint is None:
            self.vae_trainer.train_vae()
        

        # Initialize the first batch of workers
        for i in range(self.n_workers):
            run_name = f"W{i}G{0}"
            self._gen_and_send(self.workers[i], run_name)
            self.progress_tasks.append(self.progress.add_task(f"[magenta]Worker {5555+i}", total=int(self.conf.trainer.total_steps/1000)))
        
        wait_counter = 0

        while len(self.pop_dict) < self.conf.trainer.evolution.num_trials:

            # Monitor sockets for incoming data
            readable, _, _  = select.select([self.client_sock], [], [], 5)

            if readable:
                data, addr = self.client_sock.recvfrom(40960)
                print(f"Received data from {addr}")

                for worker in self.workers:
                    if worker.port == addr[1]:
                        # If we have a response, receive it and send a new request
                        print(f"Received response from {worker.address}")
                        worker.last_response_time = time.time()
                        ep_rew_mean_values = pickle.loads(data)
                        if not isinstance(ep_rew_mean_values, str):
                            # Finished running the training job
                            self.progress.update(self.progress_tasks[worker.worker_id], completed=100)
                            if self.fitness_type == "mean":
                                score = np.mean(ep_rew_mean_values)
                            elif self.fitness_type == "tail":
                                tail_length = max(1, int(len(ep_rew_mean_values) * 0.1))
                                score = np.median(ep_rew_mean_values[-tail_length:])
                            # score = np.mean(ep_rew_mean_values[-10:])
                            n_modules = pipe5x1_to_n_modules(worker.design_pipeline)
                            if self.fitness_per_module:
                                score = score / n_modules
                            print(f"Received response from {worker.address}: {score} (Run {worker.run_name})")
                            self._update_pop_dict(worker, score)
                            run_name = update_run_name(worker.run_name)
                            self._gen_and_send(worker, run_name)
                            self.progress.reset(self.progress_tasks[worker.worker_id])

                        else:
                            # Received intermediate loggings
                            print(f"Received response from {worker.address}: {ep_rew_mean_values}")
                            self.progress.update(self.progress_tasks[worker.worker_id], advance=1)
            else:
                for worker in self.workers:
                    if time.time() - worker.last_request_time >= self.timeout:
                        # If individual timeout has occurred, resend the request
                        print(f"Timeout occurred for {worker.address}, resending request")
                        worker.resend_request()

            print(f"[{wait_counter%10}] Waiting for responses... ", end='\r', flush=True)

            wait_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, default='evolution')
    args = parser.parse_args()

    ga = AsyncVAE(cfg_name=args.cfg)
    ga.run()