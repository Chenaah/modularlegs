'''
The main asynchronous VAE evolution script. 
'''

import argparse
import copy
import csv
import datetime
from functools import partial
import os

import pdb
import shutil
import time
import numpy as np
from omegaconf import OmegaConf
import ray
# from ray.experimental.tqdm_ray import tqdm
# ray.shutdown()
# ray.init(logging_level="DEBUG")
import wandb
import socket
import select
from rich.progress import Progress
# from sb3_contrib import CrossQ
from sbx import CrossQ
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from modular_legs import LEG_ROOT_DIR
from modular_legs.envs.env_sim import ZeroSim
from modular_legs.envs.gym.rendering import RecordVideo
from modular_legs.sim.evolution.async_ga_meta import are_ok, update_run_name
from modular_legs.sim.evolution.pose_optimizer import update_cfg_with_draft_asset, update_cfg_with_optimized_pose
from modular_legs.sim.evolution.run_server import start_server
from modular_legs.sim.evolution.utils import CSVLogger, CSVLoggerPro, gen_log_dir_5x1, is_metapipeline_valid, update_cfg_with_pipeline
from modular_legs.sim.evolution.vae.vae_trainer import VAETrainer, get_vae_cfg
# from modular_legs.sim.robot_metadesigner import MetaDesigner
from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_DRAFT, MESH_DICT_FINE, ROBOT_CFG_AIR1S
from modular_legs.utils.files import load_cfg, get_log_name
from modular_legs.utils.model import is_headless




class ProgressBarCallback(BaseCallback):

    def __init__(self) -> None:
        super().__init__()

    # def _on_training_start(self) -> None:
    #     # Initialize progress bar
    #     # Remove timesteps that were done in previous training sessions
    #     self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        # self.pbar.update(self.training_env.num_envs)
        if self.num_timesteps % 10000 == 0:
            print(f"[Worker {ray.get_runtime_context().get_actor_id()}] Step: {self.num_timesteps} / {self.locals['total_timesteps']}")
        return True

    # def _on_training_end(self) -> None:
    #     # Flush and close progress bar
    #     self.pbar.refresh()
    #     self.pbar.close()


def train_job(conf, filter_negative=True):

    # Update the config with optimized initial pose and joint positions
    # This should be done before initializing wandb so that the optimized values are logged
    os.makedirs(conf.logging.data_dir, exist_ok=True)
    if conf.sim.init_quat == "?" and conf.sim.init_pos.startswith("?") and conf.agent.default_dof_pos == "?":
        conf = update_cfg_with_draft_asset(conf, MESH_DICT_DRAFT, ROBOT_CFG_AIR1S)
        opt_params = conf.trainer.evolution.pose_optimization_params
        print(f"[Worker {ray.get_runtime_context().get_actor_id()}] Optimizing pose with params: {conf.trainer.evolution.pose_optimization_type}, {opt_params}")
        conf = update_cfg_with_optimized_pose(conf, opt_params[0], opt_params[1], conf.trainer.evolution.pose_optimization_type, log_dir=conf.logging.data_dir)
        pose_score = conf.trainer.evolution.pose_score
        if pose_score < 0 and filter_negative:
            print(f"[Worker {ray.get_runtime_context().get_actor_id()}] Pose optimization failed with score: {pose_score}")
            return False


    with open(os.path.join(conf.logging.data_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)
    shutil.copy(conf.sim.asset_file, conf.logging.data_dir)
    shutil.copy(conf.sim.asset_draft, conf.logging.data_dir)

    run = wandb.init(
        project="OctopusLite",
        name=conf.trainer.evolution.run_name,
        config=OmegaConf.to_container(conf),
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        mode="online" if conf.trainer.wandb_on else "disabled",
        notes=conf.trainer.evolution.design_pipeline
    )
    log_dir =conf.logging.data_dir
    # logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    logger = configure(log_dir, ["csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
        )
    wandb_callback = WandbCallback(log="all")
    pbar_callback = ProgressBarCallback()


    assert conf.robot.mode == "sim"
    if conf.sim.render and is_headless():
        conf.sim.render = False
        # print("Running in headless mode; render is turned off!")
    env = gym.wrappers.TimeLimit(
                ZeroSim(conf), max_episode_steps=1000
            )
    # Is the crash due to the video recording?
    # if not conf.sim.render:
    #     trigger = lambda t: t == int(conf.trainer.total_steps / 1000) - 1
    #     env = RecordVideo(env, 
    #                       video_folder=conf.logging.data_dir, 
    #                       episode_trigger=trigger, 
    #                       disable_logger=True,
    #                       fps=1/conf.robot.dt,
    #                       video_length=500
    #                       )

    model = CrossQ("MlpPolicy", env, verbose=1, device=conf.trainer.device)
    if conf.trainer.load_run is not None:
        model = CrossQ.load(conf.trainer.load_run, env=env)
    if conf.trainer.load_replay_buffer is not None:
        model.load_replay_buffer(conf.trainer.load_replay_buffer)

    model.set_logger(logger)

    assert conf.trainer.mode == "train", "Only training mode is supported for now"
    model.learn(total_timesteps=conf.trainer.total_steps, 
                callback=[checkpoint_callback, wandb_callback, pbar_callback]) 
    
    if conf.trainer.wandb_on:
        run.finish()
    env.close()

    return True



# Define the worker actor
@ray.remote(num_gpus=0.3, memory=64 * 1024 * 1024 * 1024)
class Worker:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['MUJOCO_GL'] = 'egl'
    def __init__(self, worker_id):
        self.idle = True  # Keep track of worker's state
        self.worker_id = worker_id
        self.i_gen = 0
        self.design_pipeline = None
    

    def process_task(self, conf):
        self.idle = False
        print(f"Worker {ray.get_runtime_context().get_actor_id()} processing: {self.run_name}")

        self.design_pipeline = conf.trainer.evolution.design_pipeline_list

        # Reset different seeds for each run
        conf.trainer.seed += int(self.worker_id)*10+int(self.i_gen)

        # Train the model
        pose_opt_flag = train_job(conf)
        if not pose_opt_flag:
            # Can't find a valid pose
            self.update_generation()
            self.idle = True
            return 0

        train_log_dir = conf.logging.data_dir
        ep_rew_mean_values = []
        with open(os.path.join(train_log_dir, "progress.csv"), mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                ep_rew_mean_values.append(float(row["rollout/ep_rew_mean"]))
        fitness_type = conf.trainer.evolution.fitness_type
        if fitness_type == "mean":
            score = np.mean(ep_rew_mean_values)
        elif fitness_type == "tail":
            tail_length = max(1, int(len(ep_rew_mean_values) * 0.1))
            score = np.median(ep_rew_mean_values[-tail_length:])
        
        self.update_generation()
        self.idle = True
        return score

    def is_idle(self):
        print(f"Worker {ray.get_runtime_context().get_actor_id()} is idle: {self.idle}")
        return self.idle
    
    def update_generation(self):
        self.i_gen += 1
        return self.i_gen
    
    @property
    def run_name(self):
        return f"W{self.worker_id}G{self.i_gen}"
    
    def get_design_pipeline(self):
        return self.design_pipeline
    
    def get_run_name(self):
        return self.run_name
    
    def get_last_run_name(self):
        return f"W{self.worker_id}G{self.i_gen-1}"
    
    
        
class AsyncVAERay(object):
    def __init__(self, cfg_name="evolution"):
        self.cfg_name = cfg_name

        self.conf = load_cfg(name=self.cfg_name, alg="evolve")

        # Configuration
        self.n_workers = self.conf.trainer.evolution.num_servers
        # n_gpu = self.conf.trainer.evolution.num_gpus # Deprecated
        visiable_gpus = self.conf.trainer.evolution.visiable_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in visiable_gpus])

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
        self.load_gp = self.conf.trainer.evolution.load_gp

        # Create the client socket (self)
        # self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.client_sock.bind(("0.0.0.0", 5550))
        # print(f"Client bound to port 5550")

        self.progress = Progress()
        self.progress.start()
        self.progress_tasks = []

        self.workers = [Worker.remote(i) for i in range(self.n_workers)]

        # Initialize the population
        self.g = partial(are_ok, self_collision_threshold=self_collision_threshold, ave_speed_threshold=ave_speed_threshold)
        ini_xs = []
        ini_ys = []
        self.pop_dict = {f"init{i}":(x,y) for i,(x,y) in enumerate(zip(ini_xs, ini_ys))} # i: design idx, the order of the design does not matter

        run = wandb.init(project="OctopusLite", 
                            name=self.master_run_name, 
                            config=OmegaConf.to_container(self.conf), 
                            sync_tensorboard=True, 
                            notes=self.conf.trainer.notes, mode="online" if self.wandb_on else "disabled"
                            )
        self.csv_logger_old = CSVLogger(self.evolve_log_dir)
        self.csv_logger = CSVLoggerPro(os.path.join(self.evolve_log_dir, "log_detail.csv"))
        self.run_idx = 0
        
        self.best_fitness = -np.inf
        self.best_design_pipeline = None

        # Setup the VAE
        cfg = get_vae_cfg()
        OmegaConf.update(cfg, "log_dir", os.path.join(self.evolve_log_dir, "VAE"))
        OmegaConf.update(cfg, "load_from_checkpoint", self.vae_checkpoint)
        OmegaConf.update(cfg, "dataset_path", os.path.join(LEG_ROOT_DIR, f"data/designs/{self.dataset_name}.pkl"))
        OmegaConf.update(cfg, "device", f"cuda" if n_gpu > 0 else "cpu")
        OmegaConf.update(cfg, "likelihood_variance", self.conf.trainer.evolution.likelihood_variance)
        OmegaConf.update(cfg, "latent_dim", self.conf.trainer.evolution.latent_dim)
        OmegaConf.update(cfg, "opt_bounds", self.conf.trainer.evolution.opt_bounds)
        OmegaConf.update(cfg, "seed", self.conf.trainer.seed)
        
        self.vae_trainer = VAETrainer(cfg, self.n_workers, self.use_result_buffer, run, optimizer=self.optimizer_type, load_gp=self.load_gp, logger=self.csv_logger)


    def _get_new_task(self, run_name):
        new_design = self.vae_trainer.get_new_design()
        return self._design_to_task(new_design, run_name)
    
    def _get_init_tasks(self):
        # Generate init n_workers tasks
        self.init_designs = self.vae_trainer.get_init_designs()
        assert len(self.init_designs) == self.n_workers, "Number of init designs must be equal to number of workers"
        return [self._design_to_task(d, f"W{i}G{0}") for i,d in enumerate(self.init_designs)]

    def _design_to_task(self, new_design, run_name):

        # TODO: from config
        robot_cfg=ROBOT_CFG_AIR1S
        mesh_dict=MESH_DICT_FINE

        conf = update_cfg_with_pipeline(self.conf, 
                                        new_design, 
                                        robot_cfg,
                                        mesh_dict, 
                                        run_name=run_name, 
                                        wandb_run_name=f"E{self.master_run_name.split('-')[-1]}-{run_name}", 
                                        evolve_log_dir=self.evolve_log_dir,
                                        init_pose_type=self.init_pose_type
                                        )
        return conf
    
    
    def _tell_result(self, design_pipeline, score, run_name):
        # Tell the blackbox optimizer the result
        self.vae_trainer.tell_result(design_pipeline, score, run_name)
        # Update the population dictionary for logging
        self._update_pop_dict(design_pipeline, score, run_name)

    def _tell_init_results(self, scores):
        self.vae_trainer.tell_init_results(scores, run_names=[f"W{i}G{0}" for i in range(len(self.init_designs))])
        for i, (design_pipeline, score) in enumerate(zip(self.init_designs, scores)):
            self._update_pop_dict(design_pipeline, score, f"W{i}G{0}")


    def _update_pop_dict(self, design_pipeline, score, run_name):

        # Update the population dictionary for logging
        self.pop_dict[run_name] = (design_pipeline, score)

        if score > self.best_fitness:
            self.best_fitness = score
            self.best_design_pipeline = design_pipeline

        if self.wandb_on:
            wandb.log({"Fitness": score, 
                       "DesignPipeline": wandb.Histogram(design_pipeline),
                       "BestFitness": self.best_fitness,
                       "PopSize": len(self.pop_dict)
                       })
        self.csv_logger_old.log(run_name, design_pipeline, score)

        self.run_idx += 1



    def run(self):

        # Train the VAE
        if self.vae_checkpoint is None:
            self.vae_trainer.train_vae()
        
        init_tasks = self._get_init_tasks()
        init_futures = [worker.process_task.remote(task) for worker, task in zip(self.workers, init_tasks)]
        init_results = ray.get(init_futures)
        self._tell_init_results(init_results)
        # task_queue = [self._get_new_task(ray.get(worker.get_run_name.remote())) for worker in self.workers]

        print("Finished initializing jobs")

        task_futures = {}
        active_workers = set()

        while True:
            # Check for completed tasks
            ready, _ = ray.wait(list(task_futures.keys()), timeout=1, num_returns=1)
            
            for future in ready:
                worker = task_futures.pop(future)
                active_workers.remove(worker)
                try:
                    result = ray.get(future)
                    run_name = ray.get(worker.get_last_run_name.remote())
                    design_pipeline = ray.get(worker.get_design_pipeline.remote())
                    self._tell_result(design_pipeline, result, run_name)
                except Exception as e:
                    print(f"Task failed: {e}")

            # Assign new tasks to idle workers
            for worker in self.workers:
                print("I am at E at ", datetime.datetime.now().strftime("%H:%M:%S") )
                if worker not in active_workers and ray.get(worker.is_idle.remote()):
                    print("I am at F at ", datetime.datetime.now().strftime("%H:%M:%S") )
                    run_name = ray.get(worker.get_run_name.remote())
                    new_task = self._get_new_task(run_name)
                    future = worker.process_task.remote(new_task)
                    task_futures[future] = worker
                    active_workers.add(worker)


        # Clean up workers
        ray.shutdown()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='evolution_vae_asym_air1s_debug')
    args = parser.parse_args()

    ga = AsyncVAERay(cfg_name=args.cfg)
    ga.run()