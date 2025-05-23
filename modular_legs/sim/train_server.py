import ast
import csv
import importlib
import os
import pdb
import shutil

from modular_legs.sim.evolution.pose_optimizer import get_local_vectors, optimize_pose, update_cfg_with_draft_asset, update_cfg_with_optimized_pose
from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_DRAFT, ROBOT_CFG_AIR1S
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
# from sb3_contrib import CrossQ
from sbx import CrossQ
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from omegaconf import OmegaConf
import wandb
from wandb.integration.sb3 import WandbCallback
from modular_legs import LEG_ROOT_DIR
from modular_legs.envs.env_sim import ZeroSim
from modular_legs.utils.files import load_cfg
from modular_legs.envs.gym.rendering import RecordVideo
import gymnasium as gym
import socket
import pickle
import msgpack
import time
from modular_legs.utils.model import is_headless

DEBUG = False


class SocketCallback(BaseCallback):

    def __init__(self, server_sock, client_addr) -> None:
        super().__init__()
        self.server_sock = server_sock
        self.client_addr = client_addr


    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        num_timesteps = self.model.num_timesteps
        if num_timesteps % 1000 == 0:
            response = f"Step: {num_timesteps}"
            self.server_sock.sendto(pickle.dumps(response), self.client_addr)
            print(f"Sent response: {response} to {self.client_addr}")
        else:
            print(f"Step: {num_timesteps}")
        print("Device: ", self.model.device)
        print("CUDA_VISIBLE_DEVICES:  ", os.environ['CUDA_VISIBLE_DEVICES'])
        return True


def train(conf, server_sock, client_addr):

    print("CUDA_VISIBLE_DEVICES:  ", os.environ['CUDA_VISIBLE_DEVICES'])


    # Update the config with optimized initial pose and joint positions
    # This should be done before initializing wandb so that the optimized values are logged
    if conf.sim.init_quat == "?" and conf.sim.init_pos.startswith("?") and conf.agent.default_dof_pos == "?":
        # design_pipeline = ast.literal_eval(conf.trainer.evolution.design_pipeline)
        # init_pos, init_quat, init_joint, info = optimize_pose(design_pipeline, 150, 250, conf.trainer.evolution.pose_optimization_type)
        # conf.sim.init_quat = init_quat
        # if conf.sim.init_pos.startswith("?+"):
        #     add_h = float(conf.sim.init_pos.split("+")[1])
        #     init_pos[2] += add_h
        # conf.sim.init_pos = init_pos
        # conf.agent.default_dof_pos = init_joint
        # if conf.agent.forward_vec == "?":
        #     conf.agent.forward_vec = info["forward_vec"].tolist()
        conf = update_cfg_with_draft_asset(conf, MESH_DICT_DRAFT, ROBOT_CFG_AIR1S)
        opt_params = conf.trainer.evolution.pose_optimization_params
        conf = update_cfg_with_optimized_pose(conf, opt_params[0], opt_params[1], conf.trainer.evolution.pose_optimization_type)

    # if conf.agent.projected_forward_vec == "?" or conf.agent.projected_upward_vec == "?":
    #     projected_forward, projected_upward = get_local_vectors(design_pipeline, init_pos=init_pos, init_quat=init_quat, init_joint=init_joint)
    #     conf.agent.projected_forward_vec = projected_forward.tolist()
    #     conf.agent.projected_upward_vec = projected_upward.tolist()

    os.makedirs(conf.logging.data_dir, exist_ok=True)
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
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
        )
    wandb_callback = WandbCallback(log="all")
    socket_callback = SocketCallback(server_sock, client_addr)


    assert conf.robot.mode == "sim"
    if conf.sim.render and is_headless():
        conf.sim.render = False
        print("Running in headless mode; render is turned off!")
    env = gym.wrappers.TimeLimit(
                ZeroSim(conf), max_episode_steps=1000
            )
    if not conf.sim.render:
        trigger = lambda t: t == int(conf.trainer.total_steps / 1000) - 1
        env = RecordVideo(env, 
                          video_folder=conf.logging.data_dir, 
                          episode_trigger=trigger, 
                          disable_logger=True,
                          fps=1/conf.robot.dt,
                          video_length=500
                          )

    model = CrossQ("MlpPolicy", env, verbose=1, device="auto")
    if conf.trainer.load_run is not None:
        model = CrossQ.load(conf.trainer.load_run, env=env)
    if conf.trainer.load_replay_buffer is not None:
        model.load_replay_buffer(conf.trainer.load_replay_buffer)

    model.set_logger(logger)

    assert conf.trainer.mode == "train", "Only training mode is supported for now"
    model.learn(total_timesteps=conf.trainer.total_steps, 
                callback=[checkpoint_callback, wandb_callback, socket_callback],
                progress_bar=not conf.logging.print_data) 
    
    if conf.trainer.wandb_on:
        run.finish()
    env.close()

def train_dummy(conf, server_sock, client_addr):


    print("Debugging!!!!")
    for i in range(10):
        print("Debugging!!!!")
        response = "Step " + str(i)
        server_sock.sendto(pickle.dumps(response), client_addr)
        time.sleep(3)

def _start_server(port):
    # context = zmq.Context()
    # socket = context.socket(zmq.REP)
    # socket.bind(f"tcp://*:{port}")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind(("0.0.0.0", int(port)))
    print(f"Server listening on port {port}")

    print("Server started! Waiting for requests...")

    while True:
        # Wait for next request from client
        message, addr = server_sock.recvfrom(40960)
        conf = pickle.loads(message)
        # received_data = msgpack.unpackb(message)

        print(f"Received request!")
        print(f"Logging config: {conf.logging.data_dir}")

        if not DEBUG:
            train(conf, server_sock, addr)
        else:
            train_dummy(conf, server_sock, addr)

        # Create a response object
        train_log_dir = conf.logging.data_dir
        ep_rew_mean_values = []
        if not DEBUG:
            with open(os.path.join(train_log_dir, "progress.csv"), mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    ep_rew_mean_values.append(float(row["rollout/ep_rew_mean"]))
        
        response = ep_rew_mean_values
        # socket.send(pickle.dumps(response))
        server_sock.sendto(pickle.dumps(response), addr)

        print(f"Sent response!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_server.py <port>")
        sys.exit(1)
    
    port = sys.argv[1]
    _start_server(port)