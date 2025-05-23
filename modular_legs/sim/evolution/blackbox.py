



import copy
import csv
import datetime
import math
import multiprocessing
import os
import pdb
import pickle
import random
import shutil
import time
import numpy as np
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ, DroQ
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from omegaconf import OmegaConf
import wandb
from wandb.integration.sb3 import WandbCallback
from modular_legs import LEG_ROOT_DIR
from modular_legs.envs.env_sim import ZeroSim
from modular_legs.sim.evolution.run_server import start_servers
from modular_legs.sim.evolution.utils import CSVLogger, decode, gen_log_dir, is_pipeline_valid, split_list
from modular_legs.sim.robot_designer import RobotDesigner
from modular_legs.utils.files import load_cfg
import gymnasium as gym
import zmq
from modular_legs.utils.others import is_list_like



        

class BlackBox:
    '''
        Input: blackbox.reset(design_pipelines)
        Output: blackbox.fitness()
                blackbox.valid()
    '''

    def __init__(self, conf="evolution", connect_to_server=True):

        # Config
        self.conf = load_cfg(name=conf, alg="evolve")
        self.n_servers = self.conf.trainer.evolution.num_servers
        self.n_backup_servers = 4
        self.self_collision_threshold = self.conf.trainer.evolution.self_collision_threshold
        self.ave_speed_threshold = self.conf.trainer.evolution.ave_speed_threshold
        self.max_depth = self.conf.trainer.evolution.max_mutate_depth
        self.wandb_on = self.conf.trainer.wandb_on

        if connect_to_server:
            self.context = zmq.Context()
            self._setup_socket()

            self.run_token = datetime.datetime.now().strftime("%m%d%H%M%S")
            run = wandb.init(project="OctopusLite", 
                            name=f"[Evolution]{self.conf.agent.obs_version}-{self.conf.agent.reward_version}{self.run_token}", 
                            config=OmegaConf.to_container(self.conf), 
                            sync_tensorboard=True, 
                            notes=self.conf.trainer.notes, mode="online" if self.conf.trainer.wandb_on else "disabled")
            self.evolve_log_dir = os.path.join(LEG_ROOT_DIR,
                                            "exp", 
                                            "sim_evolve", 
                                            f"{self.conf.agent.obs_version}-{self.conf.agent.reward_version}{self.run_token}"
                                            )
            os.makedirs(self.evolve_log_dir, exist_ok=True)
            self.csv_logger = CSVLogger(self.evolve_log_dir)

        self.run_counter = 0
        self.generation_counter = 0
        self.best_fitness = -np.inf
        self.best_design_pipeline = None
        self.n_designs = 0

    def reset(self, design_pipelines):
        '''
        Input shape: (B, n), where n%4 == 0
        '''
        assert all([is_list_like(i) for i in design_pipelines]), "design_pipelines should be a list of list"
        design_pipelines = [list(i) for i in design_pipelines]
        self.design_pipelines = design_pipelines
        self.asset_dirs = []
        self.designers = []
        for design_pipeline in design_pipelines:
            self.asset_dirs.append(gen_log_dir(design_pipeline))
            designer = RobotDesigner()
            designer.reset()
            for design_step in np.reshape(design_pipeline, (-1, 4)):
                designer.step(design_step)
            self.designers.append(designer)

    def _replace_socket(self, socket):
        socket.close()
        new_socket = self.context.socket( zmq.REQ )
        # zmq_req_socket.setsockopt( zmq.RCVTIMEO, 500 ) # milliseconds
        self.socket_addresses[new_socket] = self.backup_server_addresses[0]
        self.backup_server_addresses.pop(0)
        del self.socket_addresses[socket]
        new_socket.connect( self.socket_addresses[new_socket] )
        self.sockets[self.sockets.index(socket)] = new_socket
        self.poller.unregister(socket)
        self.poller.register(new_socket, zmq.POLLIN)

        return new_socket
        
    def _reset_sockets(self):
        for socket in self.sockets:
            socket.close()
            self.poller.unregister(socket)
        self._setup_socket()

    def _setup_socket(self):
        # Set up the zmq server
        self.server_addresses = ["tcp://localhost:{i}".format(i=5555+i) for i in range(self.n_servers)]
        self.backup_server_addresses = ["tcp://localhost:{i}".format(i=5560+self.n_servers+i) for i in range(self.n_backup_servers)]
        self.socket_addresses = {}

        # Socket to talk to server
        print("Connecting to server…")
        # Create a list of sockets
        self.sockets = []
        for address in self.server_addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)
            self.socket_addresses[socket] = address
        # Create poller and register sockets
        self.poller = zmq.Poller()
        for socket in self.sockets:
            self.poller.register(socket, zmq.POLLIN)

    def _decode(self, save=True):
        """
        Decode the design pipeline into a xml file / property file
        """
        self.xmls = []
        self.robot_properties_list = []
        self.num_act_list = []
        self.num_obs_list = []
        self.xml_file_list = []
        self.yaml_file_list = []

        for designer, design_pipeline, asset_dir in zip(self.designers, self.design_pipelines, self.asset_dirs):
            xml, robot_properties = decode(designer, design_pipeline)
            self.xmls.append(copy.deepcopy(xml))
            self.robot_properties_list.append(copy.deepcopy(robot_properties))

            num_joints = robot_properties["num_joints"]
            self.num_act_list.append(copy.deepcopy(num_joints))
            self.num_obs_list.append(6 + num_joints*3) # for robust_proprioception
                
            if save:
                xml_file, yaml_file = designer.save(asset_dir, render=False)
                self.xml_file_list.append(xml_file)
                self.yaml_file_list.append(yaml_file)
        

    def _train(self):
        '''
        Train the robot with the design pipeline
        '''
        self._decode()

        conf_list = []
        train_log_dir_list = []

        for i in range(len(self.design_pipelines)):
            conf = copy.deepcopy(self.conf)
            xml_file = self.xml_file_list[i]
            yaml_file = self.yaml_file_list[i]
            robot_properties = self.robot_properties_list[i]
            num_act = self.num_act_list[i]
            num_obs = self.num_obs_list[i]
            design_pipeline = self.design_pipelines[i]

            conf.agent.num_act = num_act
            conf.agent.num_obs = num_obs
            conf.sim.asset_file = xml_file
            conf.sim.init_pos = robot_properties["stable_pos"][0].tolist()
            conf.sim.init_quat = robot_properties["stable_quat"][0].tolist()
            OmegaConf.update(conf, "trainer.evolution.design_pipeline", f"{[int(i) for i in design_pipeline]}") # for logging
            OmegaConf.update(conf, "trainer.evolution.run_name", f"E{self.run_token}-G{self.generation_counter}D{i}") # for logging

        
            train_log_dir = os.path.join(self.evolve_log_dir,
                                        f"gen_{self.generation_counter}",
                                        f"design_{i}"
                                        )
            conf.logging.data_dir = train_log_dir
            os.makedirs(train_log_dir, exist_ok=True)
            shutil.copy(xml_file , train_log_dir)
            shutil.copy(yaml_file , train_log_dir)

            conf_list.append(conf)
            train_log_dir_list.append(train_log_dir)

        for conf_sub_list in split_list(conf_list, self.n_servers):

            conf_sent = {socket: None for socket in self.sockets}  # Track which config was sent to which socket

            for i, conf in enumerate(conf_sub_list):
                # Send the config to training server
                print(f"Sending request {i}…")
                socket = self.sockets[i % self.n_servers]
                socket.send(pickle.dumps(conf))
                conf_sent[socket] = conf  # Mark this config as sent to this socket
                self.n_designs += 1
            
            # Wait for all responses
            responses = {}
            timeout = 1.2e6 # 900000  # 15 minutes in milliseconds

            while len(responses) < len(conf_sub_list):
                print(f"Waiting for responses from {[self.socket_addresses[socket] for socket in self.sockets if socket not in responses]}...")
                socks = dict(self.poller.poll(timeout))
                if not socks:  # Timeout occurred
                    print("Timeout occurred, resending requests...")
                    for socket, conf in conf_sent.items():
                        if socket not in responses:  # Response not received for this socket
                            print(f"It seems that server {self.socket_addresses[socket]} is not responding...")
                            print(f"Try to replace the socket {socket}...")
                            new_socket = self._replace_socket(socket)
                            conf_sent[new_socket] = conf
                            del conf_sent[socket]
                            new_socket.send(pickle.dumps(conf))
                            break
                else:
                    for socket in conf_sent.keys():
                        if socket in socks and socks[socket] == zmq.POLLIN:
                            message = socket.recv()
                            response = pickle.loads(message)
                            responses[socket] = response  # Store response
                            conf_sent[socket] = None  # Mark as acknowledged
                            print(f"Received reply from server {socket}: {response}")
        

        ep_rew_mean_values_list = []
        for train_log_dir in train_log_dir_list:
            ep_rew_mean_values = []
            with open(os.path.join(train_log_dir, "progress.csv"), mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    ep_rew_mean_values.append(float(row["rollout/ep_rew_mean"]))
            ep_rew_mean_values_list.append(ep_rew_mean_values)
            


        # TODO: LOGGING
        self._reset_sockets()

        return ep_rew_mean_values_list
    

    def valid(self, level=1):
        """
        Check if the design pipeline is valid
        level 0: buildable
        level 1: self-collision free
        level 2: actively moveable
        """
        valid_list = []
        for design_pipeline in self.design_pipelines:
            valid_list.append(
                is_pipeline_valid(design_pipeline, 
                                  level=level, 
                                  conf_dict={"self_collision_threshold": self.self_collision_threshold, "ave_speed_threshold": self.ave_speed_threshold}
                                 )
                )

        return valid_list
    
    def is_builable(self):
        """
        Check if the design pipeline is strictly valid
        """
        valid_list = []
        for design_pipeline in self.design_pipelines:
            valid_list.append(is_pipeline_valid(design_pipeline, level=0))
        return valid_list


    def fitness(self):

        start_servers(self.n_servers + self.n_backup_servers)
        print("Servers started!")
        time.sleep(10)

        ep_rew_mean_values = self._train()
        # TODO: from config
        # learning_speed = index_of_first_greater_than(ep_rew_mean_values, 500) # the number of episodes to reach 500 reward
        # score = -learning_speed
        score = np.mean(ep_rew_mean_values, axis=1)

        if np.max(score) > self.best_fitness:
            self.best_fitness = np.max(score)
            self.best_design_pipeline = self.design_pipelines[np.argmax(score)]

        if self.wandb_on:
            wandb.log({"Fitness": wandb.Histogram(score), 
                       "EpRewMean": wandb.Histogram(ep_rew_mean_values),
                       "DesignPipelines": self.design_pipelines,
                       "BestFitness": self.best_fitness,
                       "NumDesigns": self.n_designs
                       })
            # wandb.save(os.path.join(train_log_dir, "progress.csv"))
            # wandb.save(os.path.join(train_log_dir, "rl_model.zip"))
        for i, (s, design) in enumerate(zip(score, self.design_pipelines)):
            self.csv_logger.log(f"Gen{self.generation_counter}/Design{i}", design, s)

        self.generation_counter += 1

        return score
    
    

            

def test_fitness():
    blackbox = BlackBox("evolution_debug_multigpu")

    design_pipelines = [[0,1,2,2, 1,0,2,13, 1,3,0,14], [0,1,2,2, 1,0,2,13, 1,3,0,12], [0,1,2,2, 1,0,2,13, 1,3,0,12]]
    blackbox.reset(design_pipelines)
    if all(blackbox.valid()):
        print("valid!")
    fitness = blackbox.fitness()
    print(fitness)


def test_mutate():
    blackbox = BlackBox("evolution_debug")

    design_pipelines = [[0,1,2,2, 1,0,2,13, 1,3,0,14], [0,1,2,2, 1,0,2,13, 1,3,0,14], [0,1,2,2, 1,0,2,13, 1,3,0,12]]
    blackbox.reset(design_pipelines)
    if all(blackbox.valid()):
        print("valid!")
    blackbox.mutate(mask=[0,1,2])
    print(blackbox.design_pipelines)

def test_constraint():
    blackbox = BlackBox("evolution_debug")
    design_pipelines = [[0,1,2,2, 1,0,2,13, 1,3,0,14], [0,1,2,2, 1,0,2,13, 1,3,0,14], [0,1,2,2, 1,0,2,13, 1,3,0,12]]
    blackbox.reset(design_pipelines)
    print(blackbox.valid(0))
    print(blackbox.valid(1))
    print(blackbox.valid(2))


if __name__ == "__main__":
    test_constraint()