from collections import defaultdict
import copy
import os
os.environ["MUJOCO_GL"] = "egl"
import pdb
import shutil
import time
import yaml
import numpy as np
import argparse
import gymnasium as gym
from omegaconf import OmegaConf
import sbx
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

from modular_legs import LEG_ROOT_DIR
from modular_legs.envs.gym.rendering import RecordVideo
from modular_legs.envs.env_sim import ZeroSim
from modular_legs.envs.env_real import Real
from modular_legs.envs.wrappers import VecReal
from modular_legs.utils.train import EpisodicRewardCallback, ProgressBarCallbackName, multiplex_obs, save_rollout, load_model
from modular_legs.utils.files import get_cfg_name, get_cfg_path, get_curriculum_cfg_paths, update_cfg, load_cfg, get_latest_model
from modular_legs.utils.logger import get_running_header, plot_learning_curve
from modular_legs.utils.model import is_headless
from modular_legs.utils.others import is_list_like



class Trainer:

    def __init__(self, conf_list):
        self.conf_list = conf_list
        self.is_env_setup = False
        self.is_device_set = False
        self.curriculum = False

        if "curriculum" in conf_list[0]:
            self.curriculum = True
            assert len(conf_list) == 1, "Curriculum learning only supports one configuration"
            self.conf_list = get_curriculum_cfg_paths(conf_list[0])
            curriculum_steps = [load_cfg(conf_name, alg="sbx").trainer.curriculum_step for conf_name in self.conf_list]
            # Sort the configurations based on the curriculum steps
            self.conf_list = [a for _, a in sorted(zip(curriculum_steps, self.conf_list))]


        master_conf = self.conf_list[0]

        def update_robot_log_dir(conf):
            # Update the robot data directory ONCE
            if conf.robot.mode == "real":
                if conf.logging.robot_data_dir is not None:
                    if conf.logging.robot_data_dir == "auto":
                        conf.logging.robot_data_dir = conf.logging.data_dir # One path for all the configurations
                    os.makedirs(conf.logging.robot_data_dir, exist_ok=True)
                # Also update the logging notes
                conf.trainer.notes += f"\n{get_running_header()}"

            return conf
       
        self._update_conf(master_conf, conf_update_func=update_robot_log_dir)
        self.master_conf = copy.deepcopy(self.conf)
        if self.conf.robot.mode == "real":
            self._save_conf(self.conf.logging.robot_data_dir)

        # Set up wandb
        run = wandb.init(
            project="OctopusLite",
            name=f"{self.conf.robot.mode}-{self.conf.agent.obs_version}-{self.conf.agent.reward_version}",
            config=OmegaConf.to_container(self.conf),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            # monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            mode="online" if self.conf.trainer.wandb_on else "disabled"
        )

        if self.conf.trainer.joystick and self.conf.trainer.mode == "play":
            raise NotImplementedError("Joystick is not supported yet")



    def _setup_env(self):
        # Setting up the environment
        # The environment only set up once
        if self.conf.robot.mode == "sim":
            if self.conf.sim.render and is_headless():
                self.conf.sim.render = False
                print("Running in headless mode; render is turned off!")
            self.unwarpped_env = ZeroSim(self.conf)
            self.env = gym.wrappers.TimeLimit(
                        self.unwarpped_env, max_episode_steps=1000
                    )
            
            if self.conf.trainer.num_envs >1:
                # TODO: this case in real world 
                env_funs = [lambda: Monitor(gym.wrappers.TimeLimit(ZeroSim(self.conf), max_episode_steps=1000))]*self.conf.trainer.num_envs
                self.env = DummyVecEnv(env_funs)

            elif self.conf.agent.num_envs > 1:
                tenv = gym.wrappers.TimeLimit(
                        self.unwarpped_env, max_episode_steps=1000
                    )
                trigger = lambda t: t % 199 == 0
                self.env = RecordVideo(tenv, 
                                video_folder=self.conf.logging.data_dir, 
                                episode_trigger=trigger, 
                                fps=1/self.conf.robot.dt,
                                disable_logger=True)
                self.env = VecReal(self.env, max_episode_steps=1000)

            elif not self.conf.sim.render and self.conf.trainer.mode in ["train", "play"]:
                trigger = lambda t: t % 199 == 0
                self.env = RecordVideo(self.env, 
                                video_folder=self.conf.logging.data_dir, 
                                episode_trigger=trigger, 
                                fps=1/self.conf.robot.dt,
                                disable_logger=True)
        elif self.conf.robot.mode == "real":
            self.unwarpped_env = Real(self.conf)
            if self.conf.agent.num_envs == 1:
                self.env = gym.wrappers.TimeLimit(
                            self.unwarpped_env, max_episode_steps=1000
                        )
            else:
                self.env = VecReal(self.unwarpped_env, max_episode_steps=1000)
        else:
            raise ValueError("Invalid robot mode: ", self.conf.robot.mode)
        


    def _update_conf(self, conf_name, reset_env=False, conf_update_func=None):
        self.conf = load_cfg(conf_name, alg="sbx")
        if conf_update_func is not None:
            self.conf = conf_update_func(self.conf)
        self.conf_name = conf_name

        if hasattr(self, "master_conf"):
            self.conf.interface.module_ids = self.master_conf.interface.module_ids
            self.conf.interface.torso_module_id = self.master_conf.interface.torso_module_id

        if not self.is_device_set:
            # Set the device
            device = self.conf.trainer.device
            if "cuda" in device:
                os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["JAX_PLATFORMS"] = "cpu"

        if hasattr(sbx, self.conf.trainer.algorithm):
            self.Alg = getattr(sbx, self.conf.trainer.algorithm)
        else:
            raise ValueError(f"Algorithm {self.conf.trainer.algorithm} not found in sbx")

        # Only set up the environment once 
        if (not self.is_env_setup) or reset_env:
            self._setup_env()
            self.is_env_setup = True

        # Setting up the model
        self.unwarpped_env.update_config(self.conf)

        alg_kwargs=OmegaConf.to_container(self.conf.trainer.algorithm_params) if self.conf.trainer.algorithm_params is not None else {}

        if not is_list_like(self.conf.trainer.load_run):
            self.model = load_model(self.conf.trainer.load_run, self.env, self.Alg, alg_kwargs=alg_kwargs)
            if self.conf.trainer.load_replay_buffer is not None:
                self.model.load_replay_buffer(self.conf.trainer.load_replay_buffer)
            self.models = [self.model]
            self.vec_env = self.model.get_env()
        else:
            self.models = [load_model(load_run, self.env, self.Alg, alg_kwargs=alg_kwargs) for load_run in self.conf.trainer.load_run]
            self.vec_env = self.models[0].get_env()
            self.model = self.models[0]


    def _save_conf(self, log_dir):
        shutil.copy(get_cfg_path(self.conf_name), log_dir)
        with open(os.path.join(log_dir, "running_config.yaml"), "w") as file:
            yaml.dump(OmegaConf.to_container(self.conf, resolve=True), file, default_flow_style=False)
        with open(os.path.join(log_dir, "note.txt"), 'w') as f:
            f.write(self.conf.trainer.notes)
        asset_files = [self.conf.sim.asset_file] if not is_list_like(self.conf.sim.asset_file) else self.conf.sim.asset_file
        asset_log_dir = os.path.join(log_dir, "assets")
        os.makedirs(asset_log_dir, exist_ok=True)
        for asset_file in asset_files:
            xml_file = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", asset_file)
            shutil.copy(xml_file, asset_log_dir)

    def _train(self):
        last_log_dir = None
        for i, conf_name in enumerate(self.conf_list):

            def update_log_dir(conf):
                conf.trainer.load_run = last_log_dir
                return conf

            if not i == 0:
                self._update_conf(conf_name, reset_env=True, conf_update_func=update_log_dir if self.curriculum else None)

            # Setting up the logger
            log_dir =self.conf.logging.data_dir
            logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
            checkpoint_callback = CheckpointCallback(
                save_freq=100000 if self.conf.robot.mode == "sim" else int(2000*self.conf.agent.num_envs),
                save_path=log_dir,
                name_prefix="rl_model",
                save_replay_buffer=self.conf.robot.mode == "real",
                save_vecnormalize=True,
                )
            wandb_callback = WandbCallback(log="all")
            self._save_conf(log_dir)

            self.model.set_logger(logger)
            rich_callback = ProgressBarCallbackName(get_cfg_name(conf_name))
            callbacks = [checkpoint_callback, wandb_callback, rich_callback] if self.conf.robot.mode == "sim" else [checkpoint_callback, wandb_callback]
            if self.conf.agent.num_envs > 1:
                callbacks.append(EpisodicRewardCallback())

            # Training the model
            self.model.learn(total_timesteps=self.conf.trainer.total_steps, 
                        callback=callbacks) # TODO: progress bar for real world training
            
            self.model.save(os.path.join(log_dir, "rl_model_last.zip"))

            plot_learning_curve(os.path.join(log_dir, "progress.csv"), os.path.join(log_dir, "curve.png"))
            
            last_log_dir = get_latest_model(log_dir)


    def _play(self):
        self.recovering = False
        self.multiplexing = self.conf.trainer.multiplex
        self.unwarpped_env.commands = [0,1,0]
        obs = self.vec_env.reset()
        step_count = 0
        while True:
            t0 = time.time()

            if self.multiplexing:
                if self.conf.trainer.multiplex_type == "4+1":
                    obs_tuple = multiplex_obs(obs, "4+1")
                    actions = []
                    for obs, model in zip(obs_tuple, self.models):
                        act, _states = model.predict(obs, deterministic=True)
                        actions.append(act)
                    action = np.concatenate((actions[0][:,:3], actions[1],actions[0][:,3:4]), axis=1)

                elif self.conf.trainer.multiplex_type == "3+1+1":
                    obs_tuple = multiplex_obs(obs, "3+1+1")
                    actions = []
                    for obs, model in zip(obs_tuple, self.models):
                        act, _states = model.predict(obs, deterministic=True)
                        actions.append(act)
                    action = np.concatenate((actions[0][:,:3], actions[1],actions[2]), axis=1)

            else:

                action, _states = self.model.predict(obs, deterministic=True)


            obs, reward, done, info = self.vec_env.step(action)

            # Switching the policy according to priority
            # print("Current policy: ", self.conf_name)
            policy_switch = info[0]["policy_switch"]
            upsidedown = info[0]["upsidedown"]
            chopped = info[0]["chopped"]

            joystick_data = self.joystick_server.pull_data() if self.conf.trainer.joystick and self.conf.trainer.mode == "play" else None
            joystick_command = self.joystick_compiler.get_command(joystick_data) if joystick_data is not None else None
            # print("Joystick command: ", joystick_command)
            if upsidedown and self.conf.trainer.auto_recovery:
                # Switch to the recovery policy if the robot is upside down
                conf_name = self.conf.trainer.recovery_config
                self._update_conf(conf_name)
                self.recovering = True
                obs = self.vec_env.reset()
                self.start_recovery = time.time()

            elif self.recovering and time.time() - self.start_recovery > self.conf.trainer.recovery_time and not upsidedown:
                # Switch back to the original policy after 3 seconds of recovery
                conf_name = self.conf_list[0]
                self._update_conf(conf_name)
                obs = self.vec_env.reset()
                self.recovering = False

            elif self.conf.trainer.monitored_module in chopped and self.conf.trainer.auto_multiplex:
                print("chopped: ", chopped)
                conf_name = self.conf.trainer.multiplex_config
                self._update_conf(conf_name)
                obs = self.vec_env.reset()
                self.multiplexing = True

            elif policy_switch is not None:
                # Switch to the policy specified by keyboad input
                print(f"Policy switch: {policy_switch}")
                assert isinstance(policy_switch, int), "Policy switch must be an integer"
                self.vec_env.close()
                conf_name = self.conf_list[policy_switch]
                self._update_conf(conf_name)
                obs = self.vec_env.reset()

            elif joystick_command is not None:
                print(f"Joystick command: {joystick_command}")
                # Switching the policy according to joystick input
                if joystick_command == "right_bumper":
                    # jumpCCW
                    conf_name = self.conf.trainer.candidate_configs[0]
                elif joystick_command == "left_bumper":
                    # jump
                    conf_name = self.conf.trainer.candidate_configs[1]
                elif joystick_command == "right_trigger" or joystick_command == "left_trigger":
                    # Walking policy
                    conf_name = self.conf.trainer.candidate_configs[2]
                # elif :
                #     conf_name = "real_play_quadrupedX4air1s_back"
                elif joystick_command == "neutral":
                    conf_name = self.conf_list[0]
                else:
                    conf_name = self.conf_list[0]

                self._update_conf(conf_name)
                obs = self.vec_env.reset()
                self.unwarpped_env.commands[0] = 1
                
                
            if self.conf.robot.mode == "sim" and self.conf.sim.render:
                time.sleep(max(0, t0 + self.conf.robot.dt - time.time()))

            step_count += 1

    def _record(self):

        for conf_name in self.conf_list:
            conf = load_cfg(conf_name, alg="sbx")

            record_obs_version = conf.trainer.record.obs_version

            num_envs = conf.trainer.record.num_envs
            batch_steps = int(conf.trainer.record.record_steps // num_envs)
            def make_env():
                return gym.wrappers.TimeLimit(
                    ZeroSim(conf), max_episode_steps=1000
                )
            model = load_model(conf.trainer.load_run, make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv), self.Alg)
            vec_env = model.get_env()
            obs = vec_env.reset()
            constructed_obs = [vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_version) for i in range(len(vec_env.envs))] if record_obs_version is not None else obs

            rollout = defaultdict(list)

            t0 = time.time()
            save_dir = os.path.dirname(conf.trainer.load_run)
            from rich.progress import Progress
            progress = Progress()
            progress.start()
            task = progress.add_task("[red]Recording...", total=batch_steps)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                rollout["observations"].append(constructed_obs)
                act_recorded = action if not conf.trainer.record.normalize_default_pos else action+np.array(conf.agent.default_dof_pos)
                rollout["actions"].append(act_recorded)
                obs, reward, done, info = vec_env.step(action)
                constructed_obs = [vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_version) for i in range(len(vec_env.envs))] if record_obs_version is not None else obs
                # print(done)
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                # if done[0]:
                    # step_count = 0
                    # obs = vec_env.reset()
                print(f"{len(rollout['observations'])} / {batch_steps}")
                progress.update(task, advance=1)

                if time.time() - t0 > 60*30:
                    print(f"Recording trajectories... ({len(rollout['observations'])} steps)")
                    # np.savez_compressed(os.path.join(save_dir, f"rollout.npz"), **rollout)
                    save_rollout(rollout, save_dir)
                    t0 = time.time()
                if len(rollout['observations']) >= batch_steps:
                    print(f"Finished recording trajectories ({len(rollout['observations'])} steps). Saving to {save_dir}...")
                    save_rollout(rollout, save_dir)
                    print("Done!")
                    break

            progress.stop()




    def run(self):
            
        mode = self.conf.trainer.mode

        if mode == "train":
            # Training the model
            self._train()
            
        elif mode == "play":
            # Testing the model
            self._play()

        elif mode == "record":
            # Recording the trajectories
            num_workers = self.conf.trainer.record.num_workers
            if num_workers > 1:
                raise NotImplementedError("Recording with multiple workers is not supported yet")
            else:
                self._record()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', nargs='+', default=['sim_train_m3air1s'])
    args = parser.parse_args()

    trainer = Trainer(args.cfg)
    trainer.run()
    

