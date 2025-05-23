import copy
import os
import pdb
import time

from omegaconf import OmegaConf
import gymnasium as gym
from gymnasium.spaces.box import Box
import numpy as np
import threading

import torch

from modular_legs import LEG_ROOT_DIR
from modular_legs.utils.action_filter import ActionFilterButter, ActionFilterMelter
from modular_legs.envs.brain import Brain
from modular_legs.utils.kbhit import KBHit
from modular_legs.utils.others import is_list_like, is_number



'''
An base environment for both sim / real
'''
class RealSim(gym.Env):
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg) # Copy the original config, which might be modified
        self.update_config(self.cfg)
        self.kb = KBHit()
        self.input_key = ""
        

    def update_config(self, cfg: OmegaConf):
        
        self.num_act = cfg.agent.num_act
        self.num_obs = int(cfg.agent.num_obs * cfg.agent.include_history_steps)
        self.num_envs = cfg.agent.num_envs
        # if self.num_envs > 1:
        #     assert cfg.robot.mode == "real", "Only real robot supports multiple environments" # TODO
        action_space = cfg.agent.clip_actions if cfg.agent.action_space is None else cfg.agent.action_space
        self.action_space = Box(-action_space, action_space, (cfg.agent.num_act,))
        self.observation_space = Box(-np.inf, np.inf, (self.num_obs,))
        self.action_remap_type = cfg.agent.action_remap_type
        self.command_x_choices = cfg.agent.command_x_choices
        self.commands_ranges = cfg.agent.commands_ranges
        self.play = cfg.trainer.mode == "play"
        self.clip_actions = cfg.agent.clip_actions
        self.clip_actions_min = -cfg.agent.clip_actions
        self.clip_actions_list = np.array(cfg.agent.clip_actions_list) if cfg.agent.clip_actions_list is not None else None
        assert self.clip_actions <= action_space
        self.resampling_time = cfg.agent.resampling_time
        
        self._dt = cfg.robot.dt
        self.theta = cfg.robot.theta
        self.kp, self.kd = cfg.robot.kp, cfg.robot.kd
        if not is_list_like(self.kp):
            self.kps, self.kds = np.array([self.kp]*(self.num_act*self.num_envs), dtype=np.float32), np.array([self.kd]*(self.num_act*self.num_envs), dtype=np.float32)
        else:
            self.kps, self.kds = np.array(self.kp, dtype=np.float32), np.array(self.kd, dtype=np.float32)
        self.action_scale = cfg.agent.action_scale
        
        self.default_dof_pos = np.array(cfg.agent.default_dof_pos) if is_list_like(cfg.agent.default_dof_pos) else cfg.agent.default_dof_pos
        self.control_mode = cfg.agent.control_mode # "position" or "incremental"
        self.frozen_joints = cfg.robot.frozen_joints

        self.last_action = np.zeros(self.num_act*self.num_envs)
        self.last_action_flat = np.zeros(self.num_act*self.num_envs)
        self.commands = np.zeros(3)
        self.ang_vel_sum = 0
        self.step_info = None

        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        self.brain = Brain(cfg)
        self.filter_action = cfg.agent.filter_action
        self.action_filter = ActionFilterButter(sampling_rate=1/self.dt,
                                                num_joints=self.num_act*self.num_envs,
                                                highcut=[3.0])
        self.action_melter = ActionFilterMelter(cfg.agent.action_melter_axis) if cfg.agent.action_melter else None
        self.policy_switch = None # Sent through the step info

    @property
    def dt(self) -> float:
        return self._dt
    
    def _wait_until_motor_on(self):
        pass

    def _reset_robot(self):
        pass

    def _get_observable_data(self):
        raise NotImplementedError
    
    def _update_observable_data(self):
        self.observable_data = self._get_observable_data()
        # print("Sending LAST ACTION: ", self.last_action_flat)
        self.observable_data["last_action"] = self.last_action_flat
        self.observable_data["commands"] = self.commands
        self.brain.update_state2(self.observable_data)
        self.brain.update_visualization()


    def reset(self, seed=None, options=None):
        self._wait_until_motor_on()
        self._reset_robot()
        self.action_filter.reset()
        if self.action_melter is not None:
            self.action_melter.reset()

        self._resample_commands()
        
        self._update_observable_data()
        obs = self.brain.get_observations(reset=True)
        if self.num_envs > 1:
             obs = np.reshape(obs, (self.num_envs, -1)) # Signals in Brain are kept flattened
        # print("RESET !!!! obs shape: ", obs.shape)

        self.rewards = []


        self.action_filter.init_history(self.brain.dof_pos)
        self.t0 = time.time()
        self.step_count = 0
        self.ang_vel_sum = 0
        return obs, {}
    

    def _resample_commands(self):
        if self.command_x_choices is not None and is_list_like(self.command_x_choices):
            if not self.play:
                command_x = np.random.choice(self.command_x_choices)
            else:
                command_x = self.command_x_choices[0] # TODO
            self.commands[0] = command_x
            print(f"command_x: {command_x}")
        elif self.command_x_choices == "one_hot":
            if not self.play:
                self.commands = np.zeros(3)
                self.commands[np.random.randint(3)] = 1
            print(f"self.commands: {self.commands}")

        if self.commands_ranges is not None:
            if is_list_like(self.commands_ranges):
                for i in range(len(self.commands_ranges)):
                    self.commands[i] = np.random.uniform(self.commands_ranges[i][0], self.commands_ranges[i][1])
            else:
                self.commands, info = self.brain.get_custom_commands(self.commands_ranges)
                if "clip_actions" in info:
                    self.clip_actions = info["clip_actions"]
                    self.clip_actions_min = info["clip_actions_min"]
                print(f"Clip actions: {self.clip_actions_min}, {self.clip_actions}")
                if "default_dof_pos" in info:
                    self.default_dof_pos = info["default_dof_pos"]
                    print(f"Default dof pos: {self.default_dof_pos}")


    def _perform_action(self, pos, vel=None, kps=None, kds=None):
        raise NotImplementedError

    def _is_truncated(self):
        return False
    
    def _log_data(self):
        pass

    def _check_input(self):
        if self.kb.kbhit():
            self.input_key = self.kb.getch()
            if is_number(self.input_key):
                self.policy_switch = int(self.input_key)
            if self.input_key == "j":
                print("Jump!")
                self.commands[0] = 1

    def _action_remap(self, action):
        if self.action_remap_type is None:
            return action
        else:
            type_params = self.action_remap_type.split("_")
            if type_params[0] == "repeat":
                assert self.num_act == 1
                num_repeat = int(type_params[1])
                return np.repeat(action, num_repeat)
            elif type_params[0] == "invrepeat":
                repeat_idx = int(type_params[1])
                new_arr = np.insert(action, repeat_idx + 1, -action[repeat_idx])
                return new_arr
            elif type_params[0] == "wheeleg":
                # A very specific remap for the wheel mode of quadruped robot
                assert self.control_mode == "advanced", "wheeleg only works with advanced control mode"
                vel = 8
                kp_pos = self.kps[0]
                kd_pos = self.kds[0]
                kd_vel = 4
                self.set_step_info({"pos":np.array([action[0],0,0,action[1],action[2]])+self.default_dof_pos, 
                                    "vel":np.array([0,vel,-vel,0,0]), 
                                    "kps":np.array([kp_pos,0,0,kp_pos,kp_pos]), 
                                    "kds":np.array([kd_pos,kd_vel,kd_vel,kd_pos,kd_pos])
                                    })

                return None


    def set_step_info(self, step_info):
        self.step_info = step_info

    def step(self, action, clip=True):

        if self.num_envs > 1:
            return self.step_batch(action, clip=clip)

        self._wait_until_motor_on()
        actions_scaled = action * self.action_scale
        if clip:
            if self.clip_actions_list is None:
                action_cliped = np.clip(actions_scaled, self.clip_actions_min, self.clip_actions) 
            else:
                action_cliped = np.clip(actions_scaled, -self.clip_actions_list, self.clip_actions_list)
        else:
            action_cliped = actions_scaled

        self.last_action = action_cliped # last_action should be before action remap
                                         # Not that currently self.default_dof_pos is added after the last_action is recorded
        self.last_action_flat = action_cliped.flatten()
        joint_action = self._action_remap(action_cliped) # num_actions and num_joint are matched here; action_remap should be before incremental calculation
        if self.frozen_joints is not None:
            joint_action[self.frozen_joints] = 0

        if self.control_mode == "position":
            target_action = joint_action + self.default_dof_pos
            target_action = self.action_filter.filter(target_action) if self.filter_action else target_action
            target_action = self.action_melter.filter(target_action) if self.action_melter is not None else target_action
            # Send action to the motor/wait or simulate
            # print(f"action: {action}   -->  target_action: {target_action}")
            info = self._perform_action(target_action)
        elif self.control_mode == "incremental":
            target_action = joint_action + self.brain.dof_pos
            target_action = self.action_filter.filter(target_action) if self.filter_action else target_action
            info = self._perform_action(target_action)
        elif self.control_mode == "velocity":
            # Velocity control
            # The position will be ignored in the motor
            # For now, only support pure velocity control, consistent with the real robot
            target_action = joint_action
            self.kp = 0
            info = self._perform_action(pos=np.zeros_like(target_action), vel=target_action)
        elif self.control_mode == "advanced":
            # Directly do the P control and D control
            assert self.step_info is not None, "set_step_info should be called before step"
            info = self._perform_action(pos=self.step_info["pos"], 
                                        vel=self.step_info["vel"],
                                        kps=self.step_info["kps"] if "kps" in self.step_info else self.kps,
                                        kds=self.step_info["kds"] if "kds" in self.step_info else self.kds
                                        )
        
        if self.step_count % self.resampling_time == 0 and self.step_count > 0:
            self._resample_commands()
        # print(f"Action range: {self.clip_actions_min}, {self.clip_actions}")
            
        # self._update_obs()
        self._update_observable_data()
        obs = self.brain.get_observations()

        reward, reward_info = self.brain.get_reward()

        self.step_count +=1

        truncated = self._is_truncated()

        self._log_data()
        self._check_input()

        info_extra = {
                      "policy_switch": self.policy_switch,
                      "upsidedown": self.brain.is_upsidedown(),
                    #   "chopped": self.brain.is_chopped(), 
                    }
        info.update(info_extra)
        info.update(reward_info)


        return obs, reward, self.brain.is_done(), truncated, info



    def step_batch(self, actions, clip=True):

        if hasattr(self, "t00"):
            print("TIME-1: ", time.time()-self.t00)

        t0 = time.time()
        self._wait_until_motor_on()
        assert actions.shape[0] == self.num_envs, f"actions.shape[0] should be {self.num_envs}, but got {actions.shape[0]}"
        assert actions.shape[1] == self.num_act, f"actions.shape[1] should be {self.num_act}, but got {actions.shape[1]}"
        
        actions_scaled = actions * self.action_scale
        if clip:
            if self.clip_actions_list is None:
                action_cliped = np.clip(actions_scaled, self.clip_actions_min, self.clip_actions) 
            else:
                action_cliped = np.clip(actions_scaled, -self.clip_actions_list, self.clip_actions_list)
        else:
            action_cliped = actions_scaled

        # t1 = time.time()
        # print("TIME1: ", t1-t0)

        self.last_action = copy.deepcopy(action_cliped) # last_action should be before action remap
        self.last_action_flat = action_cliped.flatten()
        self.last_action = action_cliped # last_action should be before action remap
                                         # Not that currently self.default_dof_pos is added after the last_action is recorded
        joint_action = self._action_remap(action_cliped) # num_actions and num_joint are matched here; action_remap should be before incremental calculation
        if self.frozen_joints is not None:
            joint_action[:, self.frozen_joints] = 0

        if self.control_mode == "position":
            target_action = joint_action + self.default_dof_pos
            if self.filter_action:
                target_action_flatten = target_action.flatten()
                target_action_flatten = self.action_filter.filter(target_action_flatten)
                target_action = target_action_flatten.reshape(target_action.shape)

            if self.action_melter:
                raise NotImplementedError("Batch action melter is not implemented yet")
            # target_action = self.action_melter.filter(target_action) if self.action_melter is not None else target_action
            # Send action to the motor/wait or simulate
            info = self._perform_action(target_action.flatten())
        # elif self.control_mode == "incremental":
        #     target_action = joint_action + self.brain.dof_pos
        #     target_action = self.action_filter.filter(target_action) if self.filter_action else target_action
        #     info = self._perform_action(target_action)
        # elif self.control_mode == "velocity":
        #     # Velocity control
        #     # The position will be ignored in the motor
        #     # For now, only support pure velocity control, consistent with the real robot
        #     target_action = joint_action
        #     self.kp = 0
        #     info = self._perform_action(pos=np.zeros_like(target_action), vel=target_action)
        # elif self.control_mode == "advanced":
        #     # Directly do the P control and D control
        #     assert self.step_info is not None, "set_step_info should be called before step"
        #     info = self._perform_action(pos=self.step_info["pos"], 
        #                                 vel=self.step_info["vel"],
        #                                 kps=self.step_info["kps"] if "kps" in self.step_info else self.kps,
        #                                 kds=self.step_info["kds"] if "kds" in self.step_info else self.kds
        #                                 )
        
        if self.step_count % self.resampling_time == 0 and self.step_count > 0:
            self._resample_commands()
        # print(f"Action range: {self.clip_actions_min}, {self.clip_actions}")
        # t2 = time.time()
        # print("TIME2: ", t2-t1)
        # self._update_obs()
        self._update_observable_data()
        # t3 = time.time()
        # print("TIME3: ", t3-t2)
        obs = self.brain.get_observations()
        obs = np.reshape(obs, (self.num_envs, -1))
        # print("!!!! obs shape: ", obs.shape)
        # t4 = time.time()
        # print("TIME4: ", t4-t3)

        r_start_time = time.time()
        reward, reward_info = self.brain.get_reward()
        reward = np.reshape(reward, (self.num_envs))
        print("REWARD TIME: ", time.time()-r_start_time)

        self.step_count +=1

        tr=self._is_truncated()
        truncated = np.array([tr]*self.num_envs)

        # tstartlog = time.time()
        self._log_data()
        self._check_input()
        # print("loging time: ", time.time() - tstartlog)

        # info_extra = {"rewards": reward_info, 
        #               "policy_switch": self.policy_switch,
        #               "upsidedown": self.brain.is_upsidedown(),
        #               "chopped": self.brain.is_chopped(), # TODO: Only PosC chopped is implemented
        #             }
        # info.update(info_extra)
        # self.rewards.append(reward)


        info_all = reward_info
        d = self.brain.is_done()
        done = np.array([d]*self.num_envs)

        # if np.any(done):

        #     for i in range(self.num_envs):
        #         ep_rew = np.sum(np.array(self.rewards)[:,i])
        #         ep_len = len(np.array(self.rewards)[:,i])
        #         info_all[i]["episode"] = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                
        # t5 = time.time()
        # print("TIME5: ", t5-t4)

        # self.t00 = time.time()


        return obs, reward, done, truncated, info_all



