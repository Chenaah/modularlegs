import copy
import numbers
import os
import pdb
import time
import numpy as np
import torch
import wandb
from rich.table import Table
from rich.live import Live
from modular_legs.sim.robot_designer import DEFAULT_ROBOT_CONFIG
from modular_legs.utils.model import get_jing_vector, get_local_zvec
from modular_legs.utils.obs_buffer import ObservationBuffer
from modular_legs.utils.math import AverageFilter, normalize_angle, quat_rotate, quat_rotate_inverse, quat_apply, ang_vel_to_ang_forward
from modular_legs import LEG_ROOT_DIR
from modular_legs.utils.curves import forward_reward_curve2_1, isaac_reward, plateau
from modular_legs.utils.others import is_list_like
'''
    This class should only take care of the observation and reward and the visualization of them.
'''

def sample_n_variables_simplex(N):
    x = np.random.dirichlet(np.ones(N)) * 2 - 1  # Scale to [-1,1] while ensuring sum = 1
    return x



class Brain():

    def __init__(self, cfg):

        # Default config
        self.cfg = cfg
        self.obs_version = cfg.agent.obs_version
        self.reward_version = cfg.agent.reward_version
        self.done_version = cfg.agent.done_version
        self.include_history_steps = cfg.agent.include_history_steps
        self.num_act = cfg.agent.num_act
        self.num_joint = len(cfg.agent.default_dof_pos) if is_list_like(cfg.agent.default_dof_pos) else self.num_act # TODO 
        self.num_obs = cfg.agent.num_obs
        self.num_envs = cfg.agent.num_envs
        self.sim_init_pos = cfg.sim.init_pos
        self.sim_init_quat = cfg.sim.init_quat
        self.obs_scales = {"lin_vel": 2.0,
                           "lin_acc": 0.25,
                           "ang_vel": 0.25,
                           "dof_pos": 1.0,
                           "dof_vel": 0.05,
                           "height_measurements": 5.0
                           }
        self.clip_observations = cfg.agent.clip_observations
        self.clip_actions = cfg.agent.clip_actions
        self.action_scale = cfg.agent.action_scale
        self.device = "cpu"
        self.default_dof_pos = cfg.agent.default_dof_pos
        self.gravity_vec = np.array(cfg.agent.gravity_vec)
        self.forward_vec = np.array(cfg.agent.forward_vec)
        self.commands = np.zeros(3)
        self.print_data = cfg.logging.print_data
        self.projected_forward_vec = cfg.agent.projected_forward_vec
        self.projected_upward_vec = cfg.agent.projected_upward_vec
        self.torso_node_id = cfg.agent.torso_node_id
        self.predefined_commands = cfg.agent.predefined_commands
        self.commands_ranges = cfg.agent.commands_ranges
        self.reward_params = cfg.agent.reward_params
        if self.projected_forward_vec is not None:
            self.projected_forward_vec = np.array(self.projected_forward_vec, dtype=float)
        if self.projected_upward_vec is not None:
            self.projected_upward_vec = np.array(self.projected_upward_vec, dtype=float)

        self.theta = cfg.robot.theta
        self.dt = cfg.robot.dt

        self.normalization = {"clip_observations": self.clip_observations,
                              "clip_actions": self.clip_actions}
        
        self.train_total_steps = cfg.trainer.total_steps # Used for curriculum-related signals

        # Logger
        # if logger is not None:
        #     self.logger = logger
        # else:
        #     self.logger = Logger(alg="RL")


        # Visualization
        self.info_dict = {}
        self.status_dict = {}
        if self.print_data:
            self.live = Live(self._generate_table(), refresh_per_second=20)
            self.live.__enter__()

        # Initialization
        self.commands_scale = np.array([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]]) 
        self.pos_world = np.zeros(3)
        self.vel_body = np.zeros(3)
        self.vel_world = np.zeros(3)
        self.acc_body = np.zeros(3)
        self.acc_world = np.zeros(3)
        self.ang_vel_body = np.zeros(3)
        self.ang_vel_world = np.zeros(3)
        self.projected_gravity = np.zeros(3)
        self.projected_forward = np.zeros(3)
        self.heading = np.zeros(1)
        self.quat = np.zeros(4)
        self.dof_pos = np.zeros(self.num_act)
        self.dof_vel = np.zeros(self.num_act)
        self.dof_torque = np.zeros(self.num_act)
        self.height = np.zeros(1)
        self.actions = np.zeros(self.num_act)
        self.last_action = np.zeros(self.num_act*self.num_envs)
        self.last_last_action = np.zeros(self.num_act*self.num_envs)
        self.step_counter = 0
        # self._construct_obs() # get num_obs / include_history_steps

        self.obs_buf = ObservationBuffer(self.num_obs*self.num_envs, self.include_history_steps)
        

    def update_visualization(self):
        if self.print_data:
            # This is only useful for running real robot
            self._log_info()
            self._diagnose()

            self.live.update(self._generate_table())

    def update_state2(self, data):
        for key, value in data.items():
            setattr(self, key, value)

        # Prpcess data
        self.height = np.expand_dims(self.pos_world[2], axis=0)
        self.projected_gravity = quat_rotate_inverse(self.quat, self.gravity_vec)
        self.projected_gravities = [quat_rotate_inverse(quat, self.gravity_vec) for quat in self.quats]
        # self.projected_forward = quat_rotate_inverse(self.quat, self.forward_vec)
        forward = quat_apply(self.quat, self.forward_vec)
        self.heading = np.expand_dims(np.arctan2(forward[1], forward[0]), axis=0)

        if isinstance(self.dof_pos, numbers.Number):
            self.dof_pos = np.array([self.dof_pos])
        if isinstance(self.dof_vel, numbers.Number):
            self.dof_vel = np.array([self.dof_vel])

        # For viusalization
        self.observable_data = data
        self.observable_data["projected_gravity"] = self.projected_gravity
        self.observable_data["heading"] = self.heading
        self.observable_data["dof_pos"] = self.dof_pos
        self.observable_data["dof_vel"] = self.dof_vel

    def _log_info(self):
        self.info_dict["robot"] = f"{self.cfg.robot.mode} - {self.num_act} modules"
        self.info_dict["last_action"] = self.last_action
        self.info_dict["commands"] = self.commands
        for k in self.observable_data:
            self.info_dict[k] = self.observable_data[k]

    def _diagnose(self):
        self.status_dict = dict.fromkeys(self.info_dict, "[green]NORMAL")
        # TODO: wait for new hardware

    def _generate_table(self) -> Table:
        """Make a new table."""
        table = Table()
        table.add_column("Parameter")
        table.add_column("Value")
        table.add_column("Status")
        end_keys = ["Enable", "Motor Torque"]

        for key, value in self.info_dict.items():
            if isinstance(value, float):
                value = f"{value:3.3f}"
            with np.printoptions(precision=3, suppress=True):
                table.add_row(key, f"{value}", self.status_dict[key], end_section=key in end_keys)

        return table


    def _construct_obs(self, obs_version=None):

        if obs_version is None:
            obs_version = self.obs_version

        if obs_version == "robust_proprioception":
            obs = np.concatenate((  self.projected_gravity,
                                    self.ang_vel_body,
                                    np.cos(self.dof_pos),
                                    self.dof_vel,
                                    self.last_action
                                    ))
            
        elif obs_version == "robust_proprioception_1cmd":
            obs = np.concatenate((  self.projected_gravity,
                                    self.ang_vel_body,
                                    np.cos(self.dof_pos),
                                    self.dof_vel,
                                    self.last_action,
                                    self.commands[:1]
                                    ))
            
            
        elif obs_version == "full_local":
            obs = np.concatenate((  self.vel_body,
                                    self.projected_gravity,
                                    self.ang_vel_body,
                                    np.cos(self.dof_pos),
                                    self.dof_vel,
                                    self.last_action
                                    ))
            
            
        elif obs_version == "sensed_proprioception":
            obs = []
            # print("--> self.last_action: ", self.last_action)
            # print("--> self.projected_gravities: ", self.projected_gravities)
            for i in range(self.num_act*self.num_envs):
                obs_i = np.concatenate((self.projected_gravities[i],
                                        self.gyros[i],
                                        np.cos(self.dof_pos[i:i+1]),
                                        self.dof_vel[i:i+1],
                                        self.last_action[i:i+1]
                                        ))
                obs.append(obs_i)
            # print("self.last_action: ", self.last_action)
            obs = np.concatenate(obs)
            # print("OBS SHAPE: ", obs.shape)
            # print("self.num_act*self.num_envs ", self.num_act*self.num_envs)



        elif obs_version == "centralized_proprioception":
            obs = np.concatenate((  self.projected_gravities[self.torso_node_id],
                                    self.gyros[self.torso_node_id],
                                    np.cos(self.dof_pos),
                                    self.dof_vel,
                                    self.last_action
                                    ))
            
        elif obs_version == "sensed_proprioception_lite":
            obs = []
            for i in range(self.num_act):
                obs_i = np.concatenate((self.projected_gravities[i], # 3
                                        self.gyros[i], # 3
                                        np.cos(self.dof_pos[i:i+1]), # 1
                                        self.dof_vel[i:i+1] # 1
                                        ))
                obs.append(obs_i)
            obs = np.concatenate(obs)

            
        else:
            raise NotImplementedError
            
        
        return obs
    
    def get_observations(self, insert=True, reset=False):

        if reset:
            self.last_action = np.zeros(self.num_act*self.num_envs)
            self.last_last_action = np.zeros(self.num_act*self.num_envs)
            self.reset_reward_state()
            self.reset_state_state()

        obs = self._construct_obs()

        obs = np.clip(obs, -self.normalization["clip_observations"], self.normalization["clip_observations"])

        # assert obs.shape[0] == self.num_obs, f"You said the obs size is {self.num_obs} but it is {obs.shape[0]}" # TODO

        if reset:
            self.obs_buf.reset(obs)
        elif insert:
            self.obs_buf.insert(obs)
        policy_obs = self.obs_buf.get_obs_vec(np.arange(self.include_history_steps))


        return policy_obs
    
    def get_stacked_observations(self):
        policy_obs = self.obs_buf.get_obs_vec(np.arange(self.include_history_steps))
        return policy_obs
    
    def reset_obs_stack(self):
        obs = self.get_single_observations()
        self.obs_buf.reset(obs)

    def insert_obs_stack(self):
        obs = self.get_single_observations()
        self.obs_buf.insert(obs)
    
    def get_single_observations(self):

        obs = self._construct_obs()

        obs = np.clip(obs, -self.normalization["clip_observations"], self.normalization["clip_observations"], dtype=np.float32)

        assert obs.shape[0] == self.num_obs

        return obs

        
    def get_reward(self, reward_version=None):

        if reward_version is None:
            reward_version = self.reward_version

        info = {}

        if reward_version == "cheat_isaac_general":
            # Encourage the robot to gallop (or not)
            # reward_params: weights of [lin_vel_reward, ang_vel_reward, action_rate, fly_reward, num_balls_fall]
            # predefined_commands: [forward_command, desired_ang_vel, allowed_num_contacts]
            assert self.projected_upward_vec is not None, "projected_upward_vec is None"
            assert self.projected_forward_vec is not None, "projected_forward_vec is None"

            # Set locomotion commands
            predefined_commands = [1, # desired forward velocity
                                   0, # desired z angular velocity
                                   1,  # allowed number of feet contacting the floor
                                   0.15, # tracking sigma
                                   -1 # desired height
                                   ]
            if self.predefined_commands is not None:
                predefined_commands[:len(self.predefined_commands)] = self.predefined_commands
            command = predefined_commands[0]
            desired_ang_vel = predefined_commands[1]
            allowed_num_contacts = predefined_commands[2]
            tracking_sigma = predefined_commands[3]
            desired_height = predefined_commands[4] if predefined_commands[4] != -1 else self.sim_init_pos[2]

            # Curriculum learning for the forward velocity
            # if self.step_counter < 2e5:
            #     command = min(0.6, command)
            # self.step_counter += 1

            # Action rate
            action_rate = np.sum(np.square(self.last_last_action - self.last_action)) /self.num_act
            self.last_last_action = self.last_action.copy()

            # Tracking the forward velocity
            projected_forward_vel = np.dot(self.accurate_vel_body, self.projected_forward_vec)
            lin_vel_error = np.sum(np.square(command - projected_forward_vel))
            
            lin_vel_reward = np.exp(-lin_vel_error/tracking_sigma)

            # Tracking the z angular velocity
            accurate_projected_gravity = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            # projected_z_ang = np.dot(self.accurate_ang_vel_body, self.projected_upward_vec)
            projected_z_ang = np.dot(self.accurate_ang_vel_body, accurate_projected_gravity)
            ang_vel_error = np.sum(np.square(desired_ang_vel - projected_z_ang))
            ang_vel_reward = np.exp(-ang_vel_error/tracking_sigma)

            # Penalize the robot feet touching the floor
            for key in self.contact_counter:
                self.contact_counter[key] += 1
            for c in self.contact_floor_socks:
                self.contact_counter[c] = 0
            if len(self.contact_floor_socks) >= allowed_num_contacts+1 or len(self.contact_floor_balls):
                # Two or more legs contacting the floor is not encouraged
                self.contact_counter = dict.fromkeys(self.contact_counter, 0)
            feet_air_time = np.array([value for key, value in self.contact_counter.items()])*self.dt
            fly_reward = np.sum(feet_air_time)

            # Penalize each module touching the floor
            num_balls_fall = len(self.contact_floor_balls)

            # Penalize dof velocities too close to the limit
            dof_vel_limits = 10
            dof_penalty = np.sum((np.abs(self.dof_vel) - dof_vel_limits).clip(0, 1e5))

            # Penalize dof accelerations
            dof_acc_penalty = np.sum(np.square((self.last_dof_vel - self.dof_vel) / self.dt))
            self.last_dof_vel = self.dof_vel.copy()

            # Encourage the robot to jump
            upward_vel = np.dot(self.accurate_vel_body, -accurate_projected_gravity)
            jump_reward = np.clip(upward_vel, 0, 1)

            # Encourage the robot not to fall
            upward_reward = np.dot(self.projected_upward_vec, -accurate_projected_gravity)

            # Encourage the robot to track a height
            height = self.accurate_pos_world[2]
            height_track_reward = isaac_reward(desired_height, height, 0.005)
            # print("height_track_reward: ", height_track_reward)


            # Penalize the torso touching the floor
            torso_touch_floor = np.any([self.mj_model.geom(geom).name in ["left0", "right0"] for geom in self.contact_floor_balls])
            # print("torso_touch_floor: ", torso_touch_floor)

            # Reward weights
            rp = [0.8,   # Forward velocity
                  0.2,   # Z angular velocity
                  -0.1,  # Action rate
                  0.,    # Fly reward
                  -0.02, # Num balls fall
                  -0.01, # Dof vel penalty
                  -0.000002, # Dof acc penalty
                  0,      # Jump reward
                  0,     # Upright reward
                  0,      # Height tracking reward
                  0
                  ]
            if self.reward_params is not None:
                rp[:len(self.reward_params)] = self.reward_params

            for i, p in enumerate(rp):
                if isinstance(p, str):
                    # Curriculum learning
                    if self.step_counter < 2e5:
                        rp[i] = float(p.split("-")[0])
                        # print(f"[{self.step_counter}] Curriculum learning: ", rp)
                    else:
                        rp[i] = float(p.split("-")[1])
                        # print(f"[{self.step_counter}] Curriculum learning: ", rp)

            reward_terms = np.array(rp)*np.array([lin_vel_reward, ang_vel_reward, action_rate, fly_reward, num_balls_fall, dof_penalty, dof_acc_penalty, jump_reward, upward_reward, height_track_reward, torso_touch_floor])
            with np.printoptions(precision=2, suppress=True, threshold=10):
                if wandb.run is None or isinstance(wandb.run, wandb.sdk.lib.disabled.RunDisabled) :
                    print("Reawrd: ", reward_terms)
                else:
                    wandb.log({"Reawrd Components": reward_terms.tolist()}, commit=False)
            reward = reward_terms.sum()

            # self.step_counter += 1


        elif reward_version == "recovery_auto2": #!

            weights = {}
            weights["action_rate"] = self.reward_params[0] if self.reward_params is not None else 0

            dof_error = np.sum(np.square(normalize_angle(self.accurate_dof_pos) - normalize_angle(np.array(self.default_dof_pos))))
            tracking_sigma = 10
            # dof_reward = np.exp(-dof_error/tracking_sigma)
            dof_reward = isaac_reward(normalize_angle(np.array(self.default_dof_pos)), normalize_angle(self.accurate_dof_pos), tracking_sigma)


            accurate_projected_gravity = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            upward_reward = np.dot(self.projected_upward_vec, -accurate_projected_gravity)


            # Action rate
            action_rate = np.sum(np.square(self.last_last_action - self.last_action)) /self.num_act
            self.last_last_action = self.last_action.copy()

            reward = dof_reward*upward_reward + weights["action_rate"]*action_rate

            print("self.dof_pos: ", self.dof_pos, "\nself.default_dof_pos: ", self.default_dof_pos, "\ndof_error: ", dof_error)
            print("dof_reward: ", dof_reward, " upward_reward: ", upward_reward)
            print("action_rate: ", weights["action_rate"]*action_rate)


        elif reward_version == "cheat_plateau_jing":
            # Used for simulation; advoid using noisy data for reward
            ang_vel = self.accurate_ang_vel_body
            theta = self.theta
            alpha = self.dof_pos[0]
            jing_vec = get_jing_vector(alpha, theta)
            ang_vel_forward = np.dot(jing_vec, ang_vel)

            max_desired_vel = 6 if self.predefined_commands is None else self.predefined_commands[0]
            if max_desired_vel > 12 and self.step_counter < 2e5:
                max_desired_vel = 12
            # self.step_counter += 1
            lin_vel_reward = plateau(ang_vel_forward, max_desired_vel)

            max_desired_ang_vel = 0 if self.predefined_commands is None else self.predefined_commands[1]
            global_zvec = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            ang_vel_spin = np.dot(global_zvec, ang_vel)
            if max_desired_ang_vel > 0:
                ang_vel_reward = plateau(ang_vel_spin, max_desired_ang_vel)
            elif max_desired_ang_vel < 0:
                ang_vel_reward = plateau(-ang_vel_spin, -max_desired_ang_vel)
            else:
                # encourage the robot to go straight
                ang_vel_reward = -np.square(ang_vel_spin)


            action_rate = np.sum(np.square(self.last_last_action - self.last_action)) /self.num_act
            self.last_last_action = self.last_action.copy()

            rp = [1, 0, -0.1]
            if self.reward_params is not None:
                rp[:len(self.reward_params)] = self.reward_params
            reward_terms = np.array(rp)*np.array([lin_vel_reward, ang_vel_reward, action_rate])
            # with np.printoptions(precision=2, suppress=True, threshold=10):
            #     # print(f"[{self.state_step_counter}] [Return: {self.jump_timer}]  Reawrd: {reward_terms}")
            #     print("Reawrd: ", reward_terms)
            reward = reward_terms.sum()

            self.jump_timer += reward
            self.state_step_counter += 1


        elif reward_version == "jump_timer":

            jump_time = self.reward_params[0]

            if not hasattr(self, "jump_timer"):
                self.jump_timer = 0

            print("jump_time: ", jump_time, " jump_timer: ", self.jump_timer)

            jump_sig = self.commands[0]

            if jump_sig:
                self.jump_timer += 1
                if self.jump_timer > jump_time:
                    self.commands[0] = 0
                    self.jump_timer = 0
            reward = 0


        elif reward_version == "cheat_tripod_jump": #!

            jump_sig = self.commands[0]
            flying = len(self.contact_floor_geoms) == 0

            stationary_height = self.predefined_commands[0]
            jumping_height = self.predefined_commands[1]
            spinning_speed = self.predefined_commands[2]

            desired_height = jumping_height if jump_sig else stationary_height
            height = self.accurate_pos_world[2]


            dof_error = np.sum(np.square(normalize_angle(self.accurate_dof_pos) - normalize_angle(np.array(self.default_dof_pos))))
            tracking_sigma = 10
            dof_reward = isaac_reward(normalize_angle(np.array(self.default_dof_pos)), normalize_angle(self.accurate_dof_pos), tracking_sigma)
            accurate_projected_gravity = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            upward_reward = np.dot(self.projected_upward_vec, -accurate_projected_gravity)

            jump_bonus = 0

            if not jump_sig:
                pos_reward = dof_reward*upward_reward
                height_track_reward = 0
            else:
                pos_reward = 0
                height_track_reward = plateau(height, desired_height)
                print("TIMER: ", self.jump_timer)
                self.jump_timer += 1

                if height > desired_height and flying:
                    height_track_reward = 0
                    jump_bonus = 1
                    self.commands[0] = 0

                    self.jump_timer = 0


            # Control the tuning
            spin = np.dot(-accurate_projected_gravity, self.accurate_ang_vel_body)
            if jump_sig:
                spin_reward = plateau(spin, spinning_speed)
                print("JUMPING SPIN: ", spin)
            else:
                spin_reward = isaac_reward(0, spin, 0.1)


            # Encourage the robot not falling
            up_dir_dot = np.dot([0, 0, 1], -accurate_projected_gravity)

            spin_bonus = plateau(spin, spinning_speed) if jump_sig else 0


            # reward = pos_reward + height_track_reward + 100*jump_bonus

            print("pos_reward: ", pos_reward, " height_track_reward: ", height_track_reward, " jump_bonus: ", 100*jump_bonus, "jump_sig: ", jump_sig)

            rp = [1, 1, 100, 0, 0, 0] # pos_reward, height_track_reward, jump_bonus, spin_reward, up_dir_dot, spin_bonus
            if self.reward_params is not None:
                rp[:len(self.reward_params)] = self.reward_params
            reward_terms = np.array(rp)*np.array([pos_reward, height_track_reward, jump_bonus, spin_reward, up_dir_dot, spin_bonus])
            with np.printoptions(precision=2, suppress=True, threshold=10):
                print("Reawrd: ", reward_terms)
            reward = reward_terms.sum()


        else:
            print("Reward version not found: ", self.reward_version)
            pdb.set_trace()

        self.step_counter += 1

        return reward, info
    
    def is_done(self):
        if self.done_version is None:
            done = False
        
        elif self.done_version == "ballance":
            if np.dot(np.array([0, np.cos(self.theta), np.sin(self.theta)]), -self.projected_gravity) < 0.01:
                done = True
            else:
                done = False

        elif self.done_version == "ballance_upsidedown":
            up_vec = [0,0,-1]
            accurate_projected_gravity = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            if np.dot(np.array(up_vec), -accurate_projected_gravity) < 0.1:
                done = True
            else:
                done = False

        elif self.done_version == "ballance_up":
            up_vec = [0,0,1]
            accurate_projected_gravity = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            if np.dot(np.array(up_vec), -accurate_projected_gravity) < 0.1:
                done = True
            else:
                done = False

        elif self.done_version == "ballance_auto":
            assert self.projected_upward_vec is not None, "projected_upward_vec is None"
            up_vec = self.projected_upward_vec
            accurate_projected_gravity = quat_rotate_inverse(self.accurate_quat, self.gravity_vec)
            if np.dot(np.array(up_vec), -accurate_projected_gravity) < 0.1:
                done = True
            else:
                done = False

        elif self.done_version == "torso_fall":
            # This is specifically for the quadrupedX4air1s

            if 1 in self.contact_floor_balls or 20 in self.contact_floor_balls:
                done = True
            else:
                done = False

        elif self.done_version == "three_feet":
            threeleg = all(x in self.contact_floor_geoms for x in [5,6,7]) or all(x in self.contact_floor_geoms for x in [9,10,11])
            not_moving = np.linalg.norm(self.accurate_vel_world) < 0.1
            done = threeleg and not_moving

        elif self.done_version == "ball_fall":
            done = bool(self.contact_floor_balls)

        return done
    
    def is_upsidedown(self):
        # assert self.projected_upward_vec is not None, "projected_upward_vec is None"
        if self.projected_upward_vec is not None:
            up_vec = self.projected_upward_vec
            accurate_projected_gravity = quat_rotate_inverse(self.quat, self.gravity_vec)
            if np.dot(np.array(up_vec), -accurate_projected_gravity) < 0.1:
                # print("Upsidedown! ", np.dot(np.array(up_vec), -accurate_projected_gravity))
                return True
                
            else:
                # print("Not upsidedown! ", np.dot(np.array(up_vec), -accurate_projected_gravity))
                return False
        else:
            return None
        
    

    def get_custom_commands(self, command_type):

        info = {}
        if command_type == "onehot_dirichlet":
            print("self.step_counter: ", self.step_counter)
            if self.step_counter < self.train_total_steps/2:
                commands = np.zeros(3)
                commands[np.random.randint(3)] = 1
            else:
                commands = np.random.dirichlet(np.ones(3))

        elif command_type == "onehot":
            commands = np.zeros(3)
            commands[np.random.randint(3)] = 1


        return commands, info


    def reset_reward_state(self):
        # Reset state-based reward variables
        self.ang_vel_history = []
        self.vel_history = []
        self.pos_history = []
        self.last_dof_vel = np.zeros(self.num_joint)

        self.contact_counter = {}
        self.fly_counter = 0
        self.jump_timer = 0

        self.vel_filter = AverageFilter(int(0.5/self.dt))

    def reset_state_state(self):
        # Reset state-based state variables
        self.state_step_counter = 0