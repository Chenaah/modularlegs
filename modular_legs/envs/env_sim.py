from collections import defaultdict
import copy
from datetime import datetime
import os
import pdb
# from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
import mujoco
from omegaconf import OmegaConf
from modular_legs.envs.base import RealSim
from modular_legs import LEG_ROOT_DIR
# from modular_legs.sim.gen_jxms import UniJx
from modular_legs.sim.robot_designer import DEFAULT_ROBOT_CONFIG
# from modular_legs.sim.robot_metadesigner import MetaDesigner
from modular_legs.sim.terrain import Terrain
from modular_legs.utils.kbhit import KBHit
from modular_legs.utils.math import AverageFilter, construct_quaternion, euler_to_quaternion, quat_rotate, quat_rotate_inverse, quaternion_multiply2, quaternion_to_euler, rotate_vector2D, wxyz_to_xyzw
from modular_legs.utils.model import XMLCompiler, quaternion_from_vectors, compile_xml
from modular_legs.utils.others import is_list_like, is_number, string_to_list
from modular_legs.envs.gym.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class ZeroSim(RealSim, MujocoEnv):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, cfg):
        super(ZeroSim, self).__init__(cfg)

    def _load_asset(self, asset_file_name):
        xml_file = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", asset_file_name)
        self.xml_compiler = XMLCompiler(xml_file)
        self.xml_compiler.torque_control()
        self.xml_compiler.update_timestep(self.mj_dt)
        if self.pyramidal_cone:
            self.xml_compiler.pyramidal_cone()
        if self.randomize_mass:
            self.mass_range = self.xml_compiler.get_mass_range(self.random_mass_percentage)
        self.xml_string = self.xml_compiler.get_string()


    def update_config(self, cfg: OmegaConf):

        self.num_act = cfg.agent.num_act
        self.num_envs = cfg.agent.num_envs
        self.randomize_ini_vel = cfg.sim.randomize_ini_vel
        self.randomize_orientation = cfg.sim.randomize_orientation
        self.fully_randomize_orientation = cfg.sim.fully_randomize_orientation
        self.randomize_init_joint_pos = cfg.sim.randomize_init_joint_pos
        self.noisy_init = cfg.sim.noisy_init
        self.tn_constraint = cfg.sim.tn_constraint
        self.kp, self.kd = cfg.robot.kp, cfg.robot.kd
        self.broken_motors = cfg.sim.broken_motors
        self.random_latency_scheme = cfg.sim.random_latency_scheme
        self.latency_scheme = cfg.sim.latency_scheme
        self.randomize_mass = cfg.sim.randomize_mass
        self.random_mass_percentage = cfg.sim.random_mass_percentage
        self.mass_offset = cfg.sim.mass_offset
        self.randomize_friction = cfg.sim.randomize_friction
        self.random_friction_range = cfg.sim.random_friction_range
        self.randomize_rolling_friction = cfg.sim.randomize_rolling_friction
        self.random_rolling_friction_range = cfg.sim.random_rolling_friction_range
        self.noisy_actions = cfg.sim.noisy_actions
        self.action_noise_std = cfg.sim.action_noise_std
        self.noisy_observations = cfg.sim.noisy_observations
        self.obs_noise_std = cfg.sim.obs_noise_std
        self.pyramidal_cone = cfg.sim.pyramidal_cone
        self.randomize_assemble_error = cfg.sim.randomize_assemble_error
        if self.randomize_assemble_error:
            # The robot model will be re-decoded from design pipeline in the reset function
            assert cfg.trainer.evolution.design_pipeline is not None, "design_pipeline should be provided for randomize_assemble_error"
            self.design_pipeline = string_to_list(cfg.trainer.evolution.design_pipeline)
            self.socks = cfg.sim.socks
        self.terrain = cfg.sim.terrain
        self.terrain_params = cfg.sim.terrain_params
        self.reset_terrain = cfg.sim.reset_terrain
        self.reset_terrain_type = cfg.sim.reset_terrain_type
        self.reset_terrain_params = cfg.sim.reset_terrain_params
        self.randomize_damping = cfg.sim.randomize_damping
        self.random_damping_range = cfg.sim.random_damping_range
        self.random_armature_range = cfg.sim.random_armature_range
        self.random_external_torque = cfg.sim.random_external_torque
        self.random_external_torque_range = cfg.sim.random_external_torque_range
        self.random_external_torque_bodies = cfg.sim.random_external_torque_bodies
        self.random_external_force = cfg.sim.random_external_force
        self.random_external_force_ranges = cfg.sim.random_external_force_ranges
        self.random_external_force_bodies = cfg.sim.random_external_force_bodies
        self.random_external_force_positions = cfg.sim.random_external_force_positions
        self.random_external_force_directions = cfg.sim.random_external_force_directions
        self.random_external_force_durations = cfg.sim.random_external_force_durations
        self.random_external_force_interval = cfg.sim.random_external_force_interval
        self.randomize_dof_pos = cfg.sim.randomize_dof_pos
        self.random_dof_pos_range = cfg.sim.random_dof_pos_range
        self.randomize_pd_controller = cfg.sim.randomize_pd_controller
        self.random_kp_range = cfg.sim.random_kp_range
        self.random_kd_range = cfg.sim.random_kd_range
        self.add_scaffold_walls = cfg.sim.add_scaffold_walls

        self.asset_file_names = cfg.sim.asset_file
        self.randomize_asset = is_list_like(self.asset_file_names)
        if self.randomize_asset:
            asset_file_name = np.random.choice(self.asset_file_names)
        else:
            asset_file_name = self.asset_file_names
        
        self.mj_dt = cfg.sim.mj_dt
        assert cfg.robot.dt%self.mj_dt < 1e-9, f"dt ({cfg.robot.dt}) should be a multiple of mj_dt {self.mj_dt}"
        self._load_asset(asset_file_name)
        self.terrain_resetter = Terrain()
        self.frame_skip = int(cfg.robot.dt / self.mj_dt) # the timestep in xml will be overwritten by self.mj_dt
        self.forward_vec = np.array(cfg.agent.forward_vec) if cfg.agent.forward_vec is not None else None
        self.kps, self.kds = np.array([self.kp]*(int(self.num_act*self.num_envs)), dtype=np.float32), np.array([self.kd]*(int(self.num_act*self.num_envs)), dtype=np.float32)

        self.render_size = cfg.sim.render_size
        
        MujocoEnv.__init__(
            self,
            self.xml_string,
            self.frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=self.render_size[0],
            height=self.render_size[1],
            # **kwargs,
        )

        super().update_config(cfg)

        # Simulation configuration; config should be more orgainzed
        self.init_pos = cfg.sim.init_pos
        self.init_joint_pos = cfg.sim.init_joint_pos
        lleg_vec = np.array([0, np.cos(self.theta), np.sin(self.theta)])
        if is_list_like(cfg.sim.init_quat):
            if len(cfg.sim.init_quat) == 4: # "init_quat should be a list of 4 elements"
                self.init_quat = cfg.sim.init_quat
            elif len(cfg.sim.init_quat) == 3:
                print("init_quat is set to align the robot with vector ", cfg.sim.init_quat)
                self.init_quat = quaternion_from_vectors(lleg_vec,  np.array(cfg.sim.init_quat))
        elif cfg.sim.init_quat == "x":
            # print("init_quat is set to facing x")
            self.init_quat = quaternion_from_vectors(lleg_vec,  np.array([1, 0, 0]))
        elif cfg.sim.init_quat == "y":
            # print("init_quat is set to facing y")
            self.init_quat = quaternion_from_vectors(lleg_vec,  np.array([0, 1, 0]))
        else:
            raise ValueError("init_quat should be a list of 4 elements or 'x'")
        
        

        self.num_joint = self.model.nu
        self.jointed_module_ids = sorted([int(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j).replace("joint", "") )for j in self.model.actuator_trnid[:, 0]])
        if self.action_remap_type is None:
            assert self.num_act*self.num_envs == self.num_joint, "num_act should be the same as the number of actuators in the model"
        else:
            print("action will be remapped!")
        self.joint_idx = [self.model.joint(f'joint{i}').id for i in self.jointed_module_ids]
        self.joint_geom_idx = [self.model.geom(f'left{i}').id for i in self.jointed_module_ids] + [self.model.geom(f'right{i}').id for i in self.jointed_module_ids]
        unique_joint_geom_idx = [self.model.geom(f'left{i}').id for i in self.jointed_module_ids]
        self.joint_body_idx = [self.model.geom(i).bodyid.item() for i in unique_joint_geom_idx]

        self.render_on = cfg.sim.render
        if self.render_on:
            self.render_mode = "human"
        else:
            self.render_mode = "rgb_array"
        self.render_on_bg = False

        # Record the last two actions for latency simulation
        # Note that this is not the same as the last_action in RealSim
        self.last_pos_sim = np.ones(self.num_joint) * np.array(self.default_dof_pos)
        self.last_last_pos_sim = np.ones(self.num_joint) * np.array(self.default_dof_pos)
        self.last_last_vel_sim = np.zeros(self.num_joint)
        self.last_vel_sim = np.zeros(self.num_joint)

        self.last_com_pos = np.zeros(3)
        self.episode_counter = 0
        self.sensors = defaultdict(list)

    def _reset_external_forces(self):
        self.data.qfrc_applied = np.zeros(self.model.nv)
        self.external_force_counter = {i: 0 for i in range(len(self.random_external_force_bodies))}

    def _apply_force(self, force, body, position, direction):
        # force/direction are in global frame, position is in local frame
        body_index = self.model.body(body).id
        rotation_matrix = self.data.xmat[body_index].reshape(3, 3)  # 3x3 rotation matrix
        position_global = self.data.xpos[body_index]  # Body's position in global frame
        point_local = position

        # Transform to global frame
        force_global = np.array(direction)*force
        point_global = position_global + np.dot(rotation_matrix, point_local)

        torque = np.array([0.0, 0.0, 0.0])
        qfrc_result = np.array([0.0] * len(self.data.qvel))
        mujoco.mj_applyFT(self.model, self.data, force_global, torque, point_global, body_index, qfrc_result)
        self.data.qfrc_applied = qfrc_result


    def _perform_action(self, pos, vel=None, kps=None, kds=None):
        
        # Assume the actuator has been converted to torque control

        if self.random_external_force:
            print(f"Random external force is on: {self.data.qfrc_applied}")
            # TODO: multiple external forces not implemented yet
            assert len(self.random_external_force_ranges) == 1, "Only one external force is supported"
            if self.step_count % self.random_external_force_interval == 0:
                print("Applying external force!")
                for i, (force_range, body, position, direction) in enumerate(zip(self.random_external_force_ranges, self.random_external_force_bodies, self.random_external_force_positions, self.random_external_force_directions)):
                    external_force = np.random.uniform(*force_range)
                    self._apply_force(external_force, body, position, direction)
                

            for i, duration in enumerate(self.random_external_force_durations):
                self.external_force_counter[i] += 1
                print(f"Duration: {duration}, Counter: {self.external_force_counter[i]}")
                if self.external_force_counter[i] >= duration:
                    self._reset_external_forces()
                    print("Resetting external force!")
                    self.external_force_counter[i] = 0

        assert self.frame_skip%2 == 0, "frame_skip should be an even number"
        self.kps = kps if kps is not None else self.kps # Overwrite the default kps if provided
        self.kds = kds if kds is not None else self.kds # Overwrite the default kds if provided
        
        vel = np.zeros_like(pos) if vel is None else vel
        if self.noisy_actions:
            pos = pos + self.np_random.normal(0, self.action_noise_std, size=pos.shape)
            vel = vel + self.np_random.normal(0, self.action_noise_std, size=vel.shape)
        # print(f"pos: {pos}, vel: {vel}, self.kp: {self.kp}, self.kd: {self.kd}")

        xposbefore = self.data.qpos.flat[0]
        yposbefore = self.data.qpos.flat[1]
        coordinates_general = np.array([self.data.qpos.flat[i:i+2] for i in self.free_joint_addr]).reshape(-1, 2)

        if self.latency_scheme == -1:
            self._pd_control(pos, self.frame_skip, vel_desired=vel)
        elif self.latency_scheme == 0:
            self._pd_control(self.last_pos_sim, self.frame_skip/2, vel_desired=self.last_vel_sim)
            self._pd_control(pos, self.frame_skip/2, vel_desired=vel)
        elif self.latency_scheme == 1:
            self._pd_control(self.last_last_pos_sim, self.frame_skip/2, vel_desired=self.last_last_vel_sim)
            self._pd_control(self.last_pos_sim, self.frame_skip/2, vel_desired=self.last_vel_sim)
        
        self.last_last_pos_sim = self.last_pos_sim.copy()
        self.last_pos_sim = pos.copy()
        self.last_last_vel_sim = self.last_vel_sim.copy()
        self.last_vel_sim = vel.copy()

        xposafter = self.data.qpos.flat[0]
        yposafter = self.data.qpos.flat[1]
        next_coordinates_general = np.array([self.data.qpos.flat[i:i+2] for i in self.free_joint_addr]).reshape(-1, 2)

        info = dict(
            coordinates=np.array([xposbefore, yposbefore]),
            next_coordinates=np.array([xposafter, yposafter]),
            coordinates_general=coordinates_general,
            next_coordinates_general=next_coordinates_general,
        )
        if self.render_on_bg and self.render_mode == "rgb_array": # TODO
            info['render'] = self.render().transpose(2, 0, 1)

        return info


    def _pd_control(self, pos_desired, frame_skip, vel_desired=0):
        dof_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_idx]]
        dof_vel = self.data.qvel[self.model.jnt_dofadr[self.joint_idx]]

        torques = self.kps*(pos_desired - dof_pos) + self.kds*(vel_desired - dof_vel)
        if self.tn_constraint:
            torque_limits = np.zeros_like(dof_pos)
            torque_limits[np.abs(dof_vel) < 11.5] = 12
            torque_limits[np.abs(dof_vel) >= 11.5] = np.clip(-0.656*np.abs(dof_vel) + 19.541, a_min=0, a_max=None)[np.abs(dof_vel) >= 11.5]
            torques = np.clip(torques, -torque_limits, torque_limits)
        if self.broken_motors is not None:
            torques[self.broken_motors] = 0
        self.do_simulation(torques, int(frame_skip))
        # print("ERROR: ", pos_desired - dof_pos,  "Torques: ", torques)
        # print("Torques: ", torques)

        # max_velocity = 30.997048
        # max_velocity = 21
        # for i in self.jointed_module_ids:
        #     if self.data.qvel[6+i] > max_velocity:
        #         self.data.qvel[6+i] = max_velocity
        #     elif self.data.qvel[6+i] < -max_velocity:
        #         self.data.qvel[6+i] = -max_velocity


    def _get_observable_data(self):
        # TODO: add noise here
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        self.pos_world = qpos[:3]
        quat = qpos[3:7]
        quat = wxyz_to_xyzw(quat)
        
        dof_pos = qpos[self.model.jnt_qposadr[self.joint_idx]]
        vel_world = qvel[:3]
        vel_body = quat_rotate_inverse(quat, vel_world)
        ang_vel_body = qvel[3:6]
        ang_vel_world = quat_rotate(quat, ang_vel_body)
        dof_vel = qvel[self.model.jnt_dofadr[self.joint_idx]]
        # dof_torques = [self.data.joint(f"joint{i}").qfrc_constraint for i in self.jointed_module_ids]

        try:
            self.sensors["quat"] = np.array([self.data.sensordata[self.model.sensor(f"imu_quat{i}").adr[0]:self.model.sensor(f"imu_quat{i}").adr[0]+4] for i in self.jointed_module_ids])
            self.sensors["gyro"] = np.array([self.data.sensordata[self.model.sensor(f"imu_gyro{i}").adr[0]:self.model.sensor(f"imu_gyro{i}").adr[0]+3] for i in self.jointed_module_ids])
            self.sensors["vel"] = np.array([self.data.sensordata[self.model.sensor(f"imu_vel{i}").adr[0]:self.model.sensor(f"imu_vel{i}").adr[0]+3] for i in self.jointed_module_ids])
            self.sensors["globvel"] = np.array([self.data.sensordata[self.model.sensor(f"imu_globvel{i}").adr[0]:self.model.sensor(f"imu_globvel{i}").adr[0]+3] for i in self.jointed_module_ids])
            self.sensors["back_quat"] = np.array([self.data.sensordata[self.model.sensor(f"back_imu_quat{i}").adr[0]:self.model.sensor(f"back_imu_quat{i}").adr[0]+4] for i in self.jointed_module_ids])
            self.sensors["back_gyro"] = np.array([self.data.sensordata[self.model.sensor(f"back_imu_gyro{i}").adr[0]:self.model.sensor(f"back_imu_gyro{i}").adr[0]+3] for i in self.jointed_module_ids])
            self.sensors["back_vel"] = np.array([self.data.sensordata[self.model.sensor(f"back_imu_vel{i}").adr[0]:self.model.sensor(f"back_imu_vel{i}").adr[0]+3] for i in self.jointed_module_ids])
        except KeyError:
            pass

        try:
            self.sensors["acc"] = np.array([self.data.sensordata[self.model.sensor(f"imu_acc{i}").adr[0]:self.model.sensor(f"imu_acc{i}").adr[0]+3] for i in self.jointed_module_ids])
        except KeyError:
            pass
        data_dict = {}
        data_dict['pos_world'] = self.pos_world
        data_dict['quat'] = quat
        data_dict['dof_pos'] = dof_pos
        data_dict['dof_vel'] = dof_vel
        data_dict['vel_world'] = vel_world
        data_dict['vel_body'] = vel_body
        data_dict['ang_vel_body'] = ang_vel_body
        data_dict['ang_vel_world'] = ang_vel_world
        # For compatibility with some old obs
        data_dict['qpos'] = self.data.qpos.flatten()
        data_dict['qvel'] = self.data.qvel.flatten()
        # New data from sensros
        data_dict['quats'] = np.array([wxyz_to_xyzw(q) for q in self.sensors["quat"]])
        data_dict['gyros'] = self.sensors["gyro"]
        data_dict['accs'] = self.sensors["acc"]

        extented_data_dict = data_dict.copy()
        
        for key, value in data_dict.items():
            extented_data_dict[f"accurate_{key}"] = copy.deepcopy(value)
            if self.noisy_observations:
                if isinstance(value, list):
                    pdb.set_trace()
                extented_data_dict[key] = value + self.np_random.normal(0, self.obs_noise_std, size=value.shape)

        # Pure simulated data, no noise
        # For legged locomotion reward
        extented_data_dict['contact_geoms'] = [c.geom for c in self.data.contact]
        floor_contacts = [c.geom for c in self.data.contact if 0 in c.geom]
        # extented_data_dict['num_jointfloor_contact'] = [joint in contact for joint in self.joint_geom_idx for contact in floor_contacts].count(True)
        extented_data_dict['num_jointfloor_contact'] = [((contact[0] in self.joint_geom_idx) or (contact[1] in self.joint_geom_idx)) for contact in floor_contacts].count(True)
        extented_data_dict['contact_floor_geoms'] = list(set([geom for pair in floor_contacts for geom in pair if geom != 0]))
        extented_data_dict['contact_floor_socks'] = list(set([geom for pair in floor_contacts for geom in pair if self.model.geom(geom).name.startswith("sock")]))
        extented_data_dict['contact_floor_balls'] = list(set([geom for pair in floor_contacts for geom in pair if self.model.geom(geom).name.startswith("left") or self.model.geom(geom).name.startswith("right")]))
        extented_data_dict['mj_data'] = self.data
        extented_data_dict['mj_model'] = self.model
        extented_data_dict['adjusted_forward_vec'] = self.adjusted_forward_vec
        extented_data_dict['vels'] = self.sensors["vel"]
        # Back sensors only exist in simulation
        extented_data_dict['back_vels'] = self.sensors["back_vel"]
        extented_data_dict['back_quats'] = np.array([wxyz_to_xyzw(q) for q in self.sensors["back_quat"]])
        extented_data_dict['back_gyros'] = self.sensors["back_gyro"]

        com_pos = np.mean(self.data.xpos[self.joint_body_idx], axis=0)
        com_vel_world = (com_pos - self.last_com_pos) / self.dt
        self.last_com_pos = com_pos.copy()
        # print("com_vel_world: ", com_vel_world)
        extented_data_dict['com_vel_world'] = com_vel_world
        
        return extented_data_dict


    def _reset_robot(self):
        self.reset_model()

        self.last_pos_sim = np.ones(self.num_joint) * np.array(self.default_dof_pos)
        self.last_last_pos_sim = np.ones(self.num_joint) * np.array(self.default_dof_pos)
        self.last_vel_sim = np.zeros(self.num_joint)
        self.last_last_vel_sim = np.zeros(self.num_joint)
        self.last_com_pos = np.zeros(3)
        self.render_lookat_filter = AverageFilter(10)


    def _pre_reset(self):
        pass

    def _post_reset(self):
        pass

    def _log_data(self):
        if self.render_on:
            viewer = self.mujoco_renderer._get_viewer("human")
            viewer.cam.lookat = self.render_lookat_filter(self.pos_world)
            self.render()

        

    def reload_model(self, xml_string):
        if self.mujoco_renderer.viewer is not None:
            self.close()
        MujocoEnv.__init__(
            self,
            xml_string,
            self.frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode="human" if self.render_on else "rgb_array",
            width=self.render_size[0],
            height=self.render_size[1],
            # **kwargs,
        )

    def _need_reload_model(self):
        return self.randomize_mass or self.randomize_assemble_error or self.randomize_damping or self.add_scaffold_walls or self.reset_terrain or self.randomize_asset
    
    def reset_model(self):

        self._pre_reset()

        # Deprecated
        # if self.cfg.sim.auto_generate_assets:
        #     d_name = datetime.datetime.now().strftime("%m%d%H%M%S")
        #     n_modules = self.cfg.robot.num_modules
        #     asset_file = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "JXM", "generated", f"s{n_modules}_{d_name}.xml")
        #     # UniJx().genjxms(output_file=asset_file, n_modules=n_modules)
        #     self.reload_model(asset_file) # TODO: not converted to xml string yet

        
        if self.randomize_orientation:
            rotate_angle = np.random.uniform(0,2*np.pi)
            rand_angle = construct_quaternion([0,0,1], rotate_angle)
            final_init_quat = quaternion_multiply2(self.init_quat, rand_angle)
            if self.forward_vec is not None:
                self.adjusted_forward_vec = rotate_vector2D(self.forward_vec[:2], -rotate_angle)
        else:
            final_init_quat = self.init_quat
            if self.forward_vec is not None:
                self.adjusted_forward_vec = self.forward_vec

        if self.fully_randomize_orientation:
            # rand_quat = np.random.uniform(-1,1,(4,))
            rand_quat = np.random.normal(0,1,(4,))
            final_init_quat = rand_quat / np.linalg.norm(rand_quat)


        if self._need_reload_model():
            # Reload the model
            # Note that order of the following randomizations matters

            if self.randomize_assemble_error:
                # TODO: Update the code for new robot designer
                robot_cfg = copy.deepcopy(DEFAULT_ROBOT_CONFIG)
                robot_cfg["a"] *= np.random.uniform(0.8, 1.2) # stick center to the dock center on the side
                robot_cfg["delta_l"] += np.random.uniform(-0.01, 0.01)
                robot_cfg["stick_ball_l"] += np.random.uniform(-0.01, 0.01)
                robot_designer = MetaDesigner(init_pipeline=self.design_pipeline, robot_cfg=robot_cfg)
                if self.terrain is not None:
                    robot_designer.set_terrain(self.terrain)
                if self.socks is not None:
                    robot_designer.wear_socks(self.socks, color=(1,1,1))
                xml_string = robot_designer.get_xml()
                self.xml_compiler = XMLCompiler(xml_string)
                self.xml_compiler.torque_control()
                self.xml_compiler.update_timestep(self.mj_dt)
                if self.pyramidal_cone:
                    self.xml_compiler.pyramidal_cone()

            if self.randomize_asset:
                asset_file_name = np.random.choice(self.asset_file_names)
                self._load_asset(asset_file_name)

            if self.randomize_mass:
                mass_dict = {key: np.random.uniform(value[0], value[1])+self.mass_offset for key, value in self.mass_range.items()}
                self.xml_compiler.update_mass(mass_dict)

            if self.randomize_damping:
                self.xml_compiler.update_damping(armature=np.random.uniform(*self.random_armature_range), 
                                                 damping=np.random.uniform(*self.random_damping_range))
                
            if self.add_scaffold_walls:
                self.xml_compiler.remove_walls()
                e = quaternion_to_euler(final_init_quat)
                self.xml_compiler.add_walls(transparent=False, angle=e[0]*180/np.pi-90)

            if self.reset_terrain:
                assert self.reset_terrain_type is not None, "reset_terrain_type should be provided for reset_terrain"

                if self.reset_terrain_type != "random":
                    reset_terrain_type = self.reset_terrain_type
                    reset_terrain_param = self.reset_terrain_params[0] if self.reset_terrain_params is not None else None
                else:
                    type_idx = np.random.randint(0, len(self.reset_terrain_params))
                    reset_terrain_type = ["set_slope", "set_discrete", "set_gaps", "set_grid"][type_idx]
                    reset_terrain_param = self.reset_terrain_params[type_idx]

                self.terrain_resetter.reset_terrain(self.xml_compiler, reset_terrain_type, reset_terrain_param)

            


            self.reload_model(self.xml_compiler.get_string()) # Note that this will overwrite self.init_qpos
            # print("Stick mass:  ", self.model.body('passive0').mass)

            if self.reset_terrain:
                self.terrain_resetter.update_hfield(self.model)

            self.episode_counter += 1

        self._reset_external_forces()
        
        if self.random_external_torque:
            raise NotImplementedError("[DEBUG] random_external_torque is not implemented yet")
            for body in self.random_external_torque_bodies:
                external_torque = np.random.uniform(*self.random_external_torque_range)
                rand_theta = np.random.uniform(0, 2 * np.pi)
                self.data.xfrc_applied[body, 3:6] = [external_torque*np.cos(rand_theta), external_torque*np.sin(rand_theta), 0]
        
        if self.randomize_pd_controller:
            self.kp = np.random.uniform(*self.random_kp_range)
            self.kd = np.random.uniform(*self.random_kd_range)

        final_init_quat_list = [final_init_quat] + [[1,0,0,0]]*10 # TODO

        if not is_list_like(self.init_pos[0]):
            assert len(self.init_pos) == 3, "init_pos should be a list of 3 elements"
            self.init_qpos[:3] = self.init_pos
        else:
            assert len(self.init_pos[0]) == 3, "init_pos should be a list of 3 elements"
            init_pos_list = copy.deepcopy(self.init_pos)
            for i in range(self.model.njnt):
                jnt_type = self.model.jnt_type[i]
                qpos_adr = self.model.jnt_qposadr[i]
                if jnt_type == 0:  # mjJNT_FREE
                    self.init_qpos[qpos_adr : qpos_adr + 3] = init_pos_list.pop(0)
                    self.init_qpos[qpos_adr + 3 : qpos_adr + 7] = final_init_quat_list.pop(0)

        self.free_joint_addr = []
        for i in range(self.model.njnt):
                jnt_type = self.model.jnt_type[i]
                qpos_adr = self.model.jnt_qposadr[i]
                if jnt_type == 0:  # mjJNT_FREE
                    self.free_joint_addr.append(qpos_adr)

        self.init_qpos[3:7] = final_init_quat

        default_dof_pos = np.array(self.default_dof_pos) if is_list_like(self.default_dof_pos) else np.array([self.default_dof_pos]*self.model.nu)
        if not self.randomize_init_joint_pos:
            init_joint_pos = np.array(self.init_joint_pos) if is_list_like(self.init_joint_pos) else np.array([self.init_joint_pos]*self.model.nu)
        else:
            init_joint_pos = np.random.uniform(-self.clip_actions, self.clip_actions, self.model.nu)
        self.init_qpos[self.model.jnt_qposadr[self.joint_idx]] = default_dof_pos + init_joint_pos

        if self.randomize_dof_pos:
            self.init_qpos[self.model.jnt_qposadr[self.joint_idx]] = np.random.uniform(*self.random_dof_pos_range, size=self.model.nu)

        if self.noisy_init:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.1, high=0.1
            )
        else:
            qpos = self.init_qpos

        if self.randomize_ini_vel:
            if is_list_like(self.randomize_ini_vel):
                for i, v in enumerate(self.randomize_ini_vel):
                    self.init_qvel[i] = np.random.uniform(-v,v)
            else:
                self.init_qvel[0:6] = np.random.uniform(-1,1,(6,))

        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        # self._perform_action(np.zeros(self.num_act))
        # mujoco.mj_step(self.model, self.data)

        self.reset_pos = copy.copy(self.data.qpos[:2])

        # Resample domain randomization parameters
        if self.random_latency_scheme:
            self.latency_scheme = np.random.randint(0, 2)
        if self.randomize_rolling_friction:
            roll_friction = np.random.uniform(self.random_rolling_friction_range[0], self.random_rolling_friction_range[1], 2)
        if self.randomize_friction:
            if is_number(self.random_friction_range[0]):
                # Only one range is provided
                friction = np.random.uniform(self.random_friction_range[0], self.random_friction_range[1])
                self.model.geom('floor').friction[0] = friction
                self.model.geom('floor').priority[0] = 10
                if self.randomize_rolling_friction:
                    self.model.geom('floor').friction[1:3] = roll_friction
            elif is_list_like(self.random_friction_range[0]):
                # Ranges for balls and socks are provided
                stick_friction = np.random.uniform(self.random_friction_range[0][0], self.random_friction_range[0][1])
                ball_friction = np.random.uniform(self.random_friction_range[1][0], self.random_friction_range[1][1])
                self.model.geom('floor').priority[0] = 1
                for i in self.jointed_module_ids:
                    self.model.geom(f'left{i}').friction[0] = ball_friction
                    self.model.geom(f'right{i}').friction[0] = ball_friction
                    self.model.geom(f'stick{i}').friction[0] = stick_friction
                    self.model.geom(f'left{i}').priority[0] = 2
                    self.model.geom(f'right{i}').priority[0] = 2
                    self.model.geom(f'stick{i}').priority[0] = 2
                    if self.randomize_rolling_friction:
                        self.model.geom(f'left{i}').friction[1:3] = roll_friction
                        self.model.geom(f'right{i}').friction[1:3] = roll_friction
                # print("Ball friction: ", self.model.geom(f'left0').friction)
            else:
                raise ValueError("random_friction_range should be a number or a list of numbers")
            
            # roll_friction = np.random.uniform(0, 0.01, 2)
            # self.model.geom('floor').friction[1:3] = roll_friction
            # print("Floor friction: ", friction)
        

        

        self._post_reset()

    
