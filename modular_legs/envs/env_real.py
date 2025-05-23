import copy
import datetime
import json
import os
import pdb
import time

import numpy as np
from omegaconf import OmegaConf
from modular_legs.envs.base import RealSim
from modular_legs.embodied.interface import Interface, sanitize_dict
from modular_legs.utils.others import is_number

class Real(RealSim):
    def __init__(self, cfg):

        self.interface = Interface(cfg)
        super(Real, self).__init__(cfg)

        self.log_dir = self.cfg.logging.robot_data_dir

        # Logging data
        if self.log_dir is not None:
            self.log_file = open(os.path.join(self.log_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"), "w")



    def update_config(self, cfg: OmegaConf):
        super().update_config(cfg)
        self.interface.update_config(cfg)

    def _log_data(self):
        # if self.cfg.logging.log_raw_data:
        #     self.interface.log_raw_data()
        if self.log_dir is not None:
            self.log_file.write(json.dumps(sanitize_dict(copy.deepcopy(self.observable_data))) + "\n")
            self.log_file.flush()

    def _is_truncated(self):
        return not self.interface.ready()
    

    def _get_observable_data(self):
        self.interface.receive_module_data()
        data_dict = self.interface.get_observable_data()
        return data_dict


    def _wait_until_motor_on(self):
        t = 0
        while not self.interface.ready():
            # print("Waiting for the robot...")
            self.interface.receive_module_data()
            if self.control_mode in ["velocity", "advanced"]:
                kps = self.kps*0
            else:
                kps = self.kps
            if self.num_act != len(self.interface.module_ids):
                self.interface.send_action(np.zeros(len(self.interface.module_ids)), kps=np.zeros(len(self.interface.module_ids)), kds=np.zeros(len(self.interface.module_ids)))
            else:
                self.interface.send_action(self.last_action, kps=kps, kds=self.kds)
            if t == 0:
                print("Waiting for the robot...")
            self._check_input()
            time.sleep(0.02)
            t +=1


    def _perform_action(self, pos, vel=None, kps=None, kds=None):

        self.kps = kps if kps is not None else self.kps # Overwrite the default kps if provided
        self.kds = kds if kds is not None else self.kds # Overwrite the default kds if provided

        if vel is None:
            vel = np.zeros_like(pos)
            
        self.interface.send_action(pos, 
                                   vel, 
                                   self.kps,
                                   self.kds)
        self.interface.compute_time = time.time() - self.t0
        while time.time() - self.t0 < self.interface.dt:
            pass

        self.interface.send_dt = time.time() - self.t0
        print("DT: ", self.interface.send_dt)
        self.t0 = time.time()

        return {}

    def _check_input(self):
        super()._check_input()
        if self.input_key == "e":
            self.interface._enable_motor()
        if self.input_key == "d":
            self.interface._disable_motor()
        if self.input_key == "r":
            # Restart the ith module
            self.interface._restart_motor("auto")
            self.interface.last_motor_com_time = time.time()
        if self.input_key == "f":
            # Try to fix the motor
            self.interface._fix_motor("auto")
            self.interface.last_motor_com_time = time.time()

        if time.time() - self.interface.last_motor_com_time > 0.5:
            # Reset motor commands
            self.interface._reset_motor_commands()