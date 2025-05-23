import pdb
import struct
import time

import numpy as np

from modular_legs.embodied.interpreter import interpret_motor_msg


class SentDataStruct:
    def __init__(self, target_pos: float, target_vel: float, kp: float, kd: float, enable_filter: int, switch: int, calibrate: int, restart: int, timestamp: int):
        self.target_pos = target_pos
        self.target_vel = target_vel
        self.kp = kp
        self.kd = kd
        self.enable_filter = enable_filter
        self.switch = switch
        self.calibrate = calibrate
        self.restart = restart
        self.timestamp = timestamp

    def serialize(self):
        return struct.pack('ffffiiiif', self.target_pos, self.target_vel, self.kp, self.kd, self.enable_filter, self.switch, self.calibrate, self.restart, self.timestamp)
    


class DBCommandStruct:
    def __init__(self, switch: int, calibrate: int, restart: int):
        self.switch = switch
        self.calibrate = calibrate
        self.restart = restart

    def serialize(self):
        return struct.pack('iii', self.switch, self.calibrate, self.restart)


class RobotData:
    def __init__(self, uppacked_data):

        self.uppacked_data = uppacked_data
        self.module_id = uppacked_data[0]

    def get_data_dict(self):
        # Decode data
        # start_time is used to calculate latency

        data = {}
        uppacked_data = self.uppacked_data.copy()

        data["module_id"] = uppacked_data.pop(0) # int
        data["received_dt"] = uppacked_data.pop(0) * 1e-6 # int
        data["timestamp"] = uppacked_data.pop(0) # int
        data["switch_off_request"] = uppacked_data.pop(0) # int
        data["last_rcv_timestamp"] = uppacked_data.pop(0) # float
        info = uppacked_data.pop(0) # int
        data["log_info"] = interpret_motor_msg(info)
        data["motor_pos"] = uppacked_data.pop(0) # float
        data["energy"] = uppacked_data.pop(0) # float
        data["motor_vel"] = uppacked_data.pop(0) # float
        data["motor_torque"] = uppacked_data.pop(0) # float
        data["voltage"] = uppacked_data.pop(0) # float
        data["current"] = uppacked_data.pop(0) # float
        data["temperature"] = uppacked_data.pop(0) # int
        motor_mode_error = uppacked_data.pop(0) # int
        data["add_error"] = uppacked_data.pop(0) # int
        
        data["euler_imu"] = [uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)]
        data["body_rot_imu"] = np.array([uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)])
        data["body_omega_imu"] = np.array([uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)])
        data["acc_body_imu"] = np.array([uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)])
        data['esp_errors'] = [uppacked_data.pop(0), uppacked_data.pop(0)]

        # Process data
        data["motor_mode"] = (motor_mode_error >> 6) & 0x03
        data["motor_error"] = motor_mode_error & 0x3F
        data["motor_on"] = data["motor_mode"]  == 2

        

        return data

    @staticmethod
    def unpack(data, struct_format):
        assert struct_format == "iiiififfffffiiifffffffffffffii"
        uppacked_data = struct.unpack(struct_format, data)
        return RobotData(list(uppacked_data))
    


def half_to_float(value):
    return np.float16(value).astype(np.float32)

class RobotDataLite:
    def __init__(self, uppacked_data):

        self.uppacked_data = uppacked_data



        self.module_id = uppacked_data[0]

    def get_data_dict(self, start_time=0):
        # Decode data
        # start_time is used to calculate latency

        data = {}
        uppacked_data = self.uppacked_data.copy()

        data["module_id"] = uppacked_data.pop(0)
        data["received_dt"] = uppacked_data.pop(0) * 1e-6
        data["timestamp"] = uppacked_data.pop(0)
        data["switch_off_request"] = uppacked_data.pop(0)
        data["last_rcv_timestamp"] = uppacked_data.pop(0)
        info = uppacked_data.pop(0)
        data["log_info"] = interpret_motor_msg(info)
        data["motor_pos"] = uppacked_data.pop(0)
        data["motor_vel"] = uppacked_data.pop(0)
        data["motor_torque"] = uppacked_data.pop(0)
        data["voltage"] = uppacked_data.pop(0)
        data["temperature"] = uppacked_data.pop(0)
        motor_mode_error = uppacked_data.pop(0)
        
        data["euler_imu"] = [uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)]
        data["body_rot_imu"] = np.array([uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)])
        data["body_omega_imu"] = np.array([uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)])
        data["acc_body_imu"] = np.array([uppacked_data.pop(0), uppacked_data.pop(0), uppacked_data.pop(0)])
        data['esp_errors'] = [uppacked_data.pop(0), uppacked_data.pop(0)]

        # Process data
        data["motor_mode"] = (motor_mode_error >> 6) & 0x03
        data["motor_error"] = motor_mode_error & 0x3F
        data["motor_on"] = data["motor_mode"]  == 2

        curr_timestamp = float(time.time()) - start_time
        latency = curr_timestamp - (data["last_rcv_timestamp"])
        data["latency"] = latency

        return data

    @staticmethod
    def unpack(data, struct_format):
        assert struct_format == "BBBBHBHHHHBBHHHHHHHHHHHHHBB"
        uppacked_data = struct.unpack(struct_format, data)
        data_list = list(uppacked_data)
        for format, data in zip(struct_format, data_list):
            if format == "H":
                data = half_to_float(data)

        return RobotData(data_list)
    