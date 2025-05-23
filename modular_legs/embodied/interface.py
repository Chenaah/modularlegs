from collections import defaultdict
import collections
import copy
import json
import os
import pdb
import select
import socket
import sys
import threading
import time
import struct
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np

from modular_legs.embodied.interpreter import interpret_motor_error, interpret_motor_mode

from modular_legs.natnet.NatNetClient import NatNetClient
from modular_legs.natnet import DataDescriptions

import signal
from rich.console import Console
from rich.table import Table
from rich.live import Live
import queue
from time import ctime
from datetime import datetime

from modular_legs.utils.curves import isaac_reward
from modular_legs.utils.monitor import Monitor
from modular_legs.utils.math import calculate_angular_velocity, AverageFilter, quat_rotate_inverse, world_to_body_velocity_yaw
from modular_legs.utils.kbhit import KBHit
from modular_legs.utils.logger import Logger, cache_pings, load_cached_pings
from modular_legs.utils.others import convert_np_arrays_to_lists, get_ping_time, is_number
from modular_legs.embodied.data_struct import SentDataStruct, RobotData
from modular_legs.embodied.dashboard_client import DashboardServer

def sanitize_list( l):
    return [validation_map[type(element)](element) for element in l]

def sanitize_dict( to_sanitize):
    for key, value in to_sanitize.items():
        if type(value) not in validation_map:
            print(f"[ERROR] Unrecognized type: {type(value)}")
            print(f"[ERROR] Key: {key}, Value: {value}")
        to_sanitize[key] = validation_map[type(value)](value)
    return to_sanitize


validation_map = {dict: sanitize_dict,
                  collections.OrderedDict: sanitize_dict,
                    type([]):sanitize_list,
                    np.ndarray:lambda x: x.tolist(),
                    np.float64: lambda x: x,
                    int: lambda x: x,
                    float: lambda x: x,
                    str: lambda x: x,
                    bool: lambda x: x,
                    type(None   ): lambda x: "None"}



'''
    Communicate with the hardware.
    - Send commands to the motor from Env or keyboard

'''
class Interface():

    def __init__(self, cfg):

        # Config
        self.cfg = cfg
        self.motor_commands = defaultdict(dict)
        self.pending_counter = {}
        self.update_config(cfg)
        self.enable_sim_render = True # cfg.interface.sim_render # TODO: This should be in the cfg

        self.start_time = time.time()
        self.optitrack_time = -1
        self.optitrack_data = {}
        # self.logger = Logger(alg=mode, log_dir=cfg.logging.data_dir)
        self.module_address_book = {}
        self.pending_modules = set()
        self.pings = load_cached_pings()
        self.data = {} # self.data = {module_id: {data}}

        # Actions
        self.switch_on = 0 # TODO: control modules separately
        self.target = 0.
        
        self.send_dt = 0
        self.compute_time = 0

        self.all_motor_on = False
        self.overwrite_actions = None # for debugging
        self.ready_to_go = False
        
        signal.signal(signal.SIGINT, self.signal_handler)
        self.kb = KBHit() # TODO
        print('[e] enable; [d] disable')
        self.live = Live(self._generate_table(), refresh_per_second=20)
        self.live.__enter__()

        # Setup socket and make sure all the modules are connected
        self._setup_socket(check_connection=True, protocol=self.protocol)

        if self.enable_dashboard or self.enable_sim_render:
            self.dashboard_server = DashboardServer()
            # self.dashboard.connect_to_dashboard()

        if "optitrack" in self.sources:
            # Listening to OptiTrack
            print("[Server] Connecting to OptiTrack...")
            # self.streaming_client = NatNetClient(server_ip_address="10.105.42.167", local_ip_address="129.105.69.100", use_multicast=False)
            # self.streaming_client.on_data_frame_received_event.handlers.append(self._receive_new_frame)
            # self.streaming_client.connect()
            # self.streaming_client.request_modeldef()

            self.streaming_client = NatNetClient()
            self.streaming_client.set_client_address("0.0.0.0")
            self.streaming_client.set_server_address("129.105.73.172")
            self.streaming_client.set_use_multicast(False)
            self.streaming_client.set_print_level(0)
            # streaming_client.on_data_description_received_event.handlers.append(receive_new_desc)
            # streaming_client.on_data_frame_received_event.handlers.append(receive_new_frame)

            self.streaming_client.new_frame_listener = self._receive_new_frame
            self.streaming_client.rigid_body_listener = self._receive_rigid_body_frame

            is_running = self.streaming_client.run()

            # if not is_running:
            #     raise RuntimeError("Optitrack is unhappy :(")

        # # Logging data
        if self.log_dir is not None:
            self.log_file = open(os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_raw.txt"), "w")
        
        # Logging
        self.received_dt = 0
        self.max_received_dt = 0
        self.latency = 0
        self.max_latency = 0
        self.step_counter = 0
        last_receive_time = time.time()
        max_delta_time = 0 
        self.last_frame_time = time.time()
        self.last_sent_time = time.time()
        self.last_sign_time = -100
        self.last_motor_com_time = time.time()
        self.last_rcv_timestamp = time.time()
        self.last_log_info = ""
        self.module_info_dict = {}
        self.module_lastinfo_dict = {}
        self.publish_log_info = ""
        self.num_frames = 0
        self.retrieve_msg_time = 0
        self.info_dict = {}
        self.status_dict = {}
        self.abnormal_modules = set()
        self.motor_offset = 0
        self.calibration_command_buffer = {i: [] for i in self.module_ids}
        


        self.omega_filter = AverageFilter(4)
        self.vel_filter = AverageFilter(4)
        self.acc_imu_filter = AverageFilter(cfg.interface.imu_window_size)
        self.omega_imu_filter = AverageFilter(cfg.interface.imu_window_size)
        
    def update_config(self, cfg):
        self.module_ids = cfg.interface.module_ids #  for checking connection
        self.torso_module_id = cfg.interface.torso_module_id   # 0 # For constructing observation: Assume that the module 0 is the torso module and motor data are stacked in the order of module_ids
        self.sources = cfg.interface.sources # ["imu"] # new API # TODO: This should be in the cfg
        self.struct_format = cfg.interface.struct_format # 'iiiififfffiifffffffffffffii' # TODO: This should be in the cfg
        self.protocol = cfg.interface.protocol
        self.filter_action = cfg.agent.filter_action
        self.enable_firmware_filter = cfg.interface.enable_filter
        assert self.protocol in ["UDP", "USB"], "Only support UDP and USB for now"

        self.dt = cfg.robot.dt
        self.motor_range = np.array(cfg.robot.motor_range)
        self.kp_ratio = cfg.interface.kp_ratio
        self.kd_ratio = cfg.interface.kd_ratio
        self.calibration_modes = cfg.interface.calibration_modes
        if self.calibration_modes is not None:
            assert len(self.calibration_modes) == len(self.module_ids), f"Length of calibration_modes {len(self.calibration_modes)} should be equal to the number of modules {len(self.module_ids)}"
        self.broken_motors = cfg.interface.broken_motors

        self.enable_dashboard = cfg.interface.dashboard
        self.check_action_safety = cfg.interface.check_action_safety

        self.log_dir = cfg.logging.robot_data_dir

        self._reset_motor_commands()
        for i in self.module_ids:
            self.pending_counter[i] = 0

    def _reset_motor_commands(self):
        for i in self.module_ids:
            self.motor_commands[i]["calibration"] = 0
            self.motor_commands[i]["restart"] = 0

    def _generate_table(self) -> Table:
        """Make a new table."""
        if self.ready_to_go:
            table = Table(border_style="green")
        else:
            table = Table(border_style="yellow")
        table.add_column("Module", justify="center")
        table.add_column("Connection", justify="center")
        table.add_column("Address", justify="center")
        table.add_column("Latency", justify="center")
        table.add_column("Mode", justify="center")
        table.add_column("Voltage", justify="center")
        table.add_column("Current", justify="center")
        table.add_column("Energy", justify="center")
        table.add_column("Torque", justify="center")
        table.add_column("Switch", justify="center")
        table.add_column("Error", justify="center")


        table.add_section()

        for module_id in self.module_ids:
            addr = "..."
            conn = "[red]Disconnected"
            mode = "Unknown"
            voltage = "Unknown"
            current = "Unknown"
            energy = "Unk"
            torque = "Unknown"
            switch = "[green]On" if self.switch_on else "[red]Off"
            error = "Unknown"
            ping = "Unknown"
            if module_id in self.module_address_book:
                conn = "[green]Connected"
                addrs = self.module_address_book[module_id]
                addr = f"{addrs[0]}:{addrs[1]}"
            if module_id in self.pending_modules:
                conn = "[yellow]Lost"
                # magenta
            if module_id in self.data:
                mode = "[green]Running" if self.data[module_id]["motor_mode"] == 2 else f"[red]{interpret_motor_mode(self.data[module_id]['motor_mode'])}"  
                error_id = self.data[module_id]["motor_error"]
                error = interpret_motor_error(error_id)
                if error == "":
                    if self.data[module_id]['esp_errors'][0] != 1:
                        error = f"ESP32 Rebooted ({self.data[module_id]['esp_errors'][0]})"
                error = "[red]" + error if error != "" else "[green]None"
                # error += str(self.data[module_id]["add_error"])
                raw_v = self.data[module_id]['voltage']
                # voltage = f"{0.0215*(raw_v-940)+24.4:.2f} V" if raw_v != 0 else "Unknown"
                voltage = f"{raw_v:.2f} V" if raw_v != 0 else "Unknown"
                raw_current = self.data[module_id]['current']
                # actual_current = 0.01146812726623845 + 0.7974886681113309 * raw_current - 0.015117591222085305 *raw_current**2
                current = f"{raw_current/1:10.6f} A"
                energy = f"{self.data[module_id]['energy']:.2f} J"
                torque = f"{self.data[module_id]['motor_torque']:.2f} Nm"
            if module_id in self.pings:
                ping = f"{self.pings[module_id]:.2f} ms"
            table.add_row(f"[bold]{module_id}", conn, addr, ping, mode, voltage, current, energy, torque, switch, error, end_section=True)

        table.add_row("[bold]dt", f"{self.send_dt:.3f}", "[bold]Comp Time", f"{self.compute_time:.3f}", "", "", "", "", "", "", "", end_section=True)
        return table
    

    def _update_motor_commands(self):
        # Receive motor commands from the dashboard (or gamepad in the future)
        enable, disable, calibrate, reset, debug_pos_list = self.dashboard_server.get_commands()
        # print("[Server] Received commands: ", enable, disable, calibrate, reset, debug_pos_list)
        if enable:
            self.switch_on = 1
        if disable:
            self.switch_on = 0
        if calibrate:
            for i in self.module_ids:
                self.motor_commands[i]["calibration"] = 1
            self.last_motor_com_time = time.time()
        if reset:
            for i in self.module_ids:
                self.motor_commands[i]["restart"] = 1
            self.last_motor_com_time = time.time()
        if debug_pos_list is not None:
            self.overwrite_actions = np.array(debug_pos_list)

        if time.time() - self.last_motor_com_time > 0.5:
            # Reset motor commands
            self._reset_motor_commands()


    def _setup_socket(self, check_connection=True, protocol="UDP"):
        if protocol != "UDP":
            raise NotImplementedError
        
        print("[Server] Setting up socket server...")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Setting buffer size
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        server_address = ('0.0.0.0', 6666)
        self.server_socket.bind(server_address)

        if check_connection:
            # Check if all the modules are connected
            to_be_connected = copy.deepcopy(self.module_ids)
            while to_be_connected:
                # print(f"[Server] Waiting for modules {to_be_connected} to connect...")
                data, client_address = self.server_socket.recvfrom(struct.calcsize(self.struct_format))
                uppacked_data = struct.unpack(self.struct_format, data)
                uppacked_data = list(uppacked_data)
                module_id = uppacked_data[0] # TODO: May change
                if module_id in to_be_connected:
                    to_be_connected.remove(module_id)
                    self.module_address_book[module_id] = client_address
                print("Recieved data from module: ", module_id)
                # print(f"[Server] Connection established with module {module_id} : ", client_address)
                self.pending_modules.update(set(to_be_connected))
                if module_id not in self.pings:
                    self.pings[module_id] = get_ping_time(client_address[0])
                    cache_pings(self.pings)

                self.live.update(self._generate_table())

        self.server_socket.setblocking(0)


    def receive_module_data(self):
        # Receive message from client
        # New API,
        # This should allow: multiple modules to connect to the server / they can change IP address during resetting
        # data_dict = {}
        to_be_connected = copy.deepcopy(self.module_ids)
        ready = select.select([self.server_socket], [], [], 5)[0]
        while not ready:
            print(f"[ERROR][Server] I didn't hear angthing from Module {to_be_connected}!")
            self.pending_modules.update(set(to_be_connected))
            ready = select.select([self.server_socket], [], [], 5)[0]
            self.live.update(self._generate_table())

        for _ in range(100):
            # Empty the socket buffer while also preventing the server from getting stuck in an infinite loop
            try:
                data, address = self.server_socket.recvfrom(struct.calcsize(self.struct_format))
            except BlockingIOError:
                break
            robot_data = RobotData.unpack(data, self.struct_format)
            module_id = robot_data.module_id

            data_dict =  robot_data.get_data_dict()

            # Calculate the latency
            curr_timestamp = float(time.time()) - self.start_time
            data_dict["latency"] = (curr_timestamp - data_dict["last_rcv_timestamp"] - self.dt) / 2

            # Store the data
            self.data[module_id] = data_dict
            self.module_info_dict[module_id] = data_dict["log_info"]
            if module_id in self.module_lastinfo_dict:
                if data_dict["log_info"] != self.module_lastinfo_dict[module_id] and data_dict["log_info"] != "":
                    print(f"[ESP32 MESSAGE] [Module {module_id}] ", data_dict["log_info"])

                    self.module_lastinfo_dict[module_id] = data_dict["log_info"]
            else:
                self.module_lastinfo_dict[module_id] = data_dict["log_info"]
            self.publish_log_info = data_dict["log_info"]

            if data_dict["switch_off_request"]:
                print(f"[Server] Receive switch off request from Module {module_id}. Switch off!")
                self._disable_motor()
            self.latency = data_dict["latency"]

            if module_id in to_be_connected:
                to_be_connected.remove(module_id)
                self.module_address_book[module_id] = address # Update the address book, which allows the client to change IP address
            if module_id in self.pending_modules:
                self.pending_modules.remove(module_id)
                self.pending_counter[module_id] = 0

        if self.log_dir is not None:
            self.log_file.write(json.dumps(sanitize_dict(copy.deepcopy(self.data))) + "\n")
            self.log_file.flush()

        
        if to_be_connected:
            # Miss data from some modules
            # While the sending rate is higher than the receiving rate, this should not happen
            # print(f"[WARN][Server] I didn't hear from Module {to_be_connected}!")
            self.pending_modules.update(set(to_be_connected))
            for i in to_be_connected:
                self.pending_counter[i] += 1
                
        
        on_modules = set(self.data.keys()) & set(self.module_ids)
        self.all_motor_on = all([self.data[module_id]["motor_on"] for module_id in on_modules])
        # self.all_motor_on = all([self.data[module_id]["motor_on"] for module_id in self.module_ids])
        
        self._check_health()
        

    def _check_health(self):

        if not self.all_motor_on and self.switch_on:
            if not all([not self.data[module_id]["motor_on"] for module_id in self.module_ids]):
                # Not all modules are off
                for module_id in self.module_ids:
                    if not self.data[module_id]["motor_on"]:
                        self.abnormal_modules.add(module_id)
                        print("[DEBUG] Module current: ", self.data[module_id]['current'])
                print(f"[ERROR][Server] Not all modules are on! Abnormal modules: {self.abnormal_modules}")
        for i in self.module_ids:
            if self.pending_counter[i] > 5/self.dt:
                print(f"[ERROR][Server] Module {i} is not connected!")
                self.abnormal_modules.add(i)
            else:
                if i in self.data and i in self.abnormal_modules:
                    if self.data[i]["motor_on"]:
                        self.abnormal_modules.remove(i)
        if self.abnormal_modules:
            print(f"[Server] Abnormal modules: {self.abnormal_modules}")

    def _action_safety_check(self, target, module_id):
        # Check if the target is too far from the current position
        if module_id in self.data:
            curr_pos = self.data[module_id]["motor_pos"]
            if abs(target - curr_pos) > 3.14:
                print(f"[WARN][Server] Module {module_id} target {target} is too far from the current position {curr_pos}!")
                return curr_pos + 0.1*np.tanh(target - curr_pos)
            else:
                return target
        else:
            print(f"[WARN][Server] Module {module_id} is not connected!")
            return None


    def send_action(self, 
                    pos_actions: np.ndarray, 
                    vel_actions: Optional[np.ndarray] = None,
                    kps: Optional[np.ndarray] = 8,
                    kds: Optional[np.ndarray] = 0.2
                    ):
        
        # print(f"[DEBUG] Sending actions: {pos_actions} {vel_actions} {kps} {kds}")
        if self.enable_dashboard:
            self._update_motor_commands()

        if vel_actions is None:
            vel_actions = np.zeros_like(pos_actions)

        if self.overwrite_actions is not None:
            pos_actions = self.overwrite_actions # TODO: for compatibility with the dashboard
            print("[Server] Overwrite actions: ", pos_actions)

        assert len(pos_actions) == len(self.module_ids), f"Length of pos_actions {len(pos_actions)} should be equal to the number of modules {len(self.module_ids)}"
        assert len(vel_actions) == len(self.module_ids), f"Length of vel_actions {len(vel_actions)} should be equal to the number of modules {len(self.module_ids)}"
        assert len(kps) == len(self.module_ids), f"Length of kps {len(kps)} should be equal to the number of modules {len(self.module_ids)}"
        assert len(kds) == len(self.module_ids), f"Length of kds {len(kds)} should be equal to the number of modules {len(self.module_ids)}"
            
        self.actions = pos_actions if vel_actions is None else vel_actions # for logging
        # pos_actions = np.clip(pos_actions, self.motor_range[0], self.motor_range[1])
        vel_actions = np.clip(vel_actions, -20, 20) # TODO: This should be in the cfg

        kps_real = kps*self.kp_ratio
        kds_real = kds*self.kd_ratio
        kps_real = np.clip(kps_real, 0, 100) # TODO: This should be in the cfg
        kds_real = np.clip(kds_real, 0, 100) # TODO: This should be in the cfg

        if self.broken_motors is not None:
            kps_real[self.broken_motors] = 0
            kds_real[self.broken_motors] = 0
            print("Kp: ", kps_real)

        self.curr_timestamp = time.time()-self.start_time
        for target_pos, target_vel, module_id, kp, kd in zip(pos_actions, vel_actions, self.module_ids, kps_real, kds_real):
            # Switch / calibration / restart for ALL modules for now
            # Calibration:
            #     0: No calibration
            #     1: Manual-Calibration
            #     2: Auto-Calibration
            #     3: Setting the zero position

            if self.check_action_safety:
                target_pos = self._action_safety_check(target_pos, module_id)
                if target_pos is None:
                    continue


            data_to_send = [target_pos, 
                            target_vel,
                            kp,
                            kd,
                            int(self.enable_firmware_filter),
                            self.switch_on, 
                            self.calibration_command_buffer[module_id].pop(0) if self.calibration_command_buffer[module_id] else 0, 
                            self.motor_commands[module_id]["restart"], 
                            self.curr_timestamp]
            if self.motor_commands[module_id]["restart"] == 1: # Reset the command
                print(f"[Server] Restarting Module {module_id}...")
            # print("data_to_send ", data_to_send)
            self._send_msg(data_to_send, self.module_address_book[module_id])
            # print(f"Send to Module {module_id} (in {self.module_ids}): {self.module_address_book[module_id]}")
            self.step_counter += 1

    def _send_msg(self, data: list, address: tuple):
        data = SentDataStruct(*data)
        serialized_data = data.serialize()

        if not self.server_socket._closed == True:
            self.server_socket.sendto(serialized_data, address)


    # def _connect_to_motor(self):
    #     print("[Server] Connecting to ESP32...")
    #     self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     # Setting buffer size
    #     self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    #     server_address = ('0.0.0.0', 4321)
    #     self.server_socket.bind(server_address)
    #     data, self.client_address = self.server_socket.recvfrom(1024)
    #     self.server_socket.setblocking(0)

    #     print(f'[Server] Connection established with ESP32: {self.client_address}')

    # def _reconnect_to_motor(self):
    #     self.server_socket.close()
    #     try:
    #         self._connect_to_motor()
    #         self._reset()
    #     except OSError as e:
    #         print("[ERROR][Server] Reconnection fails ")

    def _restart_motor(self, module_id=None):
        self._disable_motor()
        print(f'[Server] Try to restart ESP32')
        if module_id is None:
            for i in self.module_ids:
                self.motor_commands[i]["restart"] = 1
        elif module_id == "auto":
            for i in self.abnormal_modules:
                self.motor_commands[i]["restart"] = 1
            if not self.abnormal_modules:
                for i in self.module_ids:
                    self.motor_commands[i]["restart"] = 1
        else:
            self.motor_commands[module_id]["restart"] = 1
            
        self._reset()
        # TODO: reconnect

    def _fix_motor(self, module_id=None):
        self._disable_motor()
        print(f'[Server] Try to fix ESP32')
        if module_id == "auto":
            for i in self.abnormal_modules:
                self.motor_commands[i]["calibration"] = 2
            
        self._reset()

    def _reset(self):
        self.max_received_dt = 0
        self.max_latency = 0



    def signal_handler(self, signum, frame):
        print("[Server] Sending stop signal to the motor...")
        # self.streaming_client.shutdown()
        for _ in range(10):
            self.switch_on = 0
            zeros = np.zeros(len(self.module_ids))
            self.send_action(zeros, kps=zeros, kds=zeros)
            time.sleep(0.01)
        raise NotImplementedError
        sys.exit()


    
    def ready(self):
        self.ready_to_go = self.all_motor_on and self.switch_on # TODO: Check all modules
        # if not ready:
        #     if not self.all_motor_on:
        #         print("[Server] Not all modules are on!")
        #     if not self.switch_on:
        #         print("[Server] Switch is off!")
        #     for module_id in self.module_ids:
        #         if module_id in self.data and not self.data[module_id]["motor_on"]:
        #             print(f"[Server] Module {module_id} is not ready!")
        self.live.update(self._generate_table())
        return self.ready_to_go
        
    
    def _receive_new_frame(self, data_frame):
        self.optitrack_time = data_frame["frame_number"]
        # print("self.optitrack_time ", self.optitrack_time)

    def _receive_rigid_body_frame(self, new_id, pos, rot):
        self.optitrack_data[new_id] = [list(pos), list(rot)]

    def get_observable_data(self):
        self.observable_data = {}
        self.data_source = {}
        for source in self.sources:
            # Order in self.sources definds the overwriting order
            if source == "imu":
                self.observable_data["acc_body"] = self.data[self.torso_module_id]["acc_body_imu"]
                # print("ACC BODY: ", self.observable_data["acc_body"])
                # print("ACC NORM: ", np.linalg.norm(self.observable_data["acc_body"]))

                # action_rate_reward = isaac_reward(0, np.linalg.norm(self.observable_data["acc_body"]), 2)
                # print(f"ACC RWD: {action_rate_reward:.3f}")

                
                self.observable_data["ang_vel_body"] = self.data[self.torso_module_id]["body_omega_imu"]
                self.observable_data["quat"] = self.data[self.torso_module_id]["body_rot_imu"]
                self.data_source.update({k: "IMU" for k in ["acc_body", "ang_vel_body", "quat"]})
            elif source == "optitrack":
                pass
            #     self.observable_data["pos_world"] = self.pos_world_opti
            #     self.observable_data["vel_body"] = self.vel_body_opti
            #     self.observable_data["ang_vel_world"] = self.body_omega_opti
            #     self.observable_data["quat"] = self.body_rot_opti
            #     self.data_source.update({k: "Optitrack" for k in ["pos_world", "vel_body", "ang_vel_world", "quat"]})
            elif source == "uwb":
                self.observable_data["pos_world"] = self.data[self.torso_module_id]["pos_world_uwb"]
                self.observable_data["vel_world"] = self.data[self.torso_module_id]["vel_world_uwb"]
                self.data_source.update({k: "UWB" for k in ["pos_world", "vel_world"]})
            elif source == "gps":
                raise NotImplementedError
            else:
                raise NotImplementedError
            

        # sorted_data = [self.data[id] for id in sorted(self.data.keys())]
        sorted_data = [self.data[id] for id in self.module_ids] # The observations should be ordered in the order of module_ids
                                                                # which should be consistent with the order of actions
        
        self.observable_data["dof_pos"] = np.array([sorted_data[i]["motor_pos"] for i in range(len(sorted_data))])
        self.observable_data["energy"] = np.array([sorted_data[i]["energy"] for i in range(len(sorted_data))])
        # print("motor_pos:  ", np.array([sorted_data[i]["motor_pos"] for i in range(len(sorted_data))]))
        # print("large_motor_pos:  ", np.array([sorted_data[i]["large_motor_pos"] for i in range(len(sorted_data))]))
        # print(self.observable_data)
        self.observable_data["dof_vel"] = np.array([sorted_data[i]["motor_vel"] for i in range(len(sorted_data))])

        # New data for a general state space
        self.observable_data["quats"] = np.array([sorted_data[i]["body_rot_imu"] for i in range(len(sorted_data))])
        self.observable_data["gyros"] = np.array([sorted_data[i]["body_omega_imu"] for i in range(len(sorted_data))])
        self.observable_data["accs"] = np.array([sorted_data[i]["acc_body_imu"] for i in range(len(sorted_data))])

        # Robot state data
        self.observable_data["robot_switch_on"] = self.switch_on
        self.observable_data["robot_motor_torque"] = [sorted_data[i]["motor_torque"] for i in range(len(sorted_data))]
        self.observable_data["robot_send_dt"] = self.send_dt # TODO
        self.observable_data["robot_received_dt"] = np.array([self.received_dt, self.max_received_dt]) # TODO
        self.observable_data["robot_latency"] = np.array([self.latency, self.max_latency]) # TODO
        self.observable_data["robot_temperature"] = [sorted_data[i]["temperature"] for i in range(len(sorted_data))]
        self.observable_data["robot_voltage"] = [sorted_data[i]["voltage"] for i in range(len(sorted_data))]
        self.observable_data["robot_current"] = [sorted_data[i]["current"] for i in range(len(sorted_data))]
        self.observable_data["robot_motor_error"] = [sorted_data[i]["motor_error"] for i in range(len(sorted_data))]
        self.observable_data["robot_motor_mode"] = [sorted_data[i]["motor_mode"] for i in range(len(sorted_data))]
        self.observable_data["robot_esp_errors"] = [sorted_data[i]['esp_errors'] for i in range(len(sorted_data))]
        self.observable_data["robot_motor_commands"] = [i for i in self.motor_commands.values()] # TODO
        self.observable_data["robot_motor_message"] = [self.publish_log_info] # [sorted_data[i]["log_info"] for i in range(len(sorted_data))] #TODO
        self.observable_data["optitrack_time"] = self.optitrack_time
        for key, value in self.optitrack_data.items():
            self.observable_data[f"optitrack_rigibody{key}"] = value

        # print("self.observable_data")
        # print(self.observable_data)

        if self.enable_dashboard:
            self.dashboard_server.send_data(convert_np_arrays_to_lists(self.observable_data))
        self.live.update(self._generate_table())


        return self.observable_data

        
    def _enable_motor(self):
        self.switch_on = 1
        self.last_sign_time = time.time()
        self.step_counter = 0
        # self.action_filter.reset()
        # Testing
        if self.calibration_modes is not None:
            for i, module in enumerate(self.module_ids):
                self.calibration_command_buffer[module] = [self.calibration_modes[i]]*20

    def _disable_motor(self):
        self.switch_on = 0
        
    # def update_ui(self):

    #     if self.kb.kbhit():
    #         c = self.kb.getch()
    #         # if ord(c) == 27: # ESC
    #         #     break
    #         if c in ["e", "E", "d", "D", "w", "W", "s", "S", "c", "C", "r", "R", "i", "I"]:
    #             print(f"[Server] Key {c} is pressed.")
    #         if c == "e" or c == "E":
    #             self._enable_motor()
    #         elif c == "d" or c == "D":
    #             self._disable_motor()
    #         elif c == "w" or c == "W":
    #             self.commands[0] = min(self.commands[0]+0.1, self.command_range[1])
    #         elif c == "s" or c == "S":
    #             self.commands[0] = max(self.commands[0]-0.1, self.command_range[0])
    #         elif c == "c" or c == "C":
    #             for i in self.module_ids:
    #                 self.motor_commands[i]["calibration"] = 1
    #             self.last_motor_com_time = time.time()
    #         elif c == "r" or c == "R":
    #             self._restart_motor()
    #             self.last_motor_com_time = time.time()
    #         elif c == "i" or c == "I":
    #             self.switch_on = 0
    #             self._reconnect_to_motor()
    #         elif is_number(c):
    #             # Restart the ith module
    #             self._restart_motor(int(c))
    #             self.last_motor_com_time = time.time()

    #     if time.time() - self.last_motor_com_time > 0.5:
    #         # Reset motor commands
    #         self._reset_motor_commands()
    

    def log_raw_data(self):
        data_to_log = {}
        for k in self.observable_data:
            info_k = f"{self.data_source[k]}/{k}" if k in self.data_source else k
            data_to_log[info_k] = self.observable_data[k]
        data_to_log["action"] = self.actions
        

            
            

            
