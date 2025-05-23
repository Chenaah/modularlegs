import socket
import threading
import time
import struct
import json
import math
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

from monitor import Monitor

target = 0

if __name__ == "__main__":
    # Create a UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the server address and port
    server_address = ('0.0.0.0', 4321)
    server_socket.bind(server_address)

    print('Server is listening...')

    # Accept connection from client
    data, client_address = server_socket.recvfrom(1024)
    print('Connection established with client:', client_address)

    # Create a thread to send messages to the client
    last_receive_time = time.time()
    max_delta_time = 0 

    monitor = Monitor(["Motor Position", "Motor Velocity", "Motor Torque", "Motor temperature", "IMU Orientation - x", "IMU Orientation - y", "IMU Orientation - z"], 
                      [(-1,1), (-2,2), (-2,2), (20,40), (-360,360), (-360,360), (-360,360)],
                      shape=(2,4))

    while True:

        # Receive message from client.decode('utf-8')
        data, _ = server_socket.recvfrom(1024)
        current_time = time.time()
        delta_t = current_time - last_receive_time
        last_receive_time = current_time
        max_delta_time = max(delta_t, max_delta_time)
        # print('Received message from client:', list(data))
        json_data = json.loads(data.decode('utf-8'))
        monitor.plot([json_data['motor']["pos"], json_data['motor']["vel"], json_data['motor']["torque"], json_data['motor']["temperature"],
                      json_data['imu']["orientation"][0], json_data['imu']["orientation"][1], json_data['imu']["orientation"][2]])
        print(json_data)
        # print(delta_t, max_delta_time)
        # print(json_data)


