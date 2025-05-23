


import select
import socket
import msgpack



class DashboardServer:
    # This class is responsible for sending data for logging / visualization

    def __init__(self):
        self._setup_socket()
                              
    def _setup_socket(self):
        print("[Server] Setting up dashboard socket...")
        self.dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Setting buffer size
        self.dashboard_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        server_address = ('127.0.0.1', 6667)
        # Port: Dashboard Server - 6667
        #       Dashboard - 6668
        #       Simulation renderer - 6669
        self.dashboard_socket.bind(server_address)
        # data, self.dashboard_address = self.dashboard_socket.recvfrom(1024)
        self.dashboard_socket.setblocking(0)

        self.dashboard_address = ('127.0.0.1', 6668)
        self.renderer_address = ('127.0.0.1', 6669)

    def get_commands(self):
        data = None
        for _ in range(100):
            try:
                data, _ = self.dashboard_socket.recvfrom(1024)
            except BlockingIOError:
                break

        if data is not None:
            received_data = msgpack.unpackb(data)

            enable = received_data["enable"]
            disable = received_data["disable"]
            calibrate = received_data["calibrate"]
            reset = received_data["reset"]
            debug_pos_list = received_data["slide"]
            print("Received commands from Dashboard. debug_pos_list: ", debug_pos_list)
        else:
            enable, disable, calibrate, reset, debug_pos_list = 0, 0, 0, 0, None

        return enable, disable, calibrate, reset, debug_pos_list
    
    def send_data(self, observable_data: dict):
        serialized_data = msgpack.packb(observable_data)

        self.dashboard_socket.sendto(serialized_data, self.dashboard_address)
        # print("Sent data to Dashboard.")
        self.dashboard_socket.sendto(serialized_data, self.renderer_address)
        # print("Sent data to Renderer.")
