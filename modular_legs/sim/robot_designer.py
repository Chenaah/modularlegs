


import copy
import os
import pdb
from collections import defaultdict
import tempfile
import time
import numpy as np
from numpy import cos, sin, tan, pi
import yaml
from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.designer_utils import movability_check, self_collision_check, stable_state_check
from modular_legs.sim.robot_builder import RobotBuilder
from modular_legs.utils.math import calculate_transformation_matrix, construct_quaternion, matrix_to_pos_quat, quaternion_multiply2, rotation_matrix, rotation_matrix_sequence
from modular_legs.utils.model import quaternion_from_vectors, is_headless
from modular_legs.utils.model import get_jing_vector, get_joint_pos_addr, get_local_zvec
import mujoco

from modular_legs.utils.others import is_list_like, is_number, numpy_to_native
from modular_legs.utils.visualization import view
if not is_headless():
    import mujoco.viewer


DEFAULT_ROBOT_CONFIG = {
    "theta": 0.4625123,
    "R": 0.07,
    "r": 0.03,
    "l_": 0.236,
    "delta_l": 0,
    "stick_ball_l": -0.1,
    "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
    "stick_mass": 0.26,
    "top_hemi_mass": 0.74,
    "bottom_hemi_mass": 0.534
}

class Connect(object):

    class Dock(object):
        def __init__(self, pos_a, pos_b, rotate_a, rotate_b, position_a):
            self.pos_a = pos_a # local position of the docking point in the parent module
            self.pos_b = pos_b # local position of the docking point in the child module
            self.rotate_a = rotate_a # rotation of the docking point in the parent module
            self.rotate_b = rotate_b # rotation of the docking point in the child module
            self.position_a = position_a # position of the docking point in the parent module

    def __init__(self, conn_type, robot_cfg, lite=False):

        R = robot_cfg["R"]
        delta_l = robot_cfg["delta_l"]
        theta = robot_cfg["theta"]
        l = robot_cfg["l"]
        stick_ball_l = robot_cfg["stick_ball_l"]
        r = robot_cfg["r"]
        a = robot_cfg["a"]

        self.lite = lite
        self.j_list = [i for i in range(4)]
        self.j_c_list = [i for i in range(4)]

        screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3]

        self.dock_list = []
        if conn_type == "00" or conn_type == "10" or conn_type == "02" or conn_type == "12":
            # ball <- ball or ball <- stick
            sides = [1, -1]
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            dock_pos = [0, 2*pi/3, 4*pi/3] # docking positions along the hemisphere

            for i, dock_theta in enumerate(dock_pos):
                for screw_theta in screw_thetas:
                    pos_a = [(R-delta_l)*cos(theta)*sin(dock_theta), 
                                (R-delta_l)*cos(theta)*cos(dock_theta), 
                                (R-delta_l)*sin(theta)]
                    rotate_a = rotation_matrix_sequence([
                        rotation_matrix([0,0,1],dock_theta),
                        rotation_matrix(pos_a,-(pi/3+screw_theta))
                    ])
                    if conn_type == "00" or conn_type == "10":
                        pos_b = [0, (R-delta_l)*cos(theta), (R-delta_l)*sin(theta)]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,0,1],pi-dock_theta),
                            rotation_matrix([1,0,0],2*theta)
                            
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i))
                    elif conn_type == "02" or conn_type == "12":
                        pos_b = [0, 0, (l-stick_ball_l)/2]
                        rotate_b = rotation_matrix_sequence([
                                rotation_matrix([cos(dock_theta),sin(dock_theta),0],-pi/2-theta),
                                rotation_matrix([0,0,1],pi/6)
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i))
                        
                        for j in self.j_list:
                            pos_b = [r*cos(j*pi/2+pi/12), r*sin(j*pi/2+pi/12), a]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],pi/2+j*pi/2-dock_theta),
                                rotation_matrix([-sin(j*pi/2),cos(j*pi/2),0],-theta),
                                rotation_matrix([0,0,1],pi/12)
                            ])
                            self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i))
                        
        elif conn_type == "20" or conn_type == "22":
            # stick <- ball or stick <- stick
            sides = [1, -1]
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            pos_counter = 0

            for side in sides:
                # tip of stick a
                pos_a = [0, 0, (l-stick_ball_l)/2*side]
                for screw_theta in screw_thetas:
                    if conn_type == "20":
                        rotate_a = rotation_matrix_sequence([
                            rotation_matrix([1,0,0],-side*pi/2-theta),
                            rotation_matrix([0,0,1],-pi/6+screw_theta)
                        ])
                        pos_b = [0, (R-delta_l)*cos(theta), (R-delta_l)*sin(theta)]
                        rotate_b = np.eye(3)
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter))
                    elif conn_type == "22":
                        rotate_a = rotation_matrix_sequence([
                            rotation_matrix([0,0,1],screw_theta)
                        ])
                        pos_b = [0, 0, (l-stick_ball_l)/2]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([1,0,0],pi/2+side*pi/2),
                            rotation_matrix([0,0,1],pi/2+side*pi/2)
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter))

                        for j_b in self.j_list:
                            pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), a]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,1,0],-side*pi/2),
                                rotation_matrix([0,0,1],j_b*pi/2+pi/12)
                            ])
                            # rotate_b = np.eye(3)
                            self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter))

                pos_counter += 1
                # side of stick a

                for j_a in self.j_list:
                    pos_a = [r*cos(j_a*pi/2+pi/12)*side, r*sin(j_a*pi/2+pi/12)*side, a*side]
                    rotate_a = rotation_matrix_sequence([
                        rotation_matrix([0,0,1],pi/12)
                    ])
                    for screw_theta in screw_thetas:
                        if conn_type == "20":
                            pos_b = [0, (R-delta_l)*cos(theta), (R-delta_l)*sin(theta)]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],-side*pi/2-j_a*pi/2),
                                rotation_matrix([1,0,0],theta),
                                rotation_matrix(pos_b,screw_theta+pi/2-side*pi/2)
                            ])
                            self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter))
                        elif conn_type == "22":
                            pos_b = [0, 0, (l-stick_ball_l)/2]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([-sin(j_a*pi/2),cos(j_a*pi/2),0],pi/2*side),
                                rotation_matrix([0,0,1],-j_a*pi/2+screw_theta)
                            ])
                            self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter))
                            if screw_theta == 0 or screw_theta == pi:
                                continue

                            for j_b in self.j_list:
                                pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), a]
                                rotate_b = rotation_matrix_sequence([
                                    rotation_matrix([0,1,0],side*pi/2+pi/2),
                                    rotation_matrix([0,0,1],j_b*pi/2+side*j_a*pi/2),
                                    rotation_matrix([cos(j_b*pi/2), sin(j_b*pi/2), 0], screw_theta),
                                    rotation_matrix([0,0,1],pi/12)
                                ])
                                self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter))
                    pos_counter += 1

        # elif conn_type == "02" or conn_type == "12":
        #     # ball <- stick
        #     pos_a = [0,0,0]
        #     pos_b = [0, (R-delta_l)*cos(theta), (R-delta_l)*sin(theta)]
        

    def __getitem__(self, index):
        # Return the element at the specified index
        return self.dock_list[index]

class ConnectAsym(object):

    class Dock(object):
        def __init__(self, pos_a, pos_b, rotate_a, rotate_b, position_a, position_b):
            self.pos_a = pos_a # local position of the docking point in the parent module
            self.pos_b = pos_b # local position of the docking point in the child module
            self.rotate_a = rotate_a # rotation of the docking point in the parent module
            self.rotate_b = rotate_b # rotation of the docking point in the child module
            self.position_a = position_a # position of the docking point in the parent module
            self.position_b = position_b # position of the docking point in the child module

    def __init__(self, conn_type, robot_cfg, lite=False):

        R = robot_cfg["R"]
        delta_l = robot_cfg["delta_l"]
        theta = robot_cfg["theta"]
        l = robot_cfg["l"]
        stick_ball_l = robot_cfg["stick_ball_l"]
        r = robot_cfg["r"] - stick_ball_l
        a = robot_cfg["a"]

        self.lite = lite
        self.j_list = [1,3]
        self.j_c_list = [0,2]
        
        side_pos_list = [a, 0, -a]
        side_j_list = [self.j_list, self.j_c_list, self.j_list]

        screw_thetas = [0, 2*pi/3, 4*pi/3]

        self.dock_list = []
        
        dock_pos = [0, 2*pi/3, 4*pi/3] # docking positions along the hemisphere

        # conn_type == "00" or conn_type == "10" or conn_type == "02" or conn_type == "12"
        if conn_type[0] in ["0", "1"]:
            # ball <- ball or ball <- stick
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position

            for i, dock_theta_a in enumerate(dock_pos):
                for screw_theta in screw_thetas:
                    pos_a = [(R-delta_l)*cos(theta)*sin(dock_theta_a), 
                                (R-delta_l)*cos(theta)*cos(dock_theta_a), 
                                (R-delta_l)*sin(theta)]
                    rotate_a = rotation_matrix_sequence([
                        rotation_matrix([0,0,1],dock_theta_a),
                        rotation_matrix(pos_a,-(pi/3+screw_theta))
                    ])
                    if conn_type == "00" or conn_type == "10":
                        for j, dock_theta_b in enumerate(dock_pos):
                            pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                        (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                        (R-delta_l)*sin(theta)]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],pi-dock_theta_a),
                                rotation_matrix([1,0,0],2*theta),
                                rotation_matrix([0,0,1],-dock_theta_b)
                                
                            ])
                            self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i, j))
                    # elif conn_type == "01" or conn_type == "11":
                    #     for j, dock_theta_b in enumerate(dock_pos):
                    #         pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                    #                     -(R-delta_l)*cos(theta)*cos(dock_theta_b), 
                    #                     -(R-delta_l)*sin(theta)]
                    #         rotate_b = rotation_matrix_sequence([
                    #             rotation_matrix([0,0,1],-dock_theta_a),
                    #             rotation_matrix([0,0,1],-dock_theta_b),
                    #             rotation_matrix(pos_b, pi/3),
                                
                    #         ])
                    #         self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i, j))
                    elif conn_type == "02" or conn_type == "12":
                        pos_counter_b = 0
                        pos_b = [0, 0, (l-stick_ball_l)/2]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([cos(dock_theta_a),sin(dock_theta_a),0],-pi/2-theta),
                            rotation_matrix([0,0,1],pi/6)
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                        pos_counter_b += 1

                        for m in range(3):
                            for j in side_j_list[m]:
                                pos_b = [r*cos(j*pi/2+pi/12), r*sin(j*pi/2+pi/12), side_pos_list[m]]
                                rotate_b = rotation_matrix_sequence([
                                    rotation_matrix([0,0,1],pi/2+j*pi/2-dock_theta_a),
                                    rotation_matrix([-sin(j*pi/2),cos(j*pi/2),0],-theta),
                                    rotation_matrix([1,0,0],pi*((m+1)%2)),
                                    rotation_matrix([0,0,1],pi/12+pi*((m+1)%2))
                                ])
                                self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                                pos_counter_b += 1

                        pos_b = [0, 0, -(l-stick_ball_l)/2]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([cos(dock_theta_a),sin(dock_theta_a),0],-pi/2-theta),
                            rotation_matrix([0,0,1],pi/3),
                            rotation_matrix([1,0,0],pi),
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                        pos_counter_b += 1
                        
        elif conn_type == "20":
            # stick <- ball
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            pos_counter = 0

            # tip of stick a
            pos_a = [0, 0, (l-stick_ball_l)/2]
            for screw_theta in screw_thetas:
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([1,0,0],-pi/2-theta),
                    rotation_matrix([0,0,1],-pi/6+screw_theta)
                ])
                
                for i, dock_theta_b in enumerate(dock_pos):
                    pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                        (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                        (R-delta_l)*sin(theta)]
                    rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],-dock_theta_b)
                            ])
                    self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, i))
                
            pos_counter += 1

            # side of stick a
            for m in range(3):
                for j_a in side_j_list[m]:
                    pos_a = [r*cos(j_a*pi/2+pi/12), r*sin(j_a*pi/2+pi/12), side_pos_list[m]]
                    rotate_a = rotation_matrix_sequence([
                        rotation_matrix([1,0,0],pi*((m+1)%2)),
                        rotation_matrix([0,0,1],pi/12)
                    ])
                    for screw_theta in screw_thetas:
                        for i, dock_theta_b in enumerate(dock_pos):
                            pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                        (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                        (R-delta_l)*sin(theta)]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],pi/2-j_a*pi/2+pi*(m%2)),
                                rotation_matrix([1,0,0],theta),
                                rotation_matrix([0,0,1],-dock_theta_b),
                                rotation_matrix(pos_b,screw_theta+pi),
                            ])
                            self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, i))
                        
                    pos_counter += 1
            
            # tip of stick a
            pos_a = [0, 0, -(l-stick_ball_l)/2]
            for screw_theta in screw_thetas:
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([1,0,0],pi/2-theta),
                    rotation_matrix([0,0,1],screw_theta+pi-pi/3)
                ])
                for i, dock_theta_b in enumerate(dock_pos):
                    pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                (R-delta_l)*sin(theta)]
                    rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],-dock_theta_b)
                            ])
                    self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, i))

            pos_counter += 1
        
        elif conn_type == "22":
            # stick <- stick
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            pos_counter = 0

            # tip of stick a
            pos_a = [0, 0, (l-stick_ball_l)/2]
            for screw_theta in screw_thetas:
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([0,0,1],screw_theta)
                ])

                pos_counter_b = 0
                pos_b = [0, 0, (l-stick_ball_l)/2]
                rotate_b = rotation_matrix_sequence([
                    rotation_matrix([1,0,0],pi),
                    rotation_matrix([0,0,1],pi),
                    rotation_matrix([0,0,1],pi/6),
                ])
                self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                pos_counter_b += 1

                for m in range(3):
                    for j_b in side_j_list[m]:
                        pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), side_pos_list[m]]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,1,0],-pi/2),
                            rotation_matrix([0,0,1],j_b*pi/2+pi/12)
                        ])
                        # rotate_b = np.eye(3)
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                        pos_counter_b += 1
                
                pos_b = [0, 0, -(l-stick_ball_l)/2]
                rotate_b = rotation_matrix_sequence([
                    rotation_matrix([0,0,1],-pi/6)
                ])
                self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))

            pos_counter += 1

            # side of stick a
            for m in range(3):
                for j_a in side_j_list[m]:
                    pos_a = [r*cos(j_a*pi/2+pi/12), r*sin(j_a*pi/2+pi/12), side_pos_list[m]]
                    rotate_a = rotation_matrix_sequence([
                        rotation_matrix([1,0,0],pi*((m+1)%2)),
                        rotation_matrix([0,0,1],pi/12+pi*(m%2))
                    ])
                    for screw_theta in screw_thetas:
                        pos_counter_b = 0
                        pos_b = [0, 0, (l-stick_ball_l)/2]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([-sin(j_a*pi/2),cos(j_a*pi/2),0],-pi/2),
                            rotation_matrix([0,0,1],-j_a*pi/2+screw_theta),
                            rotation_matrix([0,0,1],pi/6)
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                        pos_counter_b += 1
                        
                        if screw_theta != 0:
                            for n in range(3):
                                for j_b in side_j_list[n]:
                                    pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), side_pos_list[n]]
                                    rotate_b = rotation_matrix_sequence([
                                        rotation_matrix([0,1,0],pi*(n%2)),
                                        rotation_matrix([0,0,1],j_b*pi/2-j_a*pi/2),
                                        rotation_matrix([cos(j_b*pi/2), sin(j_b*pi/2), 0], screw_theta),
                                        rotation_matrix([0,0,1],pi/12+pi*(n%2)*(m%2))
                                    ])
                                    self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                                    pos_counter_b += 1
                        else:
                            pos_counter_b += 6
                        
                        pos_b = [0, 0, -(l-stick_ball_l)/2]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([-sin(j_a*pi/2),cos(j_a*pi/2),0],-pi/2),
                            rotation_matrix([0,0,1],-j_a*pi/2+screw_theta+pi/6),
                            rotation_matrix([1,0,0],pi)
                        ])
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                        
                    pos_counter += 1
            
            
            # tip of stick a
            pos_a = [0, 0, -(l-stick_ball_l)/2]
            for screw_theta in screw_thetas:
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([0,0,1],screw_theta-pi/6)
                ])
                pos_counter_b = 0
                pos_b = [0, 0, (l-stick_ball_l)/2]
                rotate_b = np.eye(3)
                self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                pos_counter_b += 1

                for m in range(3):
                    for j_b in side_j_list[m]:
                        pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), side_pos_list[m]]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,1,0],pi/2+pi*((m+1)%2)),
                            rotation_matrix([0,0,1],j_b*pi/2+pi/12+pi*((m+1)%2)),
                        ])
                        # rotate_b = np.eye(3)
                        self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                        pos_counter_b += 1
                
                pos_b = [0, 0, -(l-stick_ball_l)/2]
                rotate_b = rotation_matrix_sequence([
                    rotation_matrix([1,0,0],pi)
                ])
                self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))

            pos_counter += 1





        # elif conn_type == "02" or conn_type == "12":
        #     # ball <- stick
        #     pos_a = [0,0,0]
        #     pos_b = [0, (R-delta_l)*cos(theta), (R-delta_l)*sin(theta)]
        

    def __getitem__(self, index):
        # Return the element at the specified index
        return self.dock_list[index]


class RobotDesigner(object):

    '''
    module: a ball or a stick
    parent_type: 0 for ball_up, 1 for ball_bottom, 2 for stick
    connection_type: 00, (01), 02, 10, (11), 12, 20, (21), 22
    position: 0,1,2 (hemisphere); 0,1,2,3,4,5,6,7 (stick)
    (position should be before connection_type so that they can be removed when they are occupied)
    available_ports: 
        {   
            parent_id: 
                {
                    position: 
                        {
                            connection_type: [(position, rotation), (position, rotation), ...],
                            ...
                        },
                    ...
                },
            ...
        }
    '''

    def __init__(self, init_pipeline=None, robot_cfg=None, mesh_dict=None, allow_custom_joint=False, allow_overlapping=False, color=None, broken_mask=None):
        self.robot_cfg = robot_cfg
        self.robot_cfg["l"]  = robot_cfg["l_"] - (robot_cfg["R"] - np.sqrt(robot_cfg["R"]**2 - robot_cfg["r"]**2))
        # self.mesh_mode = mesh_mode
        self.mesh_dict = mesh_dict
        self.allow_overlapping = allow_overlapping
        self.lite = True
        if init_pipeline is not None:
            self.reset()
            for step in np.reshape(init_pipeline, (-1, 4)):
                self.step(step)
        self.connecting = ConnectAsym if self.lite else Connect


    def _get_quat(self, pos):
        target_vec = -np.array(pos)
        quat = quaternion_from_vectors([0, 0, 1], target_vec)
        return quat
    


    def _add_ports(self, parent_id):

        parent_type = self.parent_id_to_type[parent_id]
        # all the module types that can be connected to this part
        connection_types = [i for i in self.connection_types if i[0] == parent_type]

        for connection_type in connection_types:
            # if connection_type == "00" or connection_type == "10":
                # ball <- ball, three docking positions

            docks = self.connecting(connection_type, self.robot_cfg, self.lite)
            for dock in docks:
                position_a = dock.position_a
                position_b = dock.position_b
                T_A_B = calculate_transformation_matrix(dock.pos_a, dock.rotate_a, dock.pos_b, dock.rotate_b)
                pos, quat = matrix_to_pos_quat(T_A_B)
                self.available_ports[parent_id][position_a][position_b][connection_type].append((pos, quat))
                # print(f"Add: {parent_id}/{position}/{connection_type} -> {pos}, {quat}")
                # if parent_id == 0 and position == 0 and connection_type == "00":
                #     print(f"Add: {parent_id}/{position}/{connection_type} -> {pos}, {quat}")
                


    def _add_a_ball(self, parent_id=None, position_id_a=None, position_id_b=None, rotation_id=None, passive=False):
        # print(f"_add_a_ball({parent_id}, {position_id_a}, {position_id_b}, {rotation_id})")
        # print(f"Add a ball: {parent_id}/{position_id_a}/{position_id_b}/{rotation_id}")
        if parent_id is None:
            # init a NEW ball
            node_ids = self.builder.add_simple_ball(color=self.color, range=None, passive=passive)
            # if self.mesh_mode == "default":
            #     node_ids = self.builder.add_simple_ball(color=self.color, range=None)
            # elif self.mesh_mode == "pretty":
            #     node_ids = self.builder.add_simple_ball(color=self.color, range=None)
            # elif self.mesh_mode == "all_meshes":
            #     node_ids = self.builder.add_simple_ball(color=self.color, range=None, model_motor=False)
            # elif self.mesh_mode in ["all_primitives", "draft"]:
            #     node_ids = self.builder.add_draft_ball(color=self.color, range=None, model_motor=not self.mesh_mode=="draft")
            # else:
            #     raise ValueError("Invalid mesh mode, ", self.mesh_mode)

            types = ["0", "1"]
            # for i, t in zip(node_ids, types):
            #     self.parent_id_to_type[i] = t
            #     self._add_ports(i)
        else:
            assert rotation_id is not None, "Ball does not have rotation"

            parent_type = self.parent_id_to_type[parent_id]
            connection_type = parent_type + "0" # assume that children use the upper half to connect to the parent
            ports = self.available_ports[parent_id][position_id_a][position_id_b][connection_type]
            port = ports[rotation_id] if rotation_id is not None else np.random.choice(ports)
            # print(f"Add a ball: {parent_id}/{position_id}/{rotation_id} -> {port[0]}, {port[1]}")
            node_ids = self.builder.add_simple_ball(parent_id, port[0], port[1], range=None, color=self.color, passive=passive)
            # if self.mesh_mode == "default":
            #     node_ids = self.builder.add_simple_ball(parent_id, port[0], port[1], range=None, color=self.color)
            # elif self.mesh_mode == "pretty":
            #     node_ids = self.builder.add_simple_ball(parent_id, port[0], port[1], range=None, color=self.color)
            # elif self.mesh_mode == "all_meshes":
            #     node_ids = self.builder.add_simple_ball(parent_id, port[0], port[1], range=None, color=self.color, model_motor=False)
            # elif self.mesh_mode in ["all_primitives", "draft"]:
            #     node_ids = self.builder.add_draft_ball(parent_id, port[0], port[1], range=None, color=self.color, model_motor=not self.mesh_mode=="draft")
            # else:
            #     raise ValueError("Invalid mesh mode, ", self.mesh_mode)
            # for conn in self.available_ports[parent_id][position_id].keys():
            #     self.available_ports[parent_id][position_id][conn] = []
            if not self.allow_overlapping:
                del self.available_ports[parent_id][position_id_a]


        types = ["0", "1"]
        for i, t in zip(node_ids, types):
            self.parent_id_to_type[i] = t
            self._add_ports(i)
            self.node_ids.append(i)
        if not parent_id is None and not self.allow_overlapping:
            # Delete the child's port if this is not the first module
            del self.available_ports[node_ids[0]][position_id_b]
        

        return node_ids

    
    def _add_a_stick(self, parent_id, position_id_a=None, position_id_b=None, rotation_id=None, broken=0, pos_offset=0):
        # print(f"_add_a_stick({parent_id}, {position_id_a}, {position_id_b}, {rotation_id}, {broken})")
        # print(f"Add a stick: {parent_id}/{position_id_a}/{position_id_b}/{rotation_id}")
        parent_type = self.parent_id_to_type[parent_id]
        connection_type = parent_type + "2"
        ports = self.available_ports[parent_id][position_id_a][position_id_b][connection_type]
        port = ports[rotation_id] # if rotation_id is not None else np.random.choice(ports)
        
        node_id = self.builder.add_independent_stick(parent_id, pos=port[0], quat=port[1], color=self.color, broken=broken, pos_offset=pos_offset)

        # if self.mesh_mode in ["default", "all_primitives"]:
        #     node_id = self.builder.add_independent_stick(parent_id, radius=self.robot_cfg["r"], length=self.robot_cfg["l_"], pos=port[0], quat=port[1], color=self.color, type="cylinder")
        # elif self.mesh_mode == "draft":
        #     node_id = self.builder.add_independent_stick(parent_id, radius=self.robot_cfg["r"], length=self.robot_cfg["l_"], pos=port[0], quat=port[1], color=self.color, type="capsule")
        # elif self.mesh_mode == "all_meshes":
        #     node_id = self.builder.add_independent_stick(parent_id, pos=port[0], quat=port[1], color=self.color, type="mesh", broken=broken)
        # elif self.mesh_mode == "pretty":
        #     node_id = self.builder.add_independent_stick(parent_id, pos=port[0], quat=port[1], color=self.color, type="mesh", broken=broken)

        if broken:
            print("broken!")

        # for conn in self.available_ports[parent_id][position_id].keys():
        #     self.available_ports[parent_id][position_id][conn] = []

        self.parent_id_to_type[node_id] = "2"
        self._add_ports(node_id)
        self.node_ids.append(node_id)

        if not self.allow_overlapping:
            del self.available_ports[parent_id][position_id_a]
            del self.available_ports[node_id][position_id_b]
            # print(f"Delete: self.available_ports{node_id}/{position_id_b}")

        return [node_id]
    




    
    def add_dummy_node(self, type):
        '''
        type: 0 for ball, 1 for stick
        '''
        n_nodes = 2 if type == 0 else 1
        for _ in range(n_nodes):
            self.builder.leg_nodes.append(None)
            self.node_ids.append(len(self.builder.leg_nodes)-1)
        return self.node_ids[-n_nodes:]
    
    def get_pos_id_list(self, module, parent_id):
        parent_type = self.parent_id_to_type[parent_id]
        child_type = "0" if module == 0 else "2"
        connection_type = parent_type + child_type
        self.pos_list = []
        for position_id, value in self.available_ports[parent_id].items():
            if connection_type in value:
                self.pos_list.append(position_id)

        return self.pos_list
    
    def get_rotation_id_list(self, module, parent_id, position_id_a, position_id_b):
        parent_type = self.parent_id_to_type[parent_id]
        child_type = "0" if module == 0 else "2"
        connection_type = parent_type + child_type
        return list(range(len(self.available_ports[parent_id][position_id_a][position_id_b][connection_type])))
    

    def reset(self):
        # self.builder = RobotBuilder(parts=["left_m3lite.obj", "right_m3lite.obj"], robot_cfg=self.robot_cfg)
        self.builder = RobotBuilder(mesh_dict=self.mesh_dict, robot_cfg=self.robot_cfg)

        self.connection_types = ["00", "02", "10", "12", "20", "22"] 
        self.color = 3

        self.available_ports = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: defaultdict(list)))))
        self.parent_id_to_type = {} # {parent_id: parent_type}

        self.node_ids = []
        self.robot_properties = {}

        self.node_ids = []
        self._add_a_ball()


    def step(self, pipeline):
        module, parent, pos, rotation = pipeline
        if module == 0:
            # add a ball
            return self._add_a_ball(parent, pos, rotation)
        elif module == 1:
            # add a stick
            return self._add_a_stick(parent, pos, rotation)
        else:
            raise ValueError(f"Invalid module type: {module}")
        
    def step_sequence(self, pipelines):
        for pipeline in np.reshape(pipelines, (-1, 4)):
            self.step(pipeline)
        
    def get_xml(self):
        # Get xml string
        return self.builder.get_xml()

    def compile(self, 
                render=False, 
                self_collision_test=True, 
                stable_state_test=True, 
                movability_test=False, 
                joint_level_optimization=False,
                config_dict=None):
        '''
        Compile the xml in mujoco and get the properties of the robot
        Note that the robot_properties may be stochastic
        '''

        xml = self.builder.get_xml(fix_file_path=True)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        if render:
            viewer = mujoco.viewer.launch_passive(m, d)
            viewer.__enter__()
        else:
            viewer = None

        self.robot_properties["num_joints"] = m.nu

        # Self collision test
        if self_collision_test:
            self_collision_check(m, d, self.robot_properties, viewer=viewer)

        # Stable state test
        if stable_state_test:
            stable_state_check(m, d, self.robot_properties, self.robot_cfg["theta"], viewer=viewer)


        ######################################################
        # Movability test
        ######################################################
        if movability_test:
            movability_check(m, d, self.robot_properties, viewer=viewer, config_dict=config_dict, 
                            stable_pos_list=self.robot_properties["stable_pos"] if "stable_pos" in self.robot_properties else None,
                            stable_quat_list=self.robot_properties["stable_quat"] if "stable_quat" in self.robot_properties else None)


        if "ave_speed" in self.robot_properties:
            print(f"Average speed: {self.robot_properties['ave_speed']}")
        if "self_collision_rate" in self.robot_properties:
            print(f"Self collision: {self.robot_properties['self_collision_rate']}")
        if "stable_height" in self.robot_properties:
            print(f"Stable height: {self.robot_properties['stable_height']}")

        if render:    
            viewer.__exit__()

    def set_terrain(self, terrains, terrain_params=None):
        if not is_list_like(terrains):
            terrains = [terrains]
        if "bumpy" in terrains:
            self.builder.add_bumps(h_func=lambda x : 0.04 + 0.02*x if 0.04 + 0.02*x < 0.1 else 0.1)
        if "walls" in terrains:
            transparent = terrain_params["transparent"] if terrain_params is not None and "transparent" in terrain_params else True
            angle = terrain_params["angle"] if terrain_params is not None and "angle" in terrain_params else False
            self.builder.add_walls(transparent=transparent, angle=angle)
        if "random_bumpy" in terrains:
            terrain_params = terrain_params if terrain_params is not None else {"num_bumps": 200, "height_range": (0.01, 0.05), "width_range": (0.1, 0.1)}
            for _ in range(terrain_params["num_bumps"]):
                pos = np.random.uniform(-5, 5, 2)
                angle = np.random.uniform(0, 360)
                height = np.random.uniform(*terrain_params["height_range"])
                width = np.random.uniform(*terrain_params["width_range"])
                self.builder.add_minibump(pos=pos, angle=angle, height=height, width=width, length=1)
        if "stairs" in terrains:
            num_steps = 20 if terrain_params is None or "num_steps" not in terrain_params else terrain_params["num_steps"]
            step_height = 0.1 if terrain_params is None or "step_height" not in terrain_params else terrain_params["step_height"]
            width = 0.5 if terrain_params is None or "width" not in terrain_params else terrain_params["width"]
            reserve_area = 2 if terrain_params is None or "reserve_area" not in terrain_params else terrain_params["reserve_area"]
            for i in range(reserve_area, num_steps+reserve_area):
                wall_center = reserve_area+width*(i-reserve_area)+width/2
                pos = [[wall_center, 0], [-wall_center, 0], [0, wall_center], [0, -wall_center]]
                for p in pos:
                    height = step_height*(i-reserve_area+1)
                    length = wall_center*2
                    angle = 0
                    if p[0] == 0:
                        length += width
                        angle = 90
                    else:
                        length -= width
                    assert length > 0, f"Length: {length}"
                    assert width > 0, f"Width: {width}"
                    assert height > 0, f"Height: {height}"
                    self.builder.add_minibump(pos=p, angle=angle, height=height, width=width, length=length)

        if "ministairs" in terrains:
            num_steps = 20
            step_height = 0.1
            width = 0.5
            reserve_area = 2
            num_substeps = 2
            for i in range(num_steps):
                wall_center = reserve_area+width*i+width/2
                flag = (i+1)%(num_substeps*2+1)
                if flag > num_substeps:
                    flag = num_substeps*2-flag
                height = round(step_height*flag,2)
                # pdb.set_trace()
                print(f"Height: {height}")
                if height == 0:
                    continue
                pos = [[wall_center, 0], [-wall_center, 0], [0, wall_center], [0, -wall_center]]
                for p in pos:
                    length = wall_center*2
                    angle = 0
                    if p[0] == 0:
                        length += width
                        angle = 90
                    else:
                        length -= width
                    assert length > 0, f"Length: {length}"
                    assert width > 0, f"Width: {width}"
                    assert height > 0, f"Height: {height}"
                    self.builder.add_minibump(pos=p, angle=angle, height=height, width=width, length=length)

        if "dune" in terrains:
            radius_x = 20 if terrain_params is None or "radius_x" not in terrain_params else terrain_params["radius_x"]
            radius_y = 20 if terrain_params is None or "radius_y" not in terrain_params else terrain_params["radius_y"]
            elevation_z = 0.5 if terrain_params is None or "elevation_z" not in terrain_params else terrain_params["elevation_z"]
            base_z = 0.01 if terrain_params is None or "base_z" not in terrain_params else terrain_params["base_z"]
            self.builder.set_hfield(file="dune.png",radius_x=radius_x, radius_y=radius_y, elevation_z=elevation_z, base_z=base_z)

        if "slope" in terrains:
            radius_x = 20 if terrain_params is None or "radius_x" not in terrain_params else terrain_params["radius_x"]
            radius_y = 20 if terrain_params is None or "radius_y" not in terrain_params else terrain_params["radius_y"]
            elevation_z = 0.5 if terrain_params is None or "elevation_z" not in terrain_params else terrain_params["elevation_z"]
            base_z = 0.01 if terrain_params is None or "base_z" not in terrain_params else terrain_params["base_z"]
            self.builder.set_hfield(file="wave20.png",radius_x=radius_x, radius_y=radius_y, elevation_z=elevation_z, base_z=base_z)

        if "hfield" in terrains:
            radius_x = 20 if terrain_params is None or "radius_x" not in terrain_params else terrain_params["radius_x"]
            radius_y = 20 if terrain_params is None or "radius_y" not in terrain_params else terrain_params["radius_y"]
            elevation_z = 0.5 if terrain_params is None or "elevation_z" not in terrain_params else terrain_params["elevation_z"]
            base_z = 0.01 if terrain_params is None or "base_z" not in terrain_params else terrain_params["base_z"]
            hfield = terrain_params["hfield"]
            self.builder.set_hfield(file=hfield,radius_x=radius_x, radius_y=radius_y, elevation_z=elevation_z, base_z=base_z)

    def wear_socks(self, node_ids, color=(1,1,1), radius=0.04, length=0.08, stick_length=0.235, thickness=0.01):
        for node_id in node_ids:
            self.builder.add_sock(node_id, radius=radius, length=length, stick_length=stick_length, color=color, thickness=thickness)

    def change_color(self, colors):
        for node_id, color in colors.items():
            self.builder.change_color(node_id, color=color)

    def save(self, save_dir, fix=True, pos=(0, 0, 0.2), quat=[1, 0, 0, 0], joint_pos=None, render=True):
        os.makedirs(save_dir, exist_ok=True)
        f_xml = os.path.join(save_dir, "robot.xml")
        f = self.builder.save(f_xml)

        f_yaml = os.path.join(save_dir, "properties.yaml")
        with open(f_yaml, 'w') as yaml_file:
            yaml.dump(numpy_to_native(self.robot_properties), yaml_file, default_flow_style=False)

        if render:
            view(f, fix, pos=pos, quat=quat, vis_contact=True, joint_pos=joint_pos)

        return f, f_yaml


    @property
    def leg_nodes(self):
        return self.builder.leg_nodes
    

def general_usage_test():

    robot_designer = RobotDesigner()
    robot_designer.reset()
    pipelines = [[1, 0, 2, 10], [1, 0, 1, 7], [1, 2, 2, 2]]
    for pipeline in pipelines:
        module = pipeline[0] # what's the next module
        print("-> module: ", module)
        parent = pipeline[1] # which part to connect to
        # print("-> parent: ", parent)
        pos_list = robot_designer.get_pos_id_list(module, parent)
        if not pos_list:
            print("pos_list is empty")
            break
        print("pos choose from: ", pos_list)
        pos = pipeline[2] # where to connect
        orientation_list = robot_designer.get_rotation_id_list(module, parent, pos)
        print("orientation choose from: ", orientation_list)
        orientation = pipeline[3] # how to connect
        # print("-> pos: ", pos, "orientation: ", orientation)
        pipe = [module, parent, pos, orientation]
        pipelines.append(pipe)
        robot_designer.step(pipe)
        print(f"Step: {pipe}")

    # pipelines = [[1, 1, 0, 12], [0, 1, 1, 1], [1, 0, 2, 13]]
    # pipelines = [[0,1,2,2], [1, 0, 2, 13], [1, 3, 0, 14]]
    # for p in pipelines:
    #     robot_designer.step(p)


    robot_designer.compile(render=False)

    file = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", "factory")
    for p in pipelines:
        p_dir = "-".join([str(i) for i in p])
        file = os.path.join(file, p_dir)
    pos = robot_designer.robot_properties["stable_pos"][0]
    quat = robot_designer.robot_properties["stable_quat"][0]
    f = robot_designer.save(file, fix=False, pos=pos, quat=quat, joint_pos=None)

    
def compile_test():
    robot_designer = RobotDesigner()
    robot_designer.reset()
    pipelines = [[1, 0, 2, 10], [1, 0, 1, 7], [1, 2, 2, 2]]
    for design_step in pipelines:
        robot_designer.step(design_step)
        print(f"Step: {design_step}")
    robot_designer.compile(render=True, self_collision_test=False, stable_state_test=True, movability_test=True,
                           config_dict={"init_pos": 0, "init_quat": 0})
    
def test_terrain():
    pipeline = [1, 0, 2, 10, 1, 0, 1, 7, 1, 2, 2, 2]
    robot_designer = RobotDesigner(pipeline)
    robot_designer.set_terrain("bumpy")
    robot_designer.save("test", fix=False, pos=(0, 0, 0.2), quat=[1, 0, 0, 0], joint_pos=None, render=True)

def test_draft_ball():
    pipeline = [1, 0, 2, 10, 1, 1, 1, 7]
    robot_designer = RobotDesigner(pipeline, mesh_mode="draft")
    robot_designer.set_terrain("bumpy")
    robot_designer.save("test", fix=False, pos=(0, 0, 0.2), quat=[1, 0, 0, 0], joint_pos=None, render=True)

def debug():
    ROBOT_CFG_AIR1S = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0, # 0,
        "stick_ball_l": 0.005,# 0, #-0.1, # ,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass": 0.1734, #0.154,
        "top_hemi_mass": 0.1153, 
        "battery_mass": 0.122,
        "motor_mass": 0.317,
        "bottom_hemi_mass": 0.1623, #0.097, 
        "pcb_mass": 0.1
    }
    MESH_DICT_FINE = {
                "up": "top_lid.obj",
                "bottom": "bottom_lid.obj",
                "stick": "leg4.4.obj",
                "battery": "battery.obj",
                "pcb": "pcb.obj",
                "motor": "motor.obj"
            }
    # robot_cfg['a'] = robot_cfg['l_'] / 4
    robot_designer = RobotDesigner(robot_cfg=ROBOT_CFG_AIR1S, mesh_dict=MESH_DICT_FINE, allow_overlapping=True)
    robot_designer.reset()
    # robot_designer._add_a_stick(0, 0, 0, 0)
    # robot_designer._add_a_stick(1, 0, 0, 0)

    ## dog
    # robot_designer._add_a_stick(0, 0, 0, 0)
    # robot_designer._add_a_stick(1, 0, 0, 0)
    # robot_designer._add_a_stick(2,3,7,0)
    # robot_designer._add_a_stick(2,5,7,0)
    # robot_designer._add_a_stick(3,4,7,0)
    # robot_designer._add_a_stick(3,6,7,0)
    # robot_designer._add_a_ball(4, 0, 0, 0)
    # robot_designer._add_a_ball(5, 0, 0, 0)
    # robot_designer._add_a_ball(6, 0, 0, 0)
    # robot_designer._add_a_ball(7, 0, 0, 0)

     ## cat
    # robot_designer._add_a_stick(0, 0, 0, 0)
    # robot_designer._add_a_stick(1, 0, 0, 0)

    # robot_designer._add_a_stick(2, 1, 7, 0)
    # robot_designer._add_a_ball(4, 0, 0, 0)
    # robot_designer._add_a_stick(6, 0, 0, 0) #7




    ## DEBUG
    robot_designer._add_a_stick(0, 0, 0, 0, 0)
    robot_designer._add_a_stick(1, 0, 0, 0, 0)
    
    # robot_designer._add_a_ball(3, 4, 1, 1)
    # robot_designer._add_a_stick(4, 0, 0, 0, 0)
    # robot_designer._add_a_stick(5, 0, 0, 0, 0)

    # robot_designer._add_a_stick(3, 5, 3, 1, 0)
    for i in range(1, 7):
        for j in [0]:
            robot_designer._add_a_stick(3, 3, i, j, 0)
            # robot_designer._add_a_stick(2, 3, i, 1, j)
    # robot_designer._add_a_ball(4, 0, 0, 0)
    # robot_designer._add_a_stick(6, 0, 0, 0, 0)


    # robot_designer._add_a_stick(3,7,3,1)
    # robot_designer._add_a_stick(3,7,3,2)
    # robot_designer._add_a_ball(4, 0, 0, 0)
    # robot_designer._add_a_stick(6, 0, 0, 0)

    # robot_designer._add_a_stick(3,5,3,0)
    # robot_designer._add_a_stick(3,5,4,0)
    # robot_designer._add_a_stick(3,5,6,0)
    # robot_designer._add_a_stick(3,5,5,0)

    # for pa in range(3,7):
    #     for r in range(2):
    #         for pb in range(1,7):
    #             # robot_designer._add_a_ball(2, pa, pb, r)
    #             robot_designer._add_a_stick(2, pa, pb, r)
    # #             robot_designer._add_a_ball(2, pa, pb, r)
    #         # robot_designer._add_a_stick(2, pa, 7, r)
    #         robot_designer._add_a_stick(0, pa, 1, r)
            # robot_designer._add_a_stick(1, pa, 3, r)
            # robot_designer._add_a_stick(1, pa, 6, r)
            # robot_designer._add_a_ball(2, 0, pa, r)
            # robot_designer._add_a_ball(2, 7, pa, r)
            # robot_designer._add_a_ball(2, 1, pa, r)
            # robot_designer._add_a_ball(2, 2, pa, r)
            # robot_designer._add_a_ball(2, 3, pa, r)
            # robot_designer._add_a_ball(2, 4, pa, r)
            # robot_designer._add_a_ball(2, 5, pa, r)
            # robot_designer._add_a_ball(2, 6, pa, r)
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(tmpdir, fix=True, pos=(0, 0, 1), quat=[1, 0, 0, 0], joint_pos=None, render=True)

if __name__ == "__main__":
    debug()