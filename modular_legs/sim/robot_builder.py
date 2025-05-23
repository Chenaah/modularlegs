import os
import pdb
from lxml import etree
import numpy as np
from numpy import sin, cos
from modular_legs import LEG_ROOT_DIR
from modular_legs.utils.math import quat_rotate, quaternion_multiply2, construct_quaternion
from modular_legs.utils.model import create_color_spectrum, fix_model_file_path, vec2string, lighten_color, quaternion_from_vectors
from modular_legs.utils.others import is_list_like
from modular_legs.utils.visualization import view

class Port():
    def __init__(self, pos, body_pos=None, leg_node=None, leg_vec=None, module_idx=0):
        self.body_pos = body_pos
        self.pos = pos
        self.leg_node = leg_node
        self.leg_vec = leg_vec
        self.module_idx = module_idx

class RobotBuilder():

    def __init__(self, template=None, terrain="flat", mesh_dict=None, robot_cfg=None):
        # Load the template
        if template is None:
            template = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", 'template.xml')
        self.terrain = terrain
        self.mesh_dict = mesh_dict
        self.robot_cfg = robot_cfg
        parser = etree.XMLParser(remove_blank_text=True)
        self.tree = etree.parse(template, parser)
        self.root = self.tree.getroot()
        self.worldbody = self.root.findall("./worldbody")[0]
        self.assets = self.root.findall("./asset")[0]
        self.actuators = self.root.findall("./actuator")[0]
        self.contact = etree.SubElement(self.root, "contact")
        self.sensors = etree.SubElement(self.root, "sensor")


        # Define the robot parameters
        # TODO: from cfg
        self.theta = 0.610865
        self.l = 0.3
        self.initial_pos = np.array([0, 0, 0])
        self.R = 0.06
        self.r = 0.03
        self.d = 0
        self.stick_mass = 0.26
        self.top_hemi_mass = 0.7
        self.bottom_hemi_mass = 0.287
        self.battery_mass = 0
        self.motor_mass = 0.3
        self.pcb_mass = 0

        self.l_ = -1
        self.delta_l = -1
        self.stick_ball_l = -1
        self.a = -1

        if robot_cfg is not None:
            assert all(hasattr(self, key) for key in robot_cfg), "The robot_cfg should be a dictionary with the keys: theta, l, initial_pos, R, r, d, stick_mass, top_hemi_mass, bottom_hemi_mass, motor_mass"
            [setattr(self, key, robot_cfg[key]) for key in robot_cfg]

        self.mass = {"stick": self.stick_mass, 
                     "top_hemi": self.top_hemi_mass, 
                     "bottom_hemi": self.bottom_hemi_mass, 
                     "motor": self.motor_mass,
                     "pcb": self.pcb_mass}

        # Define the leg vectors
        self.lleg_vec = np.array([0, self.l*np.cos(self.theta), self.l*np.sin(self.theta)])
        self.rleg_vec = np.array([0, -self.l*np.cos(self.theta), -self.l*np.sin(self.theta)])

        self.colors = create_color_spectrum(num_colors=6)

        # Data tracking
        self.n_joint = 0
        self.ports = []
        self.leg_nodes = []
        self.idx_counter = 0
        self.passive_idx_counter = 0
        self.sock_idx_counter = 0
        self.obstacle_idx_counter = 0
        self.imported_mesh = [] # list of mesh file names
        self.torsos = []

        self._add_floor()

        assert self.mesh_dict is not None, "The mesh_dict should be provided"
        self._import_mesh()

    def _add_floor(self):
        # Add the floor
        assert self.terrain == "flat" # Use the "set_terrain" method to change the terrain
        self.floor_elem = etree.SubElement(self.worldbody, "geom", name="floor", pos="0 0 0", size="40 40 40", type="plane", material="matplane", conaffinity="1", condim="6", friction="1.0 .0 .0", priority="1")
        # elif self.terrain == "rugged":
        #     etree.SubElement(self.worldbody, "geom", name="floor", pos="0 0 0", type="hfield", material="hfield", conaffinity="1", condim="3", hfield="rugged")
        #     etree.SubElement(self.assets, "hfield", name="rugged", size="18 20 0.3 0.1", file=os.path.join("..", "parts", "rugged.png"))
        # TODO: move assets out of template
        # TODO: generate hfield files here


    def _build_torsor(self):
        torso = etree.SubElement(self.worldbody, "body", name=f"torso{len(self.torsos)}", pos=vec2string(self.initial_pos)) # invisible torsor
        etree.SubElement(torso, "freejoint", name=f"root{len(self.torsos)}")
        self.torsos.append(torso)
        # etree.SubElement(self.torso, "joint", name="root", type="free", stiffness="0", damping="0.05", armature="0")


    def _import_mesh(self):
        # id = len(self.imported_mesh)
        # mesh_names = []
        # for i in range(len(parts)):
        #     etree.SubElement(self.assets, "mesh", file=parts[i], name=f"mesh{id+i}", scale="1 1 1")
        #     self.imported_mesh.append(parts[i])
        #     mesh_names.append(f"mesh{id+i}")
        for mesh_name, file in self.mesh_dict.items():
            if file.endswith(".obj") or file.endswith(".stl"):
                etree.SubElement(self.assets, "mesh", file=file, name=mesh_name, scale="1 1 1")
                self.imported_mesh.append(file)


    def add_module(self, port_id=None, pos=None, quat=None, range="-90 90"):

        # Given a port, the position of this module is strongly constrained;
        # And given the port and the position, the orientation of this module is already determined.
        # After designing the real robot docking mechanism, we can use other way to represent the pos.

        if port_id is None:
            if self.idx_counter == 0:
                parent = self.torso
            else:
                raise ValueError("The port_id is required for the module")
        else:
            assert port_id < len(self.ports), "The port_id is out of range"
            port = self.ports[port_id]
            parent = port.leg_node

        if pos is None:
            if self.idx_counter == 0:
                pos = vec2string(self.initial_pos)
            else:
                raise ValueError("The pos is required for the module")

        if quat is None:
            if self.idx_counter == 0:
                quat = vec2string([1, 0, 0, 0])
            else:
                quat = quaternion_from_vectors(self.lleg_vec,  port.pos - pos) # always align with the left leg ?

        if not isinstance(pos, str):
            pos = vec2string(pos)
        if not isinstance(quat, str):
            quat = vec2string(quat)

        idx = self.idx_counter
        l0  = etree.SubElement(parent, "body", name=f"l{idx}", pos=pos, quat=quat)
        color = self.colors[idx]
        # print("color ", color)
        etree.SubElement(l0, "geom", type="mesh", mesh="mesh0", rgba=f"{color[0]} {color[1]} {color[2]} 1")

        r0  = etree.SubElement(parent, "body", name=f"r{idx}", pos=pos, quat=quat)
        if range is not None:
            etree.SubElement(r0, "joint", axis="0 0 1", name=f"joint{idx}", pos="0 0 0", range=range, type="hinge")
        else:
            etree.SubElement(r0, "joint", axis="0 0 1", name=f"joint{idx}", pos="0 0 0", type="hinge")
        color_r = lighten_color(color, 0.5) # [1,1,1]
        etree.SubElement(r0, "geom", type="mesh", mesh="mesh1", rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 1")

        etree.SubElement(self.actuators, "position", ctrlrange="-3.14 3.14", joint=f"joint{idx}", kp="20", kv="0.5", forcerange="-12 12")

        pos = np.fromstring(pos, dtype=float, sep=' ')
        quat = np.fromstring(quat, dtype=float, sep=' ')

        lfoot_pos = self.lleg_vec
        rfoot_pos = self.rleg_vec

        # Assume that there are two ports for one module
        self.ports.append(Port(lfoot_pos, body_pos=pos, leg_node=l0, leg_vec=self.lleg_vec, module_idx=idx))
        self.ports.append(Port(rfoot_pos, body_pos=pos, leg_node=r0, leg_vec=self.rleg_vec, module_idx=idx))
        self.n_joint += 1
        self.idx_counter += 1


    def cal_tip_side_port(self, a, alpha, direction, screw):
        theta = self.theta
        AO1 = self.l+self.R+(self.r-self.d)
        AO = a + self.R
        BO1 = self.R + self.l - AO*np.tan(theta)
        OH = BO1*np.sin(theta) + AO/np.cos(theta)
        O1H = BO1*np.cos(theta)

        if direction == 0:
            # Left Tip -> Left leg
            pos = [-(AO1)*np.sin(alpha), AO*np.cos(theta)+(AO*np.cos(theta)-OH)*np.cos(alpha), AO*np.sin(theta)+(AO*np.sin(theta)+O1H)*np.cos(alpha)]
            target_vec = [(AO1)*np.sin(alpha), (OH-AO*np.cos(theta))*np.cos(alpha), -(O1H+AO*np.sin(theta))*np.cos(alpha)]
            quat = quaternion_from_vectors([0, np.cos(theta), np.sin(theta)], target_vec)
        elif direction == 1:
            # Left Tip -> Right leg
            pos = [-(AO1)*np.sin(alpha), -AO*np.cos(theta)+(AO*np.cos(theta)-OH)*np.cos(alpha), -AO*np.sin(theta)+(AO*np.sin(theta)+O1H)*np.cos(alpha)]
            target_vec = [(AO1)*np.sin(alpha), (OH-AO*np.cos(theta))*np.cos(alpha), -(O1H+AO*np.sin(theta))*np.cos(alpha)]
            quat = quaternion_from_vectors([0, np.cos(theta), np.sin(theta)], target_vec)

        screw_quat = construct_quaternion(target_vec, screw)
        quat = quaternion_multiply2(quat, screw_quat)

        return pos, quat
    
    def cal_general_port(self, a, b, alpha, beta, direction=0, screw=0):
        r = self.r
        R = self.R
        theta = self.theta
        if direction == 1:
            pos_ = [2*r*sin(alpha)-(b+R)*sin(beta)*cos(alpha), -a-R-(b+R)*cos(beta), 2*r*cos(alpha)+(b+R)*sin(beta)*sin(alpha)]
            rotate = construct_quaternion([1.,0,0], theta, "xyzw")
            pos = quat_rotate(rotate, pos_)

            target_vec_ = [(b+R)*sin(beta)*cos(alpha), (b+R)*cos(beta), -(b+R)*sin(beta)*sin(alpha)]
            target_vec = quat_rotate(rotate, target_vec_)
            quat = quaternion_from_vectors([0, np.cos(theta), np.sin(theta)], target_vec)

        elif direction == 0:
            pos_ = [2*r*sin(alpha)-(b+R)*sin(beta)*cos(alpha), a+R+(b+R)*cos(beta), 2*r*cos(alpha)+(b+R)*sin(beta)*sin(alpha)]
            rotate = construct_quaternion([1.,0,0], theta, "xyzw")
            pos = quat_rotate(rotate, pos_)

            target_vec_ = [(b+R)*sin(beta)*cos(alpha), -(b+R)*cos(beta), -(b+R)*sin(beta)*sin(alpha)]
            target_vec = quat_rotate(rotate, target_vec_)
            quat = quaternion_from_vectors([0, np.cos(theta), np.sin(theta)], target_vec)

        screw_quat = construct_quaternion(target_vec, screw)
        quat = quaternion_multiply2(quat, screw_quat)

        return pos, quat

    def _convert_quat(self, quat):
        # convert target_vec (3D) to quat
        assert is_list_like(quat), "quat should be a list"
        if not all([is_list_like(i) for i in quat]):
            if len(quat) == 4: # "init_quat should be a list of 4 elements"
                result = np.array(quat)
            elif len(quat) == 3:
                target_vec = quat
                result = quaternion_from_vectors([0, np.cos(self.theta), np.sin(self.theta)], target_vec)
        else:
            # list of list
            result = np.array([1,0,0,0])
            for q in quat:
                if len(q) == 3:
                    target_vec = q
                    q_step = quaternion_from_vectors([0, np.cos(self.theta), np.sin(self.theta)], target_vec)
                elif len(q) == 4:
                    q_step = q
                result = quaternion_multiply2(result, q_step)
        return result

    def add_one_module(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, quat_r=[1,0,0,0], joint_axis=[0,0,1]):
        # New API for testing a realistic robot docking mechanism
        # This is a lower-level API, exposing pos and quat, which can be generated by a port generator
        # If port_id is None, a new torso is created

        if parent_id is None:
            # The first module in the world
            # print("Add the first module")
            self._build_torsor()
            parent = self.torsos[-1]
            pos = self.initial_pos
            quat = [1,0,0,0]
        else:
            assert parent_id < len(self.leg_nodes), "The port_id is out of range"
            assert pos is not None, "The pos is required for the module"
            assert quat is not None, "The quat is required for the module"
            parent = self.leg_nodes[parent_id]

        if color is None:
            color_l = self.colors[self.idx_counter]
            color_r = lighten_color(color_l, 0.5) # [1,1,1]
        elif isinstance(color, (int, float)):
            color_l = self.colors[color]
            color_r = lighten_color(color_l, 0.5)
        else:
            if len(np.array(color).shape) == 1:
                color_l = color
                color_r = lighten_color(color, 0.5)
            else:
                color_l = color[0]
                color_r = color[1]

        quat = self._convert_quat(quat)

        idx = self.idx_counter
        l0  = etree.SubElement(parent, "body", name=f"l{idx}", pos=vec2string(pos), quat=vec2string(quat))
        

        quat = self._convert_quat([quat_r, quat])
        r0  = etree.SubElement(parent, "body", name=f"r{idx}", pos=vec2string(pos), quat=vec2string(quat))
        if range is not None:
            etree.SubElement(r0, "joint", axis=vec2string(joint_axis), name=f"joint{idx}", pos="0 0 0", range=range, type="hinge", armature="0.05", damping="0.2", limited="auto")
        else:
            etree.SubElement(r0, "joint", axis=vec2string(joint_axis), name=f"joint{idx}", pos="0 0 0", type="hinge", armature="0.05", damping="0.2", limited="auto")
        
        # Attatching the mesh
        # Ball
        if self.mesh_dict["up"].endswith(".obj") or self.mesh_dict["up"].endswith(".stl"):
            etree.SubElement(l0, "geom", type="mesh", name=f"left{idx}", mesh="up", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 0.5", mass=str(self.top_hemi_mass), material="metallic", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["up"] == "SPHERE":
            # Draft mode: Use sphere instead of mesh
            radius = self.robot_cfg["R"]
            etree.SubElement(l0, "geom", type="sphere", name=f"left{idx}", size=f"{radius}", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1", mass=f"{self.top_hemi_mass}", friction="1.0 .0 .0", priority="2")
        else:
            raise ValueError("The mesh should be either a .obj file or a SPHERE")
        
        if self.mesh_dict["bottom"].endswith(".obj") or self.mesh_dict["bottom"].endswith(".stl"):
            etree.SubElement(r0, "geom", type="mesh", name=f"right{idx}", mesh="bottom", rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 0.5", mass=str(self.bottom_hemi_mass), material="metallic", euler="180 0 60", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["bottom"] == "SPHERE":
            radius = self.robot_cfg["R"]
            etree.SubElement(r0, "geom", type="sphere", name=f"right{idx}", size=f"{radius}", rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 1", mass=f"{self.bottom_hemi_mass}", friction="1.0 .0 .0", priority="2")
        # Battery, PCB, Motor
        if "battery" in self.mesh_dict:
            if self.mesh_dict["battery"].endswith(".obj") or self.mesh_dict["battery"].endswith(".stl"):
                etree.SubElement(l0, "geom", type="mesh", name=f"battery{idx}", mesh="battery", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 0.5", mass=str(self.battery_mass), material="metallic", contype="10", conaffinity="0")
            elif self.mesh_dict["battery"] == "NONE":
                pass
            else:
                raise ValueError("The mesh should be either a .obj file or NONE")
        if "pcb" in self.mesh_dict:
            if self.mesh_dict["pcb"].endswith(".obj") or self.mesh_dict["pcb"].endswith(".stl"):
                etree.SubElement(l0, "geom", type="mesh", name=f"pcb{idx}", mesh="pcb", rgba=f"0 0 0 0.5", mass=str(self.pcb_mass), material="metallic", contype="10", conaffinity="0")
            elif self.mesh_dict["pcb"] == "NONE":
                pass
            else:
                raise ValueError("The mesh should be either a .obj file or NONE")
        if "motor" in self.mesh_dict:
            if self.mesh_dict["motor"].endswith(".obj") or self.mesh_dict["motor"].endswith(".stl"):
                etree.SubElement(l0, "geom", type="mesh", name=f"motor{idx}", mesh="motor", rgba=f"1 0 0 0.5", mass=str(self.motor_mass), contype="10", conaffinity="0")
            elif self.mesh_dict["motor"] == "CYLINDER":
                etree.SubElement(l0, "geom", type="cylinder", name=f"motor{idx}", pos=vec2string([0,0,-0.015]), quat=vec2string([1,0,0,0 ]), size=f"{0.05} {0.03/2}", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1", mass=str(self.motor_mass), contype="10", conaffinity="0")
            elif self.mesh_dict["motor"].startswith("VIRTUAL"):
                dist = float(self.mesh_dict["motor"].split("_")[1])
                etree.SubElement(l0, "geom", type="sphere", name=f"motor{idx}", size=f"{0.01}", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1", mass=str(self.motor_mass), contype="10", conaffinity="0", pos=vec2string([0,0,-dist]))
            elif self.mesh_dict["motor"] == "NONE":
                pass
            else:
                raise ValueError("The mesh should be either a .obj file or CYLINDER or NONE")


        # Mount the IMU
        etree.SubElement(self.sensors, "framequat", name=f"imu_quat{idx}", objtype="xbody", objname=f"l{idx}")
        etree.SubElement(l0, "site", name=f"imu_site{idx}", pos="0 0 0" ,size="0.01", rgba="0 0 1 1")
        etree.SubElement(self.sensors, "gyro", name=f"imu_gyro{idx}", site=f"imu_site{idx}")
        etree.SubElement(self.sensors, "framelinvel", name=f"imu_globvel{idx}", objtype="xbody", objname=f"l{idx}")
        etree.SubElement(self.sensors, "velocimeter", name=f"imu_vel{idx}", site=f"imu_site{idx}")
        etree.SubElement(self.sensors, "accelerometer", name=f"imu_acc{idx}", site=f"imu_site{idx}")

        # Mount the back IMU
        etree.SubElement(self.sensors, "framequat", name=f"back_imu_quat{idx}", objtype="xbody", objname=f"r{idx}")
        etree.SubElement(r0, "site", name=f"back_imu_site{idx}", pos="0 0 0" ,size="0.01", rgba="0 0 1 1")
        etree.SubElement(self.sensors, "gyro", name=f"back_imu_gyro{idx}", site=f"back_imu_site{idx}")
        # etree.SubElement(self.sensors, "framelinvel", name=f"back_imu_globvel{idx}", objtype="xbody", objname=f"r{idx}")
        etree.SubElement(self.sensors, "velocimeter", name=f"back_imu_vel{idx}", site=f"back_imu_site{idx}")

        etree.SubElement(self.contact, "exclude", body1=f"l{idx}", body2=f"r{idx}")



        if range is not None:
            etree.SubElement(self.actuators, "position", ctrlrange="-3.14 3.14", joint=f"joint{idx}", kp="20", kv="0.5", forcerange="-12 12")
        else:
            etree.SubElement(self.actuators, "position", joint=f"joint{idx}", kp="20", kv="0.5", forcerange="-12 12")


        # Assume that there are two ports for one module
        self.leg_nodes.append(l0)
        self.leg_nodes.append(r0)
        self.n_joint += 1
        self.idx_counter += 1

    def add_passive_ball(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, quat_r=[1,0,0,0], joint_axis=[0,0,1]):
        # New API for testing a realistic robot docking mechanism
        # This is a lower-level API, exposing pos and quat, which can be generated by a port generator
        # If port_id is None, a new torso is created

        if parent_id is None:
            # The first module in the world
            # print("Add the first module")
            self._build_torsor()
            parent = self.torsos[-1]
            pos = self.initial_pos
            quat = [1,0,0,0]
        else:
            assert parent_id < len(self.leg_nodes), "The port_id is out of range"
            assert pos is not None, "The pos is required for the module"
            assert quat is not None, "The quat is required for the module"
            parent = self.leg_nodes[parent_id]

        if color is None:
            color_l = self.colors[self.idx_counter]
            color_r = lighten_color(color_l, 0.5) # [1,1,1]
        elif isinstance(color, (int, float)):
            color_l = self.colors[color]
            color_r = lighten_color(color_l, 0.5)
        else:
            if len(np.array(color).shape) == 1:
                color_l = color
                color_r = lighten_color(color, 0.5)
            else:
                color_l = color[0]
                color_r = color[1]

        quat = self._convert_quat(quat)

        idx = self.idx_counter
        l0  = etree.SubElement(parent, "body", name=f"l{idx}", pos=vec2string(pos), quat=vec2string(quat))
        

        quat = self._convert_quat([quat_r, quat])
        r0  = etree.SubElement(parent, "body", name=f"r{idx}", pos=vec2string(pos), quat=vec2string(quat))
        if range is not None:
            etree.SubElement(r0, "joint", axis=vec2string(joint_axis), name=f"joint{idx}", pos="0 0 0", range=range, type="hinge", armature="0.05", damping="0.2", limited="auto")
        else:
            etree.SubElement(r0, "joint", axis=vec2string(joint_axis), name=f"joint{idx}", pos="0 0 0", type="hinge", armature="0.05", damping="0.2", limited="auto")
        
        # Attatching the mesh
        # Ball
        if self.mesh_dict["up"].endswith(".obj") or self.mesh_dict["up"].endswith(".stl"):
            etree.SubElement(l0, "geom", type="mesh", name=f"left{idx}", mesh="up", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 0.5", mass=str(self.top_hemi_mass), material="metallic", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["up"] == "SPHERE":
            # Draft mode: Use sphere instead of mesh
            radius = self.robot_cfg["R"]
            etree.SubElement(l0, "geom", type="sphere", name=f"left{idx}", size=f"{radius}", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1", mass=f"{self.top_hemi_mass}", friction="1.0 .0 .0", priority="2")
        else:
            raise ValueError("The mesh should be either a .obj file or a SPHERE")
        
        if self.mesh_dict["bottom"].endswith(".obj") or self.mesh_dict["bottom"].endswith(".stl"):
            etree.SubElement(r0, "geom", type="mesh", name=f"right{idx}", mesh="bottom", rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 0.5", mass=str(self.bottom_hemi_mass), material="metallic", euler="180 0 60", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["bottom"] == "SPHERE":
            radius = self.robot_cfg["R"]
            etree.SubElement(r0, "geom", type="sphere", name=f"right{idx}", size=f"{radius}", rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 1", mass=f"{self.bottom_hemi_mass}", friction="1.0 .0 .0", priority="2")
        # Battery, PCB, Motor
        # if "battery" in self.mesh_dict:
        #     if self.mesh_dict["battery"].endswith(".obj") or self.mesh_dict["battery"].endswith(".stl"):
        #         etree.SubElement(l0, "geom", type="mesh", name=f"battery{idx}", mesh="battery", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 0.5", mass=str(self.battery_mass), material="metallic", contype="10", conaffinity="0")
        #     elif self.mesh_dict["battery"] == "NONE":
        #         pass
        #     else:
        #         raise ValueError("The mesh should be either a .obj file or NONE")
        # if "pcb" in self.mesh_dict:
        #     if self.mesh_dict["pcb"].endswith(".obj") or self.mesh_dict["pcb"].endswith(".stl"):
        #         etree.SubElement(l0, "geom", type="mesh", name=f"pcb{idx}", mesh="pcb", rgba=f"0 0 0 0.5", mass=str(self.pcb_mass), material="metallic", contype="10", conaffinity="0")
        #     elif self.mesh_dict["pcb"] == "NONE":
        #         pass
        #     else:
        #         raise ValueError("The mesh should be either a .obj file or NONE")
        # if "motor" in self.mesh_dict:
        #     if self.mesh_dict["motor"].endswith(".obj") or self.mesh_dict["motor"].endswith(".stl"):
        #         etree.SubElement(l0, "geom", type="mesh", name=f"motor{idx}", mesh="motor", rgba=f"1 0 0 0.5", mass=str(self.motor_mass), contype="10", conaffinity="0")
        #     elif self.mesh_dict["motor"] == "CYLINDER":
        #         etree.SubElement(l0, "geom", type="cylinder", name=f"motor{idx}", pos=vec2string([0,0,-0.015]), quat=vec2string([1,0,0,0 ]), size=f"{0.05} {0.03/2}", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1", mass=str(self.motor_mass), contype="10", conaffinity="0")
        #     elif self.mesh_dict["motor"].startswith("VIRTUAL"):
        #         dist = float(self.mesh_dict["motor"].split("_")[1])
        #         etree.SubElement(l0, "geom", type="sphere", name=f"motor{idx}", size=f"{0.01}", rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1", mass=str(self.motor_mass), contype="10", conaffinity="0", pos=vec2string([0,0,-dist]))
        #     elif self.mesh_dict["motor"] == "NONE":
        #         pass
        #     else:
        #         raise ValueError("The mesh should be either a .obj file or CYLINDER or NONE")


        # # Mount the IMU
        # etree.SubElement(self.sensors, "framequat", name=f"imu_quat{idx}", objtype="xbody", objname=f"l{idx}")
        # etree.SubElement(l0, "site", name=f"imu_site{idx}", pos="0 0 0" ,size="0.01", rgba="0 0 1 1")
        # etree.SubElement(self.sensors, "gyro", name=f"imu_gyro{idx}", site=f"imu_site{idx}")
        # etree.SubElement(self.sensors, "framelinvel", name=f"imu_globvel{idx}", objtype="xbody", objname=f"l{idx}")
        # etree.SubElement(self.sensors, "velocimeter", name=f"imu_vel{idx}", site=f"imu_site{idx}")
        # etree.SubElement(self.sensors, "accelerometer", name=f"imu_acc{idx}", site=f"imu_site{idx}")

        # # Mount the back IMU
        # etree.SubElement(self.sensors, "framequat", name=f"back_imu_quat{idx}", objtype="xbody", objname=f"r{idx}")
        # etree.SubElement(r0, "site", name=f"back_imu_site{idx}", pos="0 0 0" ,size="0.01", rgba="0 0 1 1")
        # etree.SubElement(self.sensors, "gyro", name=f"back_imu_gyro{idx}", site=f"back_imu_site{idx}")
        # # etree.SubElement(self.sensors, "framelinvel", name=f"back_imu_globvel{idx}", objtype="xbody", objname=f"r{idx}")
        # etree.SubElement(self.sensors, "velocimeter", name=f"back_imu_vel{idx}", site=f"back_imu_site{idx}")

        # etree.SubElement(self.contact, "exclude", body1=f"l{idx}", body2=f"r{idx}")



        # if range is not None:
        #     etree.SubElement(self.actuators, "position", ctrlrange="-3.14 3.14", joint=f"joint{idx}", kp="20", kv="0.5", forcerange="-12 12")
        # else:
        #     etree.SubElement(self.actuators, "position", joint=f"joint{idx}", kp="20", kv="0.5", forcerange="-12 12")


        # Assume that there are two ports for one module
        self.leg_nodes.append(l0)
        self.leg_nodes.append(r0)
        self.n_joint += 1
        self.idx_counter += 1

    def add_stick(self, parent_id, radius=0.03, length=0.25, pos=[0,0,0], quat=[1,0,0,0], color=None):
        assert parent_id < len(self.leg_nodes), "The port_id is out of range"
        quat = self._convert_quat(quat)
        if color is None:
            color = self.colors[3]
        elif isinstance(color, (int, float)):
            color = self.colors[color]
        if not isinstance(pos, str):
            pos = vec2string(pos)
        if not isinstance(quat, str):
            quat = vec2string(quat)
        parent = self.leg_nodes[parent_id]
        etree.SubElement(parent, "geom", type="cylinder", pos=pos, quat=quat, size=f"{radius} {length/2}", rgba=f"{color[0]} {color[1]} {color[2]} 1")


    def add_independent_stick(self, parent_id, pos=[0,0,0], quat=[1,0,0,0], color=None, broken=0., pos_offset=0):
        # stick in a new body element
        assert parent_id < len(self.leg_nodes), "The port_id is out of range"
        quat = self._convert_quat(quat)
        if color is None:
            color = self.colors[3]
        elif isinstance(color, (int, float)):
            color = self.colors[color]

        parent = self.leg_nodes[parent_id]
        # etree.SubElement(parent, "geom", type="cylinder", pos=vec2string(pos), quat=vec2string(quat), size=f"{radius} {length/2}", rgba=f"{color[0]} {color[1]} {color[2]} 1")

        idx = self.passive_idx_counter
        stick = etree.SubElement(parent, "body", name=f"passive{idx}", pos=vec2string(pos), quat=vec2string(quat))

        radius = self.robot_cfg["r"]
        length = self.robot_cfg["l_"] 
        # assert type in ["cylinder", "capsule", "mesh"], "The type should be either 'cylinder', 'capsule' or 'mesh'"
        if self.mesh_dict["stick"].endswith(".obj") or self.mesh_dict["stick"].endswith(".stl"):
            if broken != 0:
                assert "cut_stick" in self.mesh_dict, "The cut_stick mesh should be provided"
            etree.SubElement(stick, "geom", name=f"stick{idx}", type="mesh", pos=vec2string(np.array([0,0,0])+pos_offset), quat="1 0 0 0", mesh="stick" if broken==0 else "cut_stick", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass=f"{self.mass['stick']*(1-broken)}", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["stick"] == "CYLINDER":
            # assert broken == 0, "The CYLINDER mesh does not support broken stick yet" 
            etree.SubElement(stick, "geom", name=f"stick{idx}", type="cylinder", pos=vec2string(np.array([0,0,length/2 - length*(1-broken) /2])+pos_offset), quat="1 0 0 0", size=f"{radius} {length*(1-broken) /2 }", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass=f"{self.mass['stick']*(1-broken)}", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["stick"] == "CAPSULE":
            etree.SubElement(stick, "geom", name=f"stick{idx}", type="capsule", pos=vec2string(np.array([0,0,length/2 - length*(1-broken) /2])+pos_offset), quat="1 0 0 0", size=f"{radius} {length/2 *(1-broken)}", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass=f"{self.mass['stick']*(1-broken)}", friction="1.0 .0 .0", priority="2")
        else:
            raise ValueError("The mesh should be either a .obj file or CYLINDER or CAPSULE")


        self.passive_idx_counter += 1
        self.leg_nodes.append(stick)
        node_id = len(self.leg_nodes)-1

        return node_id
    

    def add_sock(self, parent_id, radius=0.04, length=0.08, stick_length=0.235, color=None, thickness=0.01):

        if color is None:
            color = self.colors[3]
        elif isinstance(color, (int, float)):
            color = self.colors[color]
        
        etree.SubElement(self.leg_nodes[parent_id], "geom", type="cylinder", name=f"sock{self.sock_idx_counter}", pos=f"0 0 {-stick_length/2+length/2-thickness} ", quat="1 0 0 0", size=f"{radius} {length/2}", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass="0")
        self.sock_idx_counter += 1

    def add_simple_ball(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, passive=False):
        # Build a perfect sphere and return the two ids of the two hemispheres
        quat_r = construct_quaternion([1.,0,0], np.pi)
        if not passive:
            self.add_one_module(parent_id, pos, quat, range, color, quat_r=quat_r, joint_axis=[0,0,-1])
        else:
            self.add_passive_ball(parent_id, pos, quat, range, color, quat_r=quat_r, joint_axis=[0,0,-1])
        return [len(self.leg_nodes)-2, len(self.leg_nodes)-1]
    

    

    # def add_draft_ball(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, model_motor=True):
    #     # Build a sphere with two spheres, without dependence on the imported mesh
    #     quat_r = construct_quaternion([1.,0,0], np.pi)
    #     self.add_one_module(parent_id, pos, quat, range, color, radius=0.07, quat_r=quat_r, mass=[self.mass["top_hemi"], self.mass["bottom_hemi"], self.mass["motor"]], joint_axis=[0,0,-1], model_motor=model_motor)
    #     return [len(self.leg_nodes)-2, len(self.leg_nodes)-1]
        

    def add_walls(self, transparent=False, angle=0):
        wall = etree.SubElement(self.worldbody, "body", name=f"boundary", pos="0 0 0", axisangle=f"0 0 1 {angle}")
        etree.SubElement(wall, "geom", name=f"boundary/right", pos="0 1 0.25", type="box", material="boundary", size="25 0.1 0.5", **({} if not transparent else {'rgba': "0.1 0.1 0.1 0.0"}))
        etree.SubElement(wall, "geom", name=f"boundary/left", pos="0 -1 0.25", type="box", material="boundary", size="25 0.1 0.5", **({} if not transparent else {'rgba': "0.1 0.1 0.1 0.0"}))

    def add_bumps(self, h = 0.1, h_func = None):
        for i in range(20):
            if h_func is not None:
                h = h_func(i)
            etree.SubElement(self.worldbody, "geom", name=f"bump{i}", pos=f"{i+1} 0 {h/2}", type="box", material="boundary", size=f"0.1 25 {h}")

    def add_minibump(self, pos, angle, height=0.1, width=0.1, length=1):
        etree.SubElement(self.worldbody, "geom", name=f"obstacle{self.obstacle_idx_counter}", pos=f"{pos[0]} {pos[1]} {height/2}", axisangle=f"0 0 1 {angle}", type="box", material="boundary", size=f"{width/2} {length/2} {height/2}")
        self.obstacle_idx_counter += 1

    def _create_box_xml(self, name, x, y, z, width, height, depth):
        etree.SubElement(self.worldbody, 'geom', name=name, type='box', pos=f"{x} {y} {z}", size=f"{width/2} {height/2} {depth/2}", material="boundary")

    def add_stairs(self, start_distance=0, num_steps=10, step_width=2, step_height=0.1, step_depth=1, direction='x'):
        # start_distance: the distance from the origin to the first step;
        # for adding some flat plane before the stairs
        for i in range(num_steps):
            if direction == 'x':
                x, y, z = i * step_depth + start_distance, 0, i * step_height
                self._create_box_xml(f"step{i}", x, y, z, step_depth, step_width, step_height)
            elif direction == 'y':
                x, y, z = 0, i * step_depth + start_distance, i * step_height
                self._create_box_xml(f"step{i}", x, y, z, step_width, step_depth, step_height)

    def set_hfield(self, file="rugged.png", radius_x=20, radius_y=20, elevation_z=0.3, base_z=0.1):
        self.worldbody.remove(self.floor_elem)
        
        etree.SubElement(self.worldbody, "geom", name="floor", pos="0 0 0", type="hfield", material="hfield", conaffinity="1", condim="6", friction="1.0 .0 .0", hfield="rugged")
        etree.SubElement(self.assets, "hfield", name="rugged", size=f"{radius_x} {radius_y} {elevation_z} {base_z}", file=file)

    def change_color(self, node_id, color):
        # node = self.leg_nodes[node_id]
        for i, geom in enumerate(self.tree.iter('geom')):
            if 'rgba' in geom.attrib:
                # Set the 'rgba' attribute to the new color
                if i == node_id:
                    geom.set('rgba', f"{color[0]} {color[1]} {color[2]} 1")

    def change_color_name(self, name, color):
        # node = self.leg_nodes[node_id]
        for i, geom in enumerate(self.tree.iter('geom')):
            if 'rgba' in geom.attrib and 'name' in geom.attrib and name in geom.attrib['name']:
                # Set the 'rgba' attribute to the new color
                geom.set('rgba', f"{color[0]} {color[1]} {color[2]} 1")

    def remove_contact_name(self, name):
        for i, geom in enumerate(self.tree.iter('geom')):
            if 'name' in geom.attrib and name in geom.attrib['name']:
                geom.set('contype', "2")
                geom.set('conaffinity', "0")
            # etree.SubElement(self.contact, "exclude", body1=f"floor", body2=f"r{idx}")

    def delete_body(self, body_name, keep_tag=None):
        for i, body in enumerate(self.tree.iter('body')):
            if 'name' in body.attrib and body_name in body.attrib['name']:
                if keep_tag is None:
                    parent = body.getparent()  # Get the actual parent of the element
                    if parent is not None:
                        parent.remove(body)
                    for contact in self.root.findall('contact'):
                        for exclude in contact.findall('exclude'):
                            if exclude.attrib.get('body1') == body_name or exclude.attrib.get('body2') == body_name:
                                contact.remove(exclude)
                else:
                    # Collect children that should be removed
                    to_remove = [child for child in body if child.tag != keep_tag]
                    # Remove them from the parent
                    for child in to_remove:
                        print(f"Removing {child.tag}")
                        body.remove(child)
        
                
    def delete_sensor(self, sensor_name):
        for i, sensor in enumerate(self.tree.iter('sensor')):
            for child in sensor:
                if 'name' in child.attrib and sensor_name in child.attrib['name']:
                    sensor.remove(child)

    def delete_joint(self, joint_name):
        for i, joint in enumerate(self.tree.iter('joint')):
            if 'name' in joint.attrib and joint_name in joint.attrib['name']:
                parent = joint.getparent()
                if parent is not None:
                    parent.remove(joint)
        for i, actuators in enumerate(self.tree.iter('actuator')):
            for actuator in actuators:
                if 'joint' in actuator.attrib and joint_name in actuator.attrib['joint']:
                    parent = actuator.getparent()
                    if parent is not None:
                        parent.remove(actuator)

    def hind_imu(self, name):
        for i, site in enumerate(self.tree.iter('site')):
            if 'name' in site.attrib and name in site.attrib['name']:
                site.set('rgba', "0 0 0 0")


    # def delete_sensor(self, motor_name):

    def save(self, filename="m1"):
        if not filename.endswith(".xml"):
            filename = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", f'{filename}.xml')

        self.tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')
        print(f"Saved the robot to {filename}")
        return filename
    
    def get_xml(self, fix_file_path=False):
        if fix_file_path:
            root = fix_model_file_path(self.root)
        else:
            root = self.root
        return etree.tostring(root, pretty_print=True, xml_declaration=False, encoding='utf-8').decode()
    
    @property
    def init_quat(self):
        return quaternion_from_vectors(self.lleg_vec,  np.array([1, 0, 0]))



if __name__ == "__main__":
    builder = RobotBuilder(terrain="flat")
    builder.add_module()
    builder.add_stairs(start_distance = 3)
    builder.add_walls()
    f = builder.save("m1stairs")
    # f = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", f'm1rugged.xml')
    view(f, False)