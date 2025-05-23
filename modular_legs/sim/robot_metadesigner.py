


import copy
import itertools
import os
import pdb
from collections import defaultdict
import random
import tempfile
import time
import numpy as np
from numpy import cos, sin, tan, pi
import yaml
import mujoco
from modular_legs.utils.model import is_headless
if not is_headless():
    import mujoco.viewer

from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.robot_builder import RobotBuilder
from modular_legs.sim.robot_designer import DEFAULT_ROBOT_CONFIG, Connect, ConnectAsym, RobotDesigner, calculate_transformation_matrix, matrix_to_pos_quat, numpy_to_native
from modular_legs.utils.math import construct_quaternion, quaternion_multiply2
from modular_legs.utils.model import quaternion_from_vectors, is_headless
from modular_legs.utils.model import get_jing_vector, get_joint_pos_addr, get_local_zvec
from modular_legs.utils.others import is_list_like, is_number




class MetaDesignerAsym(RobotDesigner):
    '''
    A new version of the metadesigner that matches assymetric designs of the real robot
    mesh_mode: 
        "default": Use mesh files for spheres and use primitive cylinder for sticks;
        "all_meshes": Use mesh files for all parts;
        "all_primitives": Use primitive spheres / cylinders for all parts;
        "draft": Use primitive spheres / capsules for all parts;
    allow_custom_joint:
        True: Allow custom default joint positions; In this case, we allow init sibling orientation (TODO)
    '''


    def __init__(self, init_pipeline=None, robot_cfg=None, mesh_dict=None, allow_custom_joint=False, allow_overlapping=False, color=None, broken_mask=None, mesh_mode=None):

        # super().__init__(init_pipeline=init_pipeline, robot_cfg=robot_cfg, mesh_dict=mesh_dict, allow_custom_joint=allow_custom_joint, allow_overlapping=allow_overlapping, color=color, broken_mask=broken_mask)
        # robot_cfg = DEFAULT_ROBOT_CONFIG if robot_cfg is None else robot_cfg
        # assert robot_cfg is not None, "Robot config is required for MetaDesignerAsym"
        self.robot_cfg = robot_cfg
        self.robot_cfg["l"]  = robot_cfg["l_"] - (robot_cfg["R"] - np.sqrt(robot_cfg["R"]**2 - robot_cfg["r"]**2))
        self.mesh_dict = {
            "up": "top_lid.obj",
            "bottom": "bottom_lid.obj",
            "stick": "leg4.4.obj",
            "battery": "battery.obj",
            "pcb": "pcb.obj",
            "motor": "motor.obj"
        } if mesh_dict is None else mesh_dict
        self.broken_mask = copy.deepcopy(broken_mask)+[0]*100000 if broken_mask is not None else [0]*100000
        # self.mesh_mode = mesh_mode # Deprecated
        self.allow_custom_joint = allow_custom_joint
        self.allow_overlapping = allow_overlapping
        self.color = [0.6, 0.6, 0.6] if color is None else color
        self.lite = True
        self.connecting = ConnectAsym if self.lite else Connect

        init_pipeline = init_pipeline.copy() if init_pipeline is not None else None

        # How to add a ball and stick on a stick
        candidate_ball_pos = [5] if not self.lite else [7]
        candidate_ball_orientation = [0,1,2]
        candidate_stick_pos = [0,1,2]
        self.stick_sibling_poses = list(itertools.product(candidate_ball_pos, candidate_ball_orientation, candidate_stick_pos))
        self.ball_sibling_poses = [[1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]


        if init_pipeline is not None:
            if init_pipeline and init_pipeline[0] == "asym":
                # identifier check
                assert init_pipeline.pop(0) == "asym", "Invalid initial pipeline"

            assert len(init_pipeline) % 4 == 0, "[The design pipeline requires initial pose bit] Invalid init pipeline length: %d" % len(init_pipeline)
            self.reset()
            for step in np.reshape(init_pipeline, (-1, 4)):
                self.step(step)
        

    def reset(self, init_a_module=True):
        
        self.builder = RobotBuilder(mesh_dict=self.mesh_dict, robot_cfg=self.robot_cfg)
        self.connection_types = ["00", "02", "10", "12", "20", "22"] 

        self.available_ports = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: defaultdict(list)))))
        self.parent_id_to_type = {} # {parent_id: parent_type}

        self.node_ids = []
        self.robot_properties = {}

        self.node_ids = []
        self.module_id_dict = {}
        self.module_id_counter = 0
        self.available_pos_dict = {} # {module_id: [pos_id]}

        self.polished_pipeline = ["asym"]
        
        if init_a_module:
            self._init_a_module()
        
    def _init_a_module(self, broken=None):
        node_ids = []
        node_ids += self._add_a_ball()
        node_ids += self._add_a_stick(node_ids[0], 0, 0, 0, broken=self.broken_mask.pop(0) if broken is None else broken[0])
        node_ids += self._add_a_stick(node_ids[1], 0, 0, 0, broken=self.broken_mask.pop(0) if broken is None else broken[1])

        self.module_id_dict[self.module_id_counter] = node_ids
        self.available_pos_dict[self.module_id_counter] = list(range(18))
        self.module_id_counter += 1


    def _init_a_right_stick_module(self, broken=None):
        node_ids = []
        node_ids += self._add_a_ball()
        node_ids += self._add_a_stick(node_ids[0], 0, 0, 0, broken=self.broken_mask.pop(0) if broken is None else broken[0])
        node_ids += self._add_a_stick(node_ids[1], 0, 0, 0, broken=self.broken_mask.pop(0) if broken is None else broken[1])

        self.module_id_dict[self.module_id_counter] = node_ids
        self.available_pos_dict[self.module_id_counter] = list(range(18))

        self.builder.delete_body(f"l{self.module_id_counter}")
        self.builder.delete_body(f"passive{self.builder.passive_idx_counter-2}")
        self.builder.delete_body(f"passive{self.builder.passive_idx_counter-2}")
        self.builder.delete_sensor(f"imu_quat{self.module_id_counter}")
        self.builder.delete_sensor(f"imu_gyro{self.module_id_counter}")
        self.builder.delete_sensor(f"imu_globvel{self.module_id_counter}")
        self.builder.delete_sensor(f"imu_vel{self.module_id_counter}")
        self.builder.delete_joint(f"joint{self.module_id_counter}")
        self.builder.hind_imu(f"imu_site{self.module_id_counter}")

        self.module_id_counter += 1

    def _add_a_module(self, module_id, pos_a, pos_b, rotation):
        ''' 
            Position [A] on a module:
            0-6   stick on upper ball ; pos 7-1 of the stick
            7-8  upper ball           ; pos 1-2 of the ball
            9-10 lower ball           ; pos 1-2 of the ball
            11-17 stick on lower ball ; pos 1-7 of the stick
            Position [B] on a module:
            0-6   stick on upper ball ; pos 7-1 of the stick
            7-8  upper ball           ; pos 1-2 of the ball
        '''
        assert pos_a >= 0 and pos_a < 18, "Invalid module position: %d" % pos_a
        assert pos_b >= 0 and pos_b < 9, "Invalid module position: %d" % pos_b

        # print(f"Adding a module: {module_id}, {pos_a}, {pos_b}, {rotation}")

        if module_id not in self.module_id_dict:
            print(f"Invalid module id: {module_id}; Available module ids: {self.module_id_dict.keys()}")
            pdb.set_trace()
        if pos_a not in self.available_pos_dict[module_id]:
            print(f"Invalid module position: {pos_a}; Available positions: {self.available_pos_dict[module_id]}")
            pdb.set_trace()
        if pos_b not in list(range(9)):
            print(f"Invalid module position: {pos_b}; Available positions: {list(range(9))}")
            pdb.set_trace()
        if rotation not in self.get_available_rotation_ids(pos_a, pos_b):
            print(f"Invalid rotation: {rotation}; Available rotations: {self.get_available_rotation_ids(pos_a, pos_b)}")
            pdb.set_trace()



        parent_id = self._module_id_to_parent_id(module_id, pos_a)
        position_id_a = self._module_pos_to_local_pos(pos_a)
        position_id_b = self._module_pos_to_local_pos(pos_b)


        if pos_b <= 6:
            # stick -> module
            ls_id = self._add_a_stick(parent_id, position_id_a=position_id_a, position_id_b=position_id_b, rotation_id=rotation, broken=self.broken_mask.pop(0))
            ball_ids = self._add_a_ball(ls_id[0], 0, 0, 0)
            # We hardcode the stick orientation here to simplify the encoding
            rs_id = self._add_a_stick(ball_ids[1], 0, 0, 0, broken=self.broken_mask.pop(0))
        else:
            # ball -> module
            ball_ids = self._add_a_ball(parent_id, position_id_a=position_id_a, position_id_b=position_id_b, rotation_id=rotation)
            ls_id = self._add_a_stick(ball_ids[0], 0, 0, 0, broken=self.broken_mask.pop(0))
            rs_id = self._add_a_stick(ball_ids[1], 0, 0, 0, broken=self.broken_mask.pop(0))

        # upper ball, lower ball, upper stick, lower stick
        node_ids = ball_ids + ls_id + rs_id

        self.module_id_dict[self.module_id_counter] = node_ids
        self.available_pos_dict[self.module_id_counter] = list(range(18))

        self.available_pos_dict[module_id].remove(pos_a)
        self.available_pos_dict[self.module_id_counter].remove(pos_b)
        
        self.module_id_counter += 1


    def add_extra_stick(self, module_id, pos_a, pos_b, rotation, broken=0, stick_id=None, reserve_next_id=False):

        parent_id = self._module_id_to_parent_id(module_id, pos_a)
        position_id_a = self._module_pos_to_local_pos(pos_a)
        position_id_b = self._module_pos_to_local_pos(pos_b)

        cut_length = self.robot_cfg["l_"] * (broken)

        pos_offset = np.array([0.,0,-cut_length])
        if stick_id is not None:
            self.builder.passive_idx_counter = stick_id
        self._add_a_stick(parent_id, position_id_a=position_id_a, position_id_b=position_id_b, rotation_id=rotation, broken=broken, pos_offset=pos_offset)
        if reserve_next_id:
            self.builder.passive_idx_counter += 1


    def add_extra_ball(self, module_id, pos_a, pos_b, rotation):
        parent_id = self._module_id_to_parent_id(module_id, pos_a)
        position_id_a = self._module_pos_to_local_pos(pos_a)
        position_id_b = self._module_pos_to_local_pos(pos_b)

        ball_ids = self._add_a_ball(parent_id, position_id_a=position_id_a, position_id_b=position_id_b, rotation_id=rotation, passive=True)


    def _module_id_to_parent_id(self, module_id, module_pos):
        ''' 
            Position id on a module:
            0-6   stick on upper ball ; pos 7-1 of the stick
            7-8  upper ball           ; pos 1-2 of the ball
            9-10 lower ball           ; pos 1-2 of the ball
            11-17 stick on lower ball ; pos 1-7 of the stick
        '''
        node_ids = self.module_id_dict[module_id]
        assert len(node_ids) == 4, f"Invalid node_ids: {node_ids}"
        assert module_pos < 18, "Invalid module position: %d" % module_pos
        assert module_pos >= 0, "Invalid module position: %d" % module_pos
        # upper ball, lower ball, upper stick, lower stick
        if module_pos <= 6:
            return node_ids[2]
        elif module_pos <= 8:
            return node_ids[0]
        elif module_pos <= 10:
            return node_ids[1]
        else:
            return node_ids[3]

    def _module_pos_to_local_pos(self, module_pos):
        ''' 
            Position id on a module:
            0-6   stick on upper ball ; pos 7-1 of the stick
            7-8  upper ball           ; pos 1-2 of the ball
            9-10 lower ball           ; pos 1-2 of the ball
            11-17 stick on lower ball ; pos 1-7 of the stick
        '''
        assert module_pos < 18, "Invalid module position: %d" % module_pos
        assert module_pos >= 0, "Invalid module position: %d" % module_pos

        if module_pos <= 6:
            return 7 - module_pos
        elif module_pos <= 8:
            return module_pos - 6
        elif module_pos <= 10:
            return module_pos - 8
        else:
            return module_pos - 10
        


    def step(self, step):
        assert len(step) == 4, "Invalid pipeline step length: %d" % len(step)
        self._add_a_module(*step)


    def get_available_module_ids(self):
        return list(self.module_id_dict.keys())
    
    def get_available_posa_ids(self, module_id):
        return self.available_pos_dict[module_id]
    
    def get_available_posb_ids(self):
        return list(range(9))
    
    def get_available_rotation_ids(self, posa, posb):
        stick_side = [1,2,3,4,5,6,11,12,13,14,15,16]
        if posa in stick_side and posb in stick_side:
            return [0,1]
        else:
            return [0,1,2]
        
    def paint(self, color="black"):
        dark_grey = (0.15,0.15,0.15)
        black = (0.1,0.1,0.1)
        if color == "black":
            self.builder.change_color_name("l", black)
            self.builder.change_color_name("r", dark_grey)
            self.builder.change_color_name("s", dark_grey)



    

def rand_gen_test():
    robot_designer = MetaDesigner()
    robot_designer.reset(random.choice([0, 1]))
    for _ in range(3):
        module = random.choice([0, 1])
        parent = np.random.choice(robot_designer.node_ids)
        pos = np.random.choice(robot_designer.get_pos_id_list(module, parent))
        rotation = np.random.choice(robot_designer.get_rotation_id_list(module, parent, pos))
        sibling_pos = np.random.choice(robot_designer.get_sibling_id_list(module))
        robot_designer.step([module, parent, pos, rotation, sibling_pos])
    robot_designer.save("test", fix=False, pos=(0, 0, 0.5), quat=[1, 0, 0, 0], joint_pos=None, render=True)

def test_dp():
    robot_designer = MetaDesigner(init_pipeline=[1, 1, 2, 9, 7, 6, 1, 4, 0, 4, 3, 1, 4, 3, 5, 5, 0, 13, 1, 0, 3, 1, 5, 1, 9, 1, 0, 18, 5, 1, 5] , ini_pose_bit=True)
    robot_designer.set_terrain("walls")
    robot_designer.save("test", fix=False, 
                        pos=[0, 0, 0.38185636162757874], 
                        quat=[0.009872849099338055, -0.2820669412612915, -0.9210168719291687, -0.2684561014175415], 
                        joint_pos=[0.16777732968330383, 1.4343171119689941, 0.9361017346382141, -0.24445606768131256, -0.022494299337267876, -3.5943691730499268, 0.055646397173404694], 
                        render=True)
    
def test_triped():
    robot_designer = MetaDesigner(init_pipeline=[0, 1, 3, 9, 5, 0, 0, 6, 1, 2, 4] , ini_pose_bit=True)
    # robot_designer.set_terrain("walls")
    quat = construct_quaternion([1, 0, 0], pi)
    print("quat", quat)
    robot_designer.builder.save("tripped-0753W0G25")
    robot_designer.save("test", fix=False, 
                        pos=[0, 0, 0.2], 
                        quat=quat, 
                        joint_pos=[0,0,pi/1.5], 
                        render=True)

def test_local_rew():
    from modular_legs.sim.evolution.encoding_wrapper import polish

    pipe = [0, 0, 0, 2, 0, 0, 1, 3, 8, 23, 0, 1, 6, 7, 8, 0
            ]
    if pipe != polish(pipe):
        raise ValueError(f"In valid design! What if:  {polish(pipe)}")
    
    robot_designer = MetaDesigner(init_pipeline=pipe , ini_pose_bit=True)
    # robot_designer.set_terrain("walls")
    quat = construct_quaternion([0, 0, 1], -pi/2)
    print("quat", quat)
    robot_designer.builder.save("TLR")
    robot_designer.save("test", fix=True, 
                        pos=[0,0,0.1825061684846878], 
                        quat=[0.04657600820064545, -0.8202893137931824, 0.5689055323600769, 0.03609202802181244], 
                        joint_pos=[2.2334494590759277, 0.031052296981215477, 0.08712831884622574, 0.562721312046051], 
                        render=True)
# Projected Forward:  [-0.99631095 -0.06866933 -0.05146855]
# Projected Updir:  [ 0.08235737 -0.93366418 -0.34855196]

def test_new_designer():
    pipe = ["asym"]
    robot_designer = MetaDesignerAsym(init_pipeline=pipe)
    robot_designer._add_a_module(0,2,0,0)
    robot_designer._add_a_module(0,4,0,0)
    robot_designer._add_a_module(0,14,0,0)
    robot_designer._add_a_module(0,16,0,0)
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(tmpdir, fix=True, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], joint_pos=None, render=True)

def test_new_designer():
    pipe = ["asym", 0,2,0,0, 0,4,0,0, 0,14,0,0, 0,16,0,0]
    robot_designer = MetaDesignerAsym(init_pipeline=pipe)
    # robot_designer._add_a_module(0,2,0,0)
    # robot_designer._add_a_module(0,4,0,0)
    # robot_designer._add_a_module(0,14,0,0)
    # robot_designer._add_a_module(0,16,0,0)
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(tmpdir, fix=True, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], joint_pos=None, render=True)

def test_new_designer3():
    # pipe = ["asym", 0, 2, 5, 0, 1, 3, 4, 0]
    pipe = ["asym", 0, 2, 5, 0, 1, 5, 1, 1]
    
    robot_designer = MetaDesignerAsym(init_pipeline=pipe, allow_overlapping=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(tmpdir, fix=True, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], joint_pos=None, render=True)
        

def rand_gen_test():
    pipe = ['asym']
    robot_designer = MetaDesignerAsym()
    robot_designer.reset()
    for _ in range(3):
        module = random.choice(robot_designer.get_available_module_ids())
        posa = np.random.choice(robot_designer.get_available_posa_ids(module))
        posb = np.random.choice(robot_designer.get_available_posb_ids())
        rotation = np.random.choice(robot_designer.get_available_rotation_ids(posa, posb))
        step = [module, posa, posb, rotation]
        robot_designer.step(step)
        pipe += step
    with tempfile.TemporaryDirectory() as tmpdir:
        robot_designer.save(tmpdir, fix=True, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], joint_pos=None, render=True)

if __name__ == "__main__":
    # test_dp()
    # test_triped1()
    # test_local_rew()
    # test_new_desigssner()
    # rand_gen_test()
    test_new_designer3()
    pass