

import numpy as np
from modular_legs.sim.robot_metadesigner import MetaDesignerAsym


cfg = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass": 0.231, #0.26,
        "top_hemi_mass": 0.74,
        "bottom_hemi_mass": 0.534,
    }

ROBOT_CFG_AIR1S = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass": 0.1734, #0.154,
        "top_hemi_mass": 0.1153, 
        "battery_mass": 0.122,
        "motor_mass": 0.317,
        "bottom_hemi_mass": 0.1623, #0.097, 
        "pcb_mass": 0.1
    }

ROBOT_CFG_PLA = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass":0.2385,
        "top_hemi_mass": 0.225,
        "bottom_hemi_mass": 0.212,
        "battery_mass": 0.122,
        "motor_mass": 0.317,
        "pcb_mass": 0.1
    }


MESH_DICT_FINE = {
                "up": "top_lid.obj",
                "bottom": "bottom_lid.obj",
                "stick": "leg4.4.obj",
                "battery": "battery.obj",
                "pcb": "pcb.obj",
                "motor": "motor.obj",
                "cut_stick": "legcut.obj"
            }

MESH_DICT_DRAFT = {
                "up": "SPHERE",
                "bottom": "SPHERE",
                "stick": "CAPSULE",
                "battery": "NONE",
                "pcb": "NONE",
                "motor": "NONE"
            }

MESH_DICT_DRAFT_CYLINDER = {
                "up": "SPHERE",
                "bottom": "SPHERE",
                "stick": "CYLINDER",
                "battery": "NONE",
                "pcb": "NONE",
                "motor": "NONE"
            }


def gen_quadrupedX4air1s():

    # Define four additional modules
    pipe = [ 0,1,0,0, # Parent module ID, parent position ID, child position ID, rotation ID
             0,3,0,0, 
             0,13,0,0, 
             0,15,0,0 ]
    
    robot_designer = MetaDesignerAsym(init_pipeline=pipe , robot_cfg=ROBOT_CFG_AIR1S, mesh_dict=MESH_DICT_FINE)

    robot_designer.paint()

    # A Mujoco XML file will be saved in modular_legs/sim/assets/robots/quadrupedX4air1s.xml
    robot_designer.builder.save("quadrupedX4air1s")


if __name__ == "__main__":
    gen_quadrupedX4air1s()