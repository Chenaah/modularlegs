




import time
import numpy as np
import mujoco

from modular_legs.utils.model import XMLCompiler, get_joint_pos_addr, quaternion_from_vectors
from modular_legs.utils.others import is_list_like, is_number


def self_collision_check(m, d, robot_properties, viewer=None):
    '''
    Randomly set joints and check if the robot is in self-collision.
    '''

    n_sim_steps = 10
    n_random_joints = 100
    
    n_self_collision_list = []
    random_joints = [np.zeros(m.nu)] + [np.random.uniform(0,2*np.pi, m.nu) for _ in range(n_random_joints-1)]

    qpos = np.zeros(m.nq)
    qpos[2] = 1
    qvel = np.zeros(m.nv)
    
    for joints in random_joints:
        qpos[get_joint_pos_addr(m)] = joints
        n_self_collision = 0
        for _ in range(n_sim_steps):
            step_start = time.time()

            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_step(m, d)

            for contact in d.contact:
                if contact.geom1 != 0 and contact.geom2 != 0:
                    b1 = m.body(m.geom(contact.geom1).bodyid).name
                    b2 = m.body(m.geom(contact.geom2).bodyid).name
                    if not (((b1[0] == "l" and b2[0] == "r") or (b1[0] == "r" and b2[0] == "l")) and (b1[1] == b2[1])):
                        n_self_collision += 1

            if viewer is not None:
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
                # viewer.opt.frame = 1 # visulize axis
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0 :
                    time.sleep(time_until_next_step)

        n_self_collision_list.append(n_self_collision)

    robot_properties["init_self_collision"] = n_self_collision_list[0] > 0
    robot_properties["self_collision_rate"] = np.sum(np.array(n_self_collision_list) > 0) / n_random_joints


def fast_self_collision_check(pipeline):
    '''
    Roughly check if the robot is in self collision.
    '''
    from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_DRAFT_CYLINDER, ROBOT_CFG_AIR
    from modular_legs.sim.robot_metadesigner import MetaDesignerAsym


    # ROBOT_CFG_AIR["delta_l"] = -0.01
    # ROBOT_CFG_AIR["stick_ball_l"] = -0.01
    ROBOT_CFG_AIR["stick_ball_l"] = -0.02


    designer = MetaDesignerAsym(pipeline, robot_cfg=ROBOT_CFG_AIR, mesh_dict=MESH_DICT_DRAFT_CYLINDER)
    c = XMLCompiler(designer.get_xml())
    xml = c.get_string()

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    n_sim_steps = 10
    # n_random_joints = 100
    
    n_self_collision_list = []
    random_joints = [np.zeros(m.nu)]

    qpos = np.zeros(m.nq)
    qpos[2] = 1
    qvel = np.zeros(m.nv)
    
    for joints in random_joints:
        qpos[get_joint_pos_addr(m)] = joints
        n_bb_collision = 0
        for _ in range(n_sim_steps):
            step_start = time.time()

            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_step(m, d)

            for contact in d.contact:
                if contact.geom1 != 0 and contact.geom2 != 0:
                    b1 = m.body(m.geom(contact.geom1).bodyid).name
                    b2 = m.body(m.geom(contact.geom2).bodyid).name
                    if not (((b1[0] == "l" and b2[0] == "r") or (b1[0] == "r" and b2[0] == "l")) and (b1[1] == b2[1])):
                    # if (((b1[0] == "l" and b2[0] == "r") or (b1[0] == "r" and b2[0] == "l") or (b1[0] == "r" and b2[0] == "r") or (b1[0] == "l" and b2[0] == "l")) and (b1[1] != b2[1])):
                        n_bb_collision += 1


    return n_bb_collision



def stable_state_check(m, d, robot_properties, theta, viewer=None):

    qvel = np.zeros(m.nv)
    n_sim_steps = 100
    init_quat_list = [1,0,0,0] + [quaternion_from_vectors([0, np.cos(theta), np.sin(theta)], target_vec) for target_vec in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    if viewer is not None:
        n_sim_steps = 100
    
    stable_quat_list = []
    stable_pos_list = []
    heright_list = []

    for quat in init_quat_list:
        qpos = np.zeros(m.nq)
        qpos[2] = 1
        qpos[3:7] = quat
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        for _ in range(n_sim_steps):
            step_start = time.time()

            mujoco.mj_step(m, d)

            if viewer is not None:
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
                # viewer.opt.frame = 1 # visulize axis
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0 :
                #     time.sleep(time_until_next_step)
        stable_quat_list.append(d.qpos[3:7])
        stable_pos_list.append(d.qpos[0:3])

        xposes = np.array(d.xipos)
        average_height = np.mean(xposes[:,2])
        heright_list.append(average_height)

    robot_properties["stable_quat"] = stable_quat_list
    robot_properties["stable_pos"] = stable_pos_list
    robot_properties["stable_height"] = heright_list



def movability_check(m, d, robot_properties, viewer=None, config_dict=None, stable_pos_list=None, stable_quat_list=None):

    n_sim_steps = 100
    if viewer is not None:
        n_sim_steps = 100

    assert config_dict is not None, "config_dict should be provided for movability test"
    assert "init_pos" in config_dict, "init_pos should be provided for movability test"
    assert "init_quat" in config_dict, "init_quat should be provided for movability test"

    if is_list_like(config_dict["init_pos"]):
        init_pos = config_dict["init_pos"]
    elif is_number(config_dict["init_pos"]):
        assert stable_pos_list is not None, "Stable state test should be done before movability test"
        init_pos = stable_pos_list[config_dict["init_pos"]]
    if is_list_like(config_dict["init_quat"]):
        init_quat = config_dict["init_quat"]
    elif is_number(config_dict["init_quat"]):
        assert stable_quat_list is not None, "Stable state test should be done before movability test"
        init_quat = stable_quat_list[config_dict["init_quat"]]

    qpos = np.zeros(m.nq)
    qpos[:3] = init_pos
    qpos[3:7] = init_quat
    qvel = np.zeros(m.nv)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    acc_speed = 0
    for _ in range(n_sim_steps):
        step_start = time.time()

        # Assume position control here
        d.ctrl[:] = np.array(d.qpos[get_joint_pos_addr(m)]) + np.random.uniform(-1, 1, m.nu)

        mujoco.mj_step(m, d)

        speed = np.linalg.norm(d.qvel[:2])
        acc_speed += speed
        # print(f"Speed: {speed}; Acc speed: {acc_speed}")

        if viewer is not None:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
            # viewer.opt.frame = 1 # visulize axis
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0 :
                time.sleep(time_until_next_step)

    robot_properties["ave_speed"] = acc_speed / n_sim_steps