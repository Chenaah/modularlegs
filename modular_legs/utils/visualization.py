

import os
import pdb
import time
import mujoco
import numpy as np
from numpy import sin, cos
from omegaconf import OmegaConf
import omegaconf
from modular_legs.sim.designer_utils import fast_self_collision_check
from modular_legs.utils.files import load_cfg
from modular_legs.utils.math import quat_rotate, quat_rotate_inverse, quaternion_to_euler, quaternion_to_euler2, wxyz_to_xyzw
from modular_legs.utils.model import XMLCompiler, compile_xml, get_joint_pos_addr
from modular_legs.utils.others import is_list_like

DEFAULT_ROBOT_CONFIG = {
    "theta": 0.4625123,
    "R": 0.07,
    "r": 0.03,
    "l_": 0.236,
    "delta_l": -0.001,
    "stick_ball_l": -0.001,
    "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
    "stick_mass": 0.26,
    "top_hemi_mass": 0.74,
    "bottom_hemi_mass": 0.534
}

def render_line(viewer, p1, p2, color=(1,1,1,0.5)):
    a, b, c = p1
    d, e, f = p2
    # Calculate the direction vector from (a, b, c) to (d, e, f)
    direction = np.array([d - a, e - b, f - c])
    
    # Calculate the length of the line segment
    length = np.linalg.norm(direction)
    
    # Normalize the direction vector
    direction_normalized = direction / length
    
    # Calculate the midpoint of the line segment
    midpoint = np.array([(a + d) / 2, (b + e) / 2, (c + f) / 2])
    
    # Calculate the rotation matrix to align the cylinder with the direction vector
    # This involves finding a matrix that rotates the z-axis to the direction vector
    # Here, we use a simple method assuming direction is not aligned with z-axis
    if np.allclose(direction_normalized, [0, 0, 1]) or np.allclose(direction_normalized, [0, 0, -1]):
        rotation_matrix = np.eye(3)
    else:
        v = np.cross([0, 0, 1], direction_normalized)
        s = np.linalg.norm(v)
        c = np.dot([0, 0, 1], direction_normalized)
        vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    
    # Flatten the rotation matrix to pass it as an argument
    rotation_matrix_flatten = rotation_matrix.flatten()

    # Initialize the cylinder (line) in the scene
    # viewer.user_scn.ngeom = 0
    geom_index = viewer.user_scn.ngeom


    # Create a cylinder representing the line segment
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_index],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.array([0.01, length / 2, 1]),  # radius and half-length of the cylinder
        pos=midpoint,  # position at the center of the cylinder
        mat=rotation_matrix_flatten,  # rotation matrix
        rgba=np.array(color, dtype=np.float32)  # color of the cylinder
    )
    # print("midpoint:  ", midpoint)

    # Update the number of geometries
    viewer.user_scn.ngeom += 1



def view(file, fixed=False, pos=None, quat=None, vis_contact=False, joint_pos=None, callback=None):

    import mujoco.viewer

    raise NotImplementedError("This function is not implemented yet.")



def take_photo(conf=None, 
               pipeline=None, 
               save_file=None, 
               pos=None, 
               quat=None, 
               joint_pos=None, 
               fix_pos=False, 
               white_bg=False, 
               render_size=(1000, 1000), 
               cam_distance=2, 
               color_robot=None, 
               lookat=None, 
               azimuth=None, 
               elevation=None,
               color_sphere_only=False,
               shadow=True,
               alphas=None):
    from modular_legs.sim.evolution.utils import update_cfg_with_pipeline
    from modular_legs.sim.scripts.homemade_robots_asym import MESH_DICT_FINE, ROBOT_CFG_AIR1S
    from modular_legs.envs.env_sim import ZeroSim
    from PIL import Image

    if conf is None:
        assert pipeline is not None, "Pipeline must be provided if conf is None"
        if pos is None:
            pos = [0, 0, 0.5]
        if quat is None:
            quat = [1, 0, 0, 0]

        conf = load_cfg()
        conf.agent.obs_version = "robust_proprioception"
        conf = update_cfg_with_pipeline(conf, pipeline, ROBOT_CFG_AIR1S, MESH_DICT_FINE, init_pose_type="original")
        conf.sim.init_pos = pos
        conf.sim.init_quat = quat

        if joint_pos is None:
            joint_pos = [0]*conf.agent.num_act

        conf.agent.default_dof_pos = joint_pos

    elif isinstance(conf, str):
        conf = load_cfg(conf)
    elif isinstance(conf, OmegaConf):
        conf = conf
        
    new_asset_file = None
    old_asset_file = conf.sim.asset_file
    if is_list_like(old_asset_file):
        old_asset_file = old_asset_file[0]

    xc = XMLCompiler(old_asset_file)
    if white_bg:
        conf.sim.randomize_friction = False
        # xc.recolor_floor(["1 1 1", "1 1 1"])
        xc.recolor_sky(["1 1 1", "1 1 1"])
        xc.remove_floor()
        new_asset_file = old_asset_file.replace(".xml", "_white.xml")
        xc.save(new_asset_file)
        conf.sim.asset_file = new_asset_file
    else:
        # xc.recolor_floor([".4 .4 .48", "0.3 .3 0.38"])
        # xc.recolor_sky(["1 1 1", "1 1 1"])
        xc.recolor_floor(["0.7 0.7 0.7", "0.7 0.7 0.7"], mark_color=".7 .7 .7")
        new_asset_file = old_asset_file
        xc.save(old_asset_file)
        conf.sim.asset_file = new_asset_file

    if color_robot is not None:

        xc = XMLCompiler(new_asset_file)
        if alphas is not None:
            colors_new = []
            for color, alpha in zip(color_robot, alphas):
                words = color.split()
                words[-1] = str(alpha)
                color_new = " ".join(words)
                colors_new.append(color_new)
            color_robot = colors_new
            
        xc.recolor_robot(color_robot, sphere_only=color_sphere_only)
        if not shadow:
            # TODO: out of this condition
            xc.remove_shadow()
        new_asset_file = old_asset_file.replace(".xml", "_colored.xml")
        xc.save(new_asset_file)
        conf.sim.asset_file = new_asset_file


    # Update the config
    conf.sim.render = False
    conf.sim.randomize_orientation = False
    conf.robot.mode = "sim"

    conf.sim.render_size = list(render_size)

    env = ZeroSim(conf)
    env.reset()
    # pdb.set_trace() # env.data.qpos[3:7] = [0, 0, 0.5, 1, 0, 0, 0]
    if not fix_pos:
        for _ in range(200):
            env.step(np.zeros(env.num_act))

    viewer = env.mujoco_renderer._get_viewer("rgb_array")
    if lookat is None:
        viewer.cam.lookat = env.last_com_pos # env.pos_world
    else:
        viewer.cam.lookat = lookat
        
    # pdb.set_trace()
    # viewer.cam.lookat[2] -= 0.2
    viewer.cam.distance = cam_distance
    viewer.cam.elevation = -30 if elevation is None else elevation
    viewer.cam.azimuth = -(90+45) if azimuth is None else azimuth
    # pdb.set_trace()
    image = env.render()

    image = Image.fromarray(image)
    if save_file is not None:
        print(f"--> Saving image at {save_file}")
        image.save(save_file)


    return image


def take_video(conf_name, model_file, save_file=None, callback=None, file_name=None):
    rollout(conf_name, model_file, save_file=save_file, callback=callback, take_video=True, file_name=file_name)


def rollout(conf_name, model_file, save_file=None, callback=None, take_video=False, file_name=None):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import pdb
    import sbx
    import sb3_contrib
    import gymnasium as gym
    from modular_legs.scripts.train_sbx import load_model
    from modular_legs.envs.env_sim import ZeroSim
    from modular_legs.envs.gym.rendering import RecordVideo
    import wandb

    wandb.init(mode="disabled")

    if isinstance(conf_name, (OmegaConf, omegaconf.dictconfig.DictConfig)):
        conf = conf_name
    else:
        conf = load_cfg(conf_name, alg="sbx")
    if save_file is None:
        save_file = conf.logging.data_dir

    if conf.agent.command_x_choices is not None:
        print("Found Command-based Task")
        conf.agent.command_x_choices = [0]

    # Update the config
    conf.trainer.device = "cpu"

    unwarpped_env = ZeroSim(conf)
    env = gym.wrappers.TimeLimit(
                unwarpped_env, max_episode_steps=1000
            )
    trigger = lambda t: t == 0
    if take_video:
        print(f"Video will be saved at {save_file}")
        env = RecordVideo(env, 
                        video_folder=save_file, 
                        episode_trigger=trigger, 
                        fps=1/conf.robot.dt,
                        disable_logger=True,
                        full_name=file_name
                        )

    # Setting up the model
    try:
        Alg = getattr(sbx, conf.trainer.algorithm)
        model = load_model(model_file, env, Alg, device="cpu")
    except AttributeError:
        Alg = getattr(sb3_contrib, conf.trainer.algorithm)
        model = load_model(model_file, env, Alg, device="cpu")

    if model_file == "RANDOM":
        model = load_model(model_file, env, Alg, device="cpu",
                           info={"num_act": conf.agent.num_act, 
                                 "act_scale": conf.agent.clip_actions})

    vec_env = model.get_env()

    obs = vec_env.reset()
    step_count = 0
    done = False
    while not done:
        if callback is not None:
            callback(unwarpped_env, obs, 0, done, None)
        action, _states = model.predict(obs, deterministic=True)
        if step_count == 295:
            unwarpped_env.commands[0] = 1
        obs, reward, done, info = vec_env.step(action)
        step_count += 1
    vec_env.close()


def info_to_traj(info, module_idx=0):
    n_episodes = len(info)
    n_steps = len(info[0])
    pos = np.zeros((n_episodes, n_steps, 2))

    if module_idx == 0:
        for ip in range(n_episodes):
            for i in range(n_steps):
                pos[ip][i] = info[ip][i][0]["next_coordinates"]
    else:
        for ip in range(n_episodes):
            for i in range(n_steps):
                pos[ip][i] = info[ip][i][0]["next_coordinates_general"][module_idx]

    pos = pos - pos[:, 0:1, :]
    
    return pos

def draw_2d_traj(loco_file=None, trajectory=None, saved_figure=None, hide_title=False, lim=10, cmap="cool_r", linewidth=2):
    if loco_file is not None:
        loco_dict = np.load(loco_file)

        pos = loco_dict["positions"]
        trajectory = pos[0,:300, :2]

    # if trajectory is not None:
    #     assert trajectory.shape[1] == 2, "Trajectory must have shape (n_steps, 2)"

    if trajectory.ndim == 2:  
        trajectory = trajectory[np.newaxis, ...]  # Convert (n_steps, 2) -> (1, n_steps, 2)

    assert trajectory.shape[2] == 2, "Trajectory must have shape (n_steps, 2)"



    import matplotlib.pyplot as plt
    import matplotlib.collections as mc

    # Generate a sample trajectory (replace this with your actual data)

    # Create line segments
    points = trajectory.reshape(-1, 1, 2)  # Reshape for segment creation
    segments = np.hstack([points[:-1], points[1:]])  # Pair consecutive points

    # Create a colormap based on time
    t = np.linspace(0, 1, len(trajectory))  # Normalize time
    norm = plt.Normalize(t.min(), t.max())
    try:
        cmap = plt.get_cmap(cmap)
    except ValueError:
        import seaborn as sns
        cmap = sns.color_palette(cmap, as_cmap=True)


    # Create line collection
    lc = mc.LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    lc.set_capstyle('round')
    lc.set_array(t[:-1])  # Use time for color mapping
    robot_name = loco_file.split("/")[-1] if loco_file is not None else saved_figure.split("/")[-1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.add_collection(lc)

    for i, traj in enumerate(trajectory):
        # Create line segments
        points = traj.reshape(-1, 1, 2)
        segments = np.hstack([points[:-1], points[1:]])
        
        # Time-based coloring
        t = np.linspace(0, 1, len(traj))
        norm = plt.Normalize(t.min(), t.max())

        # Create LineCollection
        lc = mc.LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=1)
        lc.set_capstyle('round')
        lc.set_array(t[:-1])
        
        ax.add_collection(lc)


    ax.autoscale()
    if not hide_title:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(robot_name)
        plt.colorbar(lc, label="Time")
    else:
        ax.axis("off")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if saved_figure is None and loco_file is not None:
        saved_figure = os.path.join(os.path.dirname(loco_file), f"{robot_name}_2d.pdf")
    else:
        assert saved_figure is not None, "Please provide a saved_figure path"
    if saved_figure.endswith(".pdf"):
        plt.savefig(saved_figure)
    else:
        plt.savefig(saved_figure, dpi=400)
    # plt.show()
