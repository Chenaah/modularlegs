
import os
import copy
import pdb
import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from omegaconf import OmegaConf
from rich.progress import Progress
from scipy.spatial import ConvexHull

from modular_legs.sim.evolution.mutation_meta import random_gen
# from modular_legs.sim.robot_metadesigner import MetaDesigner
from modular_legs.sim.robot_metadesigner import MetaDesignerAsym
from modular_legs.utils.math import euler_to_quaternion, quat_rotate_inverse, quat_rotate_inverse_jax_wxyz, wxyz_to_xyzw
from modular_legs.utils.model import XMLCompiler, get_joint_pos_addr


def batch_split_almost_evenly(lines, points, tolerance=1):
    """
    Checks if each line in a batch approximately splits its corresponding batch of points evenly.

    Parameters:
    - lines: jnp.array of shape (B, 2, 2), where each (2,2) represents a line (two points).
    - points: jnp.array of shape (B, N, 2), where each (N,2) represents N points for a batch.
    - tolerance: Allowed difference in point count between the two sides (default is 1).

    Returns:
    - jnp.array of shape (B,), where each element is True if the line approximately splits the points evenly.
    """

    # Extract line points
    x1, y1 = lines[:, 0, 0], lines[:, 0, 1]  # (B,)
    x2, y2 = lines[:, 1, 0], lines[:, 1, 1]  # (B,)

    # Compute line coefficients: Ax + By + C = 0
    A = y2 - y1  # (B,)
    B = x1 - x2  # (B,)
    C = x2 * y1 - x1 * y2  # (B,)

    # Compute signed distance for each point in batch
    S = A[:, None] * points[..., 0] + B[:, None] * points[..., 1] + C[:, None]  # (B, N)

    # Count points on each side
    N_plus = jnp.sum(S > 0, axis=1)  # (B,)
    N_minus = jnp.sum(S < 0, axis=1)  # (B,)

    # Check if the split is within tolerance
    return jnp.abs(N_plus - N_minus) <= tolerance  # (B,)




def generate_symmetric_list_batch(N, batch_size, key, minval=0, maxval=1, enforce_mixed_signs=False):
    """
    Generate a batch of symmetric lists.
    
    Args:
        N: Length of each list.
        batch_size: Number of lists to generate.
        key: JAX random key.
    
    Returns:
        A batch of symmetric lists with shape (batch_size, N).
    """
    # Generate random absolute values for each pair
    key, subkey = jax.random.split(key)
    absolute_values = jax.random.uniform(subkey, (batch_size, N // 2), minval=minval, maxval=maxval)

    # Generate random signs for each pair
    key, subkey = jax.random.split(key)
    if enforce_mixed_signs:
        # Ensure each pair has one positive and one negative value
        signs = jnp.stack([jnp.ones((batch_size, N // 2)), -jnp.ones((batch_size, N // 2))], axis=-1)
        # Randomly shuffle the signs within each pair
        key, subkey = jax.random.split(key)
        signs = jax.random.permutation(subkey, signs, axis=-1, independent=True)
    else:
        # Allow any combination of signs (both positive, both negative, or mixed)
        signs = jax.random.choice(subkey, jnp.array([-1.0, 1.0]), (batch_size, N // 2, 2))


    # Create symmetric pairs
    symmetric_pairs = absolute_values[..., None] * signs  # Shape: (batch_size, N//2, 2)

    # Reshape to (batch_size, N) for even N
    symmetric_lists = symmetric_pairs.reshape(batch_size, -1)

    # If N is odd, append a 0 to each list
    if N % 2 != 0:
        symmetric_lists = jnp.concatenate(
            [symmetric_lists, jnp.zeros((batch_size, 1))], axis=1
        )

    return symmetric_lists



def is_degenerate(points):
    """Check if the points are degenerate (collinear or duplicate)."""
    # Check if all points are identical
    if np.all(points == points[0]):
        return True
    # Check if points are collinear
    vec1 = points[1] - points[0]
    for i in range(2, len(points)):
        vec2 = points[i] - points[0]
        if np.linalg.norm(np.cross(vec1, vec2)) > 1e-8:
            return False
    return True



def update_cfg_with_draft_asset(cfg, mesh_dict, robot_cfg, replace_asset=False):

    xml = cfg.sim.asset_file
    compiler = XMLCompiler(xml)
    compiler.update_mesh(mesh_dict, robot_cfg)

    new_xml = xml.replace(".xml", "_draft.xml")
    compiler.save(new_xml)

    if replace_asset:
        cfg.sim.asset_file = new_xml
    else:
        cfg.sim.asset_draft = new_xml

    return cfg

def update_cfg_with_optimized_pose(conf, drop_steps=100, move_steps=500, optimization_type="multiply", enable_progress_bar=False, log_dir=None):
    
    assert conf.sim.init_quat == "?", f"init_quat is already set: {conf.sim.init_quat}"
    assert conf.sim.init_pos.startswith("?"), f"init_pos is already set: {conf.sim.init_pos}"
    assert conf.agent.default_dof_pos == "?", f"default_dof_pos is already set: {conf.agent.default_dof_pos}"

    conf = copy.deepcopy(conf)

    if conf.sim.asset_draft is not None:
        xml = conf.sim.asset_draft
    else:
        xml = conf.sim.asset_file

    spine_assumption="spine" in optimization_type
    seed = conf.trainer.seed
    init_pos, init_quat, init_joint, info = _optimize_pose_base(xml, drop_steps, move_steps, optimization_type, enable_progress_bar, spine_assumption, seed, log_dir=log_dir)

    conf.sim.init_quat = init_quat
    if conf.sim.init_pos.startswith("?+"):
        add_h = float(conf.sim.init_pos.split("+")[1])
        init_pos[2] += add_h
    conf.sim.init_pos = init_pos
    conf.agent.default_dof_pos = init_joint
    if conf.agent.forward_vec == "?":
        conf.agent.forward_vec = info["forward_vec"].tolist()
    if conf.agent.projected_upward_vec == "?":
        conf.agent.projected_upward_vec = info["projected_upward"].tolist()
    if conf.agent.projected_forward_vec == "?":
        # untested
        conf.agent.projected_forward_vec = info["projected_forward"].tolist()

    conf.trainer.evolution.pose_score = info["score"] # for logging

    return conf

def optimize_pose(pipeline, drop_steps=100, move_steps=500, optimization_type="multiply"):
    # Old API
    if optimization_type in ["longlegs", "bigbase"]:
        move_steps = 0

    d = MetaDesignerAsym(pipeline, mesh_mode="draft")
    c = XMLCompiler(d.get_xml())
    xml = c.get_string()

    return _optimize_pose_base(xml, drop_steps, move_steps, optimization_type)


def _optimize_pose_base(xml, drop_steps, move_steps, optimization_type="multiply", enable_progress_bar=False, spine_assumption=False, seed=0, log_dir=None):
    # Note that it is assumed that the accuator is position controlled
    if os.path.isfile(xml):
        mj_model = mujoco.MjModel.from_xml_path(xml)
    else:
        mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    joint_geom_ids = [mj_model.geom(f'left{i}').id for i in range(mj_model.nu)] + [mj_model.geom(f'right{i}').id for i in range(mj_model.nu)]
    joint_unique_geom_idx = [mj_model.geom(f'left{i}').id for i in range(mj_model.nu)]
    joint_body_idx = [mj_model.geom(i).bodyid.item() for i in joint_unique_geom_idx]
    info = {}

    ############################################################################################
    # STEP1: Set a random qpos and throw the robot on the ground
    def set_random_qpos(rng):
        fixed_pos = jnp.array([0.0, 0.0, 0.4])
        quaternion = jax.random.uniform(rng, (4,), minval=-1.0, maxval=1.0)
        norm = jnp.linalg.norm(quaternion)
        quaternion = quaternion / norm
        qpos_len = mjx_data.qpos.shape[0] # mjx_data does not have batch dimension yet
        remaining_qpos = jax.random.uniform(rng, (qpos_len - 7,), minval=-jnp.pi, maxval=jnp.pi)
        new_qpos = jnp.concatenate([fixed_pos, quaternion, remaining_qpos])
        new_data = mjx_data.replace(qpos=new_qpos)
        return new_data

    rand_key = jax.random.PRNGKey(seed)
    print("Seed: ", seed)
    rngs = jax.random.split(rand_key, 4096)
    mjx_data = jax.vmap(set_random_qpos, in_axes=0)(rngs)
    if spine_assumption:
        spine_idx = get_joint_pos_addr(mj_model)[0]
        mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[:,spine_idx].set(0)) # Assume the spine is straight
        n_envs, qpos_len = mjx_data.qpos.shape
        remaining_joint_idx = get_joint_pos_addr(mj_model)[1:]
        remaining_joint_pos = generate_symmetric_list_batch(qpos_len-7-1, n_envs, rand_key, minval=0, maxval=1, enforce_mixed_signs=False)
        mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[:,remaining_joint_idx].set(remaining_joint_pos))
        

    joint_pos = mjx_data.qpos[:,get_joint_pos_addr(mj_model)]


    total_steps = drop_steps + move_steps  # Total steps in both loops combined
    progress = Progress()
    task = progress.add_task("[cyan]Optimizing pose...", total=total_steps)
    if enable_progress_bar:
        progress.start()

    last_heights = jnp.copy(mjx_data.qpos[:,2])
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    for i in range(drop_steps):
        # print("Joint Pos: ", mjx_data.qpos[:,7:])
        # mjx_data.ctrl.at[:].set(joint_pos)
        mjx_data = mjx_data.replace(ctrl=joint_pos)
        mjx_data = jit_step(mjx_model, mjx_data)
        # mean_height = mjx_data.geom_xpos.reshape(-1, mjx_model.ngeom, 3).mean(axis=1)[:,2]
        # print(f"[{i}] Height: ", mean_height, " Max: ", mean_height.max())
        progress.advance(task)

        delta_height = mjx_data.qpos[:,2] - last_heights
        last_heights = jnp.copy(mjx_data.qpos[:,2])
        # if jnp.all(delta_height < 1e-3) and i > 20:
        #     progress.advance(task, advance=total_steps-i)
        #     break

    # Set the default pos / quat / joint_pos as the current state
    # Assume the current state is stable and valid
    stable_qpos = mjx_data.qpos.clone()
    # default_joint_pos = mjx_data.qpos[:,get_joint_pos_addr(mj_model)] # Actuation order;  similar to joint_pos?
    default_joint_pos = joint_pos.copy()
    # pdb.set_trace()
    
    joint_body_ids = [mj_model.body(f'l{i}').id for i in range(mj_model.nu)] # position of the left bodies should be the same as the right bodies
    stable_avg_joint_height = mjx_data.xpos[:,joint_body_ids,2].mean(axis=1)
    stable_avg_joint_pos = mjx_data.xpos[:,joint_body_ids,:2].mean(axis=1)
    stable_highest_joint_height = mjx_data.xpos[:,joint_body_ids,2].max(axis=1)
    stable_lowest_joint_height = mjx_data.xpos[:,joint_body_ids,2].min(axis=1)

    contact_geom = mjx_data.contact.geom
    dist = mjx_data.contact.dist
    contact_pos = mjx_data.contact.pos
    contact_floor = jnp.any(contact_geom == 0, axis=2)
    contact_happen = dist < 0.02
    contact_happen_with_floor = contact_floor & contact_happen
    contact_happen_with_floor_bc = jnp.broadcast_to(contact_happen_with_floor[:, :, None], contact_pos.shape)
    floor_contact_pos = jnp.where(contact_happen_with_floor_bc, contact_pos, 0.0)
    contact_pos_sum = jnp.sum(floor_contact_pos, axis=1)
    count_contact = jnp.sum(contact_happen_with_floor, axis=1, keepdims=True)
    count_contact = jnp.maximum(count_contact, 1) # To avoid division by zero
    avg_contact_pos = (contact_pos_sum / count_contact)[:,:2]

    # Calculate the convex hull area
    # if "big" in optimization_type:
    if True:
        areas = []
        for cp, c in zip(contact_pos, contact_happen_with_floor):
            floor_points = cp[c]
            floor_points = np.asarray(floor_points)[:,:2]
            if len(floor_points) < 3 or is_degenerate(floor_points):
                area = 0
            else:
                area = ConvexHull(floor_points).volume
            areas.append(area)
        convex_hull_areas = jnp.array(areas)
    else:
        print("Skipping convex hull area calculation")

    L = jnp.array(joint_geom_ids)
    in_L = jnp.isin(contact_geom, L)
    masked_in_L = in_L * contact_happen_with_floor[..., None]
    any_in_L = jnp.any(masked_in_L, axis=-1)
    joint_touch_floor = jnp.any(any_in_L, axis=-1)

    squared_distances = jnp.sum((stable_avg_joint_pos - avg_contact_pos) ** 2, axis=1)
    com_distances = jnp.sqrt(squared_distances)

    # Calculate the projected vectors
    projected_upward = quat_rotate_inverse_jax_wxyz(stable_qpos[:,3:7], jnp.array([[0,0,1]]))
    
    def quaternion_to_euler2(quaternion):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw).
        Quaternion should be in the form [w, x, y, z].
        """
        w, x, y, z = quaternion
        # Convert quaternion to rotation matrix components
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - x * w)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x * x + y * y)

        # Extract Euler angles
        roll = jnp.arctan2(r21, r22)
        pitch = jnp.arcsin(-r20)
        yaw = jnp.arctan2(r10, r00)

        return roll, pitch, yaw
    # Vectorize the quaternion-to-Euler function
    batched_quaternion_to_euler2 = jax.vmap(quaternion_to_euler2, in_axes=(0,))

    # Function to compute local_upward and local_forward for a batch
    def compute_local_vectors(quaternions):
        euler_angles = batched_quaternion_to_euler2(quaternions)
        rolls = euler_angles[0]  # Extract roll angles (shape: [N,])

        # Compute local_upward and local_forward for each roll
        sin_roll = jnp.sin(rolls)
        cos_roll = jnp.cos(rolls)

        local_upward = jnp.stack([jnp.zeros_like(rolls), sin_roll, cos_roll], axis=-1)  # Shape: [N, 3]
        local_forward = jnp.stack([jnp.zeros_like(rolls), cos_roll, -sin_roll], axis=-1)  # Shape: [N, 3]

        return local_upward, local_forward

    if spine_assumption:
        spine_local_upward, spine_local_forward = compute_local_vectors(stable_qpos[:,3:7])

        quat = mjx_data.qpos[:,3:7]
        gravity_vec = jnp.array([[0, 0, -1]])
        projected_gravity = quat_rotate_inverse_jax_wxyz(quat, gravity_vec)
        static_upward_dot = jnp.einsum('ij,ij->i', spine_local_upward, -projected_gravity)
        # jnp.argmax(static_upward_dot)
        # default_joint_pos[jnp.argmax(static_upward_dot)]


    

    # Generate random lines (B,2,2) and points (B,N,2)
    # Check sysmetric 
    spine_lines = mjx_data.geom_xpos[:,[mj_model.geom('stick0').id, mj_model.geom('stick1').id],:2]
    other_stick_ids = [mj_model.geom(f'stick{i}').id for i in range(2, 2*mj_model.nu)]
    leg_module_pos = mjx_data.geom_xpos[:,other_stick_ids,:2]
    sysmetric = batch_split_almost_evenly(spine_lines, leg_module_pos)


    # print("Local Upward Vectors:\n", local_upward)
    # print("Local Forward Vectors:\n", local_forward)
    # projected_upward sames?
    # pdb.set_trace()


    '''
    forward_euler = quaternion_to_euler2(d.qpos[3:7])
    roll = forward_euler[0]

    local_upward = [0, sin(roll), cos(roll)]
    local_forward = [0.0, cos(roll), -sin(roll)]
    
    '''
    


    ############################################################################################
    # STEP2: Add random noise to the joint position and evaluate the performance
    actions = "sin"
    def add_random_noise(joint_position, rng):
        noise = jax.random.uniform(rng, shape=(joint_position.shape), minval=-1.0, maxval=1.0)
        return joint_position + noise
    acc_speed = jnp.zeros(mjx_data.qpos.shape[0])
    acc_height = jnp.zeros(mjx_data.qpos.shape[0])
    last_com_pos = mjx_data.xpos[:,joint_body_idx,:2].mean(axis=1)
    acc_vel = jnp.zeros(mjx_data.qpos[:,:2].shape)
    acc_projected_vel = jnp.zeros(mjx_data.qpos.shape[0])
    fall_down = jnp.zeros(mjx_data.qpos.shape[0], dtype=bool)
    gravity_vec = jnp.array([[0, 0, -1]])

    for i in range(move_steps):
        if actions == "random":
            rngs = jax.random.split(rand_key, 4096)
            rand_key = rngs[0]
            joint_pos_with_noise = jax.vmap(add_random_noise)(default_joint_pos, rngs)
        elif actions == "sin":
            joint_pos_with_noise = default_joint_pos + jnp.sin(i/10)

        # Step the simulation
        # mjx_data.ctrl.at[:].set(joint_pos_with_noise)
        mjx_data = mjx_data.replace(ctrl=joint_pos_with_noise)
        # print(f"[{i}] DESIRED POS: ", mjx_data.ctrl)
        mjx_data = jit_step(mjx_model, mjx_data)
        # speed = jnp.linalg.norm(mjx_data.qvel[:,:2], axis=1)

        # Measure global COM velocity
        com_pos = mjx_data.xpos[:,joint_body_idx,:2].mean(axis=1) # 2D COM
        com_vel = (com_pos - last_com_pos) / mj_model.opt.timestep
        last_com_pos = com_pos.copy()
        # print("Avg vel: ", com_vel)
        acc_vel += com_vel

        # Measure local spine velocity
        if spine_assumption:
            vel_body = mjx_data.qvel[:,3:6]
            projected_forward_vel = jnp.einsum('ij,ij->i', spine_local_forward, vel_body)
            acc_projected_vel += projected_forward_vel


        # Check if the robots fall down
        quat = mjx_data.qpos[:,3:7]
        projected_gravity = quat_rotate_inverse_jax_wxyz(quat, gravity_vec)
        if not spine_assumption:
            # Use the projected upward vector
            dot_results = jnp.einsum('ij,ij->i', projected_upward, -projected_gravity)
            fall = dot_results < 0.1
        else:

            dot_results = jnp.einsum('ij,ij->i', spine_local_upward, -projected_gravity)
            wall_fall = dot_results < 0.1

            # Use the contact points
            contact_geom = mjx_data.contact.geom
            dist = mjx_data.contact.dist
            contact_pos = mjx_data.contact.pos
            contact_floor = jnp.any(contact_geom == 0, axis=2)
            contact_happen = dist < 0.02
            contact_happen_with_floor = contact_floor & contact_happen
            torso_geom_ids = [mj_model.geom(f'left0').id, mj_model.geom(f'right0').id]
            L = jnp.array(torso_geom_ids)
            in_L = jnp.isin(contact_geom, L)
            masked_in_L = in_L * contact_happen_with_floor[..., None]
            any_in_L = jnp.any(masked_in_L, axis=-1)
            joint_touch_floor = jnp.any(any_in_L, axis=-1)
            fall = jnp.logical_or(joint_touch_floor, wall_fall)
            # print("Joint Touch Floor: ", fall)



        fall_down = jnp.logical_or(fall_down, fall)
        # print("Fall Down: ", fall_down)

        # acc_speed += speed
        # mean_height = mjx_data.geom_xpos.reshape(-1, mjx_model.ngeom, 3).mean(axis=1)[:,2]
        # acc_height += mean_height
        # print(f"[{i}] Height: ", mean_height, " Max: ", mean_height.max(), "Speed: ", acc_speed)
        progress.advance(task)

    # avg_speed = acc_speed / move_steps
    # avg_height = acc_height / move_steps
    avg_vel = acc_vel / move_steps
    avg_speed = jnp.linalg.norm(avg_vel, axis=1)
    avg_projected_vel = acc_projected_vel / move_steps

    ############################################################################################


    if optimization_type == "multiply":
        final_score = avg_speed*avg_height
    elif optimization_type == "longlegs":
        final_score = stable_avg_joint_height
        final_score = jnp.where(com_distances<0.1, final_score, -999)
        final_score = jnp.where(jnp.sqrt(convex_hull_areas)<stable_highest_joint_height, final_score, -999)
        if move_steps != 0:
            print("Warning: move_steps should be 0 for longlegs optimization")
    elif optimization_type == "bigbase":
        print("Big Base Optimization!")
        final_score = convex_hull_areas*stable_lowest_joint_height
        final_score = jnp.where(jnp.sqrt(convex_hull_areas)>stable_highest_joint_height, final_score, final_score-999)
        final_score = jnp.where(joint_touch_floor, final_score-999, final_score)
        if move_steps != 0:
            print("Warning: move_steps should be 0 for longlegs optimization")
    elif optimization_type == "fastbigbase":
        print("Fast Big Base Optimization!")
        final_score = convex_hull_areas*stable_lowest_joint_height + avg_speed
        final_score = jnp.where(jnp.sqrt(convex_hull_areas)>stable_highest_joint_height, final_score, final_score-100)
        final_score = jnp.where(joint_touch_floor, final_score-100, final_score)
        final_score = jnp.where(fall_down, final_score-100, final_score)
        if move_steps == 0:
            print("Warning: move_steps should be non-0 for longlegs optimization")
    elif optimization_type == "stablefast":
        # print("StableFast Optimization!")
        final_score = avg_speed
        final_score = jnp.where(fall_down, final_score-100, final_score)
    elif optimization_type == "stablefastair":
        # Penalize the ball touching the ground
        final_score = avg_speed
        final_score = jnp.where(fall_down, final_score-100, final_score)
        final_score = jnp.where(joint_touch_floor, final_score-100, final_score)
    elif optimization_type == "stablefastspine":
        final_score = avg_projected_vel + 0.2*static_upward_dot
        # stable_qpos[3453][3:7]
        # spine_local_upward[3453]
            # spine_local_upward
        
        # final_idx = jnp.argmax(jnp.where(fall_down, avg_projected_vel-100, avg_projected_vel))
        # default_joint_pos[final_idx]
        # default_joint_pos[jnp.argmax(test_score)]
        # jnp.argmax(final_score)
        final_score = jnp.where(fall_down, final_score-100, final_score)
    
    elif optimization_type == "stablespine":
        final_score = convex_hull_areas
        final_score = jnp.where(fall_down, final_score-100, final_score)
        # final_idx = jnp.argmax(jnp.where(fall_down, convex_hull_areas-100, convex_hull_areas))
        # default_joint_pos[jnp.where(fall_down, final_score-100, final_score) > 0]
        # jnp.where(jnp.where(fall_down, final_score-100, final_score) > 0)[0
        # final_score[2249]
        # convex_hull_areas[3493]
        # default_joint_pos[2249]
    elif optimization_type == "sysmetricstablespine":
        final_score = convex_hull_areas
        final_score = jnp.where(fall_down, final_score-100, final_score)
        final_score = jnp.where(sysmetric, final_score, final_score-100)

    elif optimization_type == "sysmetricstablespineair":
        final_score = convex_hull_areas
        final_score = jnp.where(fall_down, final_score-100, final_score)
        final_score = jnp.where(joint_touch_floor, final_score-100, final_score)
        final_score = jnp.where(sysmetric, final_score, final_score-100)


    # print("Final Score: ", final_score)
    # print("Max Score: ", final_score.max())


    idx = jnp.argmax(final_score)
    best_stable_qpos = stable_qpos[idx]
    init_pos = [0,0,best_stable_qpos[2].item()+0.01]
    init_quat = best_stable_qpos[3:7].tolist()
    init_joint = default_joint_pos[idx].tolist()
    forward_vec = jnp.append(avg_vel[idx], 0)
    forward_vec = forward_vec / jnp.linalg.norm(forward_vec)
    info["forward_vec"] =  forward_vec
    if not spine_assumption:
        info["projected_upward"] = projected_upward[idx]
        projected_forward = quat_rotate_inverse_jax_wxyz(stable_qpos[:,3:7], jnp.expand_dims(forward_vec, 0))[idx]
        info["projected_forward"] = projected_forward
    else:
        info["projected_forward"] = spine_local_forward[idx]
        info["projected_upward"] = spine_local_upward[idx]
    info["score"] = final_score[idx].item()

    if log_dir is not None:
        np.savez_compressed(os.path.join(log_dir, "poses.npz"), stable_qpos=stable_qpos, default_joint_pos=default_joint_pos, score=final_score, best_stable_qpos=best_stable_qpos, best_init_pos=init_pos, best_init_quat=init_quat, best_init_joint=init_joint, best_forward_vec=forward_vec, best_projected_upward=info["projected_upward"], best_projected_forward=info["projected_forward"])


    # pdb.set_trace()

    if enable_progress_bar:
        progress.stop()

    # print("=======================")
    # print("Design: ", p)
    # print("=======================")
    # print("Best Stable Qpos: ", best_stable_qpos.tolist())

    # print("=======================")
    # print("Init Pos: ", init_pos)
    # print("Init Quat: ", init_quat)
    # print("Init Joint: ", init_joint)

    return init_pos, init_quat, init_joint, info


def get_local_vectors(pipeline, init_pos, init_quat, init_joint, render=False):
    # Get the local vectors for reward / done calculation
    d = MetaDesignerAsym(pipeline, mesh_mode="default")
    c = XMLCompiler(d.get_xml())
    xml = c.get_string()

    # m = mujoco.MjModel.from_xml_path(file)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    d.qpos[0:3] = init_pos
    d.qpos[3:7] = init_quat
    d.qpos[get_joint_pos_addr(m)] = init_joint

    # Vectors in the global frame
    forward_vec = [1., 0, 0] # This can be an arbitrary vector as we require the robot to be omnidirectional
    updir_vec = [0, 0, 1]

    if render:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.__enter__()
    else:
        viewer = None

    n_ctr = 0
    last_projected_forward = np.zeros(3)
    last_projected_upward = np.zeros(3)


    for t in range(100):
        step_start = time.time()

        d.ctrl[:] = init_joint
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2) if vis_contact else 0
            # viewer.opt.frame = 1 # visulize axis
        # pdb.set_trace()
        n_self_collision = 0
        # for contact in d.contact:
        #     if contact.geom1 != 0 and contact.geom2 != 0:
        #         b1 = m.body(m.geom(contact.geom1).bodyid).name
        #         b2 = m.body(m.geom(contact.geom2).bodyid).name
        #         if not (((b1[0] == "l" and b2[0] == "r") or (b1[0] == "r" and b2[0] == "l")) and (b1[1] == b2[1])):
        #             n_self_collision += 1
        # print(n_self_collision)
        quat = d.qpos[3:7]
        projected_forward = quat_rotate_inverse(wxyz_to_xyzw(quat), np.array(forward_vec))
        # print("Projected Forward: ", projected_forward)
        projected_upward = quat_rotate_inverse(wxyz_to_xyzw(quat), np.array(updir_vec))
        # print("Projected Upward: ", projected_upward)

        if t > 10 and np.mean(projected_forward - last_projected_forward) < 0.001 and np.mean(projected_upward - last_projected_upward) < 0.001:
            break

        # quat = d.qpos[3:7]
        # accurate_projected_gravity = quat_rotate_inverse(wxyz_to_xyzw(quat), np.array([0, 0, -1]))
        # print("DOT: ", np.dot(np.array([np.sin(np.pi/12), 0, np.cos(np.pi/12)]), -accurate_projected_gravity))
        

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        if render:
            viewer.sync()


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0 and render:
            time.sleep(time_until_next_step)
        
        n_ctr += 1
    
    if render:    
        viewer.__exit__()

    return projected_forward, projected_upward





if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    p = random_gen(6, None)
    # init_pos, init_quat, init_joint = optimize_pose(p, 100, 0)
    # init_pos, init_quat, init_joint = optimize_pose(p, 150, 0, "longlegs")
    init_pos, init_quat, init_joint, _ = optimize_pose(p, 150, 250, "fastbigbase")
    print("Design: ", p)
    print("Init Pos: ", init_pos)
    print("Init Quat: ", init_quat)
    print("Init Joint: ", init_joint)

    # get_local_vectors([0, 0, 0, 2, 0, 0, 1, 3, 8, 23, 0, 1, 6, 7, 8, 0],
    #                   init_pos=[0,0,0.1825061684846878],
    #                   init_quat=[0.04657600820064545, -0.8202893137931824, 0.5689055323600769, 0.03609202802181244],
    #                   init_joint=[2.2334494590759277, 0.031052296981215477, 0.08712831884622574, 0.562721312046051])

    pdb.set_trace()