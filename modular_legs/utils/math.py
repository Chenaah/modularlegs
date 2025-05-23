import datetime
import os
import pdb
import numpy as np
# from omegaconf import OmegaConf
import torch
try:
    import jax.numpy as jnp
except ImportError:
    print("JAX is not installed. Please install JAX to use the JAX version of the math functions.")
from modular_legs import LEG_ROOT_DIR


def quat_rotate_inverse_jax(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).reshape(-1, 1)
    b = jnp.cross(q_vec, v, axis=-1) * q_w.reshape(-1, 1) * 2.0
    c = q_vec * jnp.einsum('bi,bi->b', q_vec, v).reshape(-1, 1) * 2.0
    return a - b + c

def quat_rotate_inverse_jax_wxyz(q, v):
    shape = q.shape
    q_w = q[:, 0]  # Extract w component
    q_vec = q[:, 1:]  # Extract x, y, z components
    a = v * (2.0 * q_w ** 2 - 1.0).reshape(-1, 1)
    b = jnp.cross(q_vec, v, axis=-1) * q_w.reshape(-1, 1) * 2.0
    c = q_vec * jnp.einsum('bi,bi->b', q_vec, v).reshape(-1, 1) * 2.0
    return a - b + c


def quat_rotate_inverse_batch(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def quat_rotate_inverse(q, v):
    """
    Rotate a vector in the inverse direction of a given quaternion.

    Parameters:
    q (list): A list representing the quaternion [x, y, z, w].
    v (list): A list representing the vector [x, y, z].

    Returns:
    list: The rotated vector.

    """
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                                [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    return rotation_matrix

def calculate_angular_velocity(initial_quaternion, final_quaternion, dt):
    """
    Calculate the angular velocity from two quaternions and the time difference between them.
    """
    # Convert quaternions to rotation matrices
    R1 = quaternion_to_rotation_matrix(initial_quaternion)
    R2 = quaternion_to_rotation_matrix(final_quaternion)

    # Compute the rotation matrix derivative
    dR = (R2 - R1) / dt

    # Extract the skew-symmetric part to get the angular velocity vector
    angular_velocity = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])

    return angular_velocity

class AverageFilter():
    def __init__(self, windows):
        self.windows = windows
        self.buffer = []

    def reset_windows(self, windows):
        self.windows = windows
        
    def __call__(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.windows:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)


def velocity_transform_yaw(V_world, quaternion):
    """
    Transform velocity from world frame to body frame using yaw angle
    Inputs:
        V_world: 3x1 array representing velocity in the world frame [vx, vy, vz]
        quaternion: 4x1 array representing the orientation of the robot in quaternion [w, x, y, z]
    Returns:
        V_body: 3x1 array representing velocity in the body frame [vx_body, vy_body, vz_body]
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)
    
    # Extract the elements corresponding to the yaw angle
    R_yaw = R[:2, :2]  # Extract the top-left 2x2 submatrix (yaw rotation)

    
    # Transform velocity from world frame to body frame (considering only yaw angle)
    V_body = np.dot(R_yaw, V_world.reshape(2, 1))

    
    return V_body


def world_to_body_velocity_yaw(world_vel, quat):
    """
    Transform velocity from world frame to body frame, dependent only on yaw angle.
    
    Args:
        world_vel (numpy.ndarray): 3D velocity vector in the world frame [vx, vy, vz].
        quat (numpy.ndarray): Quaternion representing the orientation of the body frame [qw, qx, qy, qz].
        
    Returns:
        numpy.ndarray: 3D velocity vector in the body frame [v_body_x, v_body_y, v_body_z], dependent only on yaw.
    """
    # Extract the yaw angle from the quaternion
    quat_norm = quat / np.linalg.norm(quat)
    yaw = np.arctan2(2 * (quat_norm[0] * quat_norm[3] + quat_norm[1] * quat_norm[2]),
                     1 - 2 * (quat_norm[2]**2 + quat_norm[3]**2))
    
    # Create the 3D rotation matrix based on the yaw angle
    rotation_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])
    
    # Apply the rotation matrix to the world velocity vector
    body_vel = np.dot(rotation_mat, world_vel)
    
    return body_vel

def world_velocity_to_forward_body_velocity(quaternion, v_world):
    """
    Convert world velocity to forward body velocity using quaternion representation of orientation.
    
    Parameters:
        quaternion (array-like): Quaternion representing the orientation [w, x, y, z].
        v_world (array-like): World velocity vector [vx, vy, vz].
        
    Returns:
        vx_body (float): Forward body velocity.
    """
    R = quaternion_to_rotation_matrix(quaternion)
    v_world = np.array(v_world)
    v_body = np.dot(R, v_world)
    vx_body = v_body[0]  # Take only the x-component (forward direction)
    return vx_body

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_multiply2(q1, q2):
    # NOTE: The order of the quaternions is wxyz
    # Extract the components of the first quaternion
    w1, x1, y1, z1 = q1
    # Extract the components of the second quaternion
    w2, x2, y2, z2 = q2
    # Calculate the product of the two quaternions
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    # Return the resulting quaternion
    return np.array([w, x, y, z])

def quat_apply_batch(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def quat_apply(a, b):
    xyz = a[:3]
    t = np.cross(xyz, b) * 2
    return (b + a[3:] * t + np.cross(xyz, t))


def construct_quaternion(axis, angle, order="wxyz"):
    axis = np.array(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)  # Normalize axis

    # Calculate the sine and cosine of half the angle
    sin_half_angle = np.sin(angle / 2)
    cos_half_angle = np.cos(angle / 2)
    
    # Construct the quaternion
    w = cos_half_angle
    x = axis[0] * sin_half_angle
    y = axis[1] * sin_half_angle
    z = axis[2] * sin_half_angle
    
    # Return the quaternion
    if order == "wxyz":
        return np.array([w, x, y, z])
    elif order == "xyzw":
        return np.array([x, y, z, w])

def euler_to_quaternion(euler_angles):
    roll, pitch, yaw = euler_angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    """
    Convert a quaternion to Euler angles (yaw, pitch, roll) in radians.
    
    Args:
    q -- A numpy array of shape (4,), representing the quaternion [w, x, y, z]
    
    Returns:
    euler_angles -- A numpy array of shape (3,), representing the Euler angles [yaw, pitch, roll]
    """
    w, x, y, z = q

    # Yaw (Z-axis rotation)
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t0, t1)

    # Pitch (Y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # Clamp value to avoid numerical issues
    pitch = np.arcsin(t2)

    # Roll (X-axis rotation)
    t3 = 2.0 * (w * x + y * z)
    t4 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t3, t4)

    return np.array([yaw, pitch, roll])

def quaternion_to_euler2(quaternion):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    Quaternion should be in the form [w, x, y, z].
    """
    w, x, y, z = quaternion

    # Convert quaternion to rotation matrix
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
    roll = np.arctan2(r21, r22)
    pitch = np.arcsin(-r20)
    yaw = np.arctan2(r10, r00)

    return roll, pitch, yaw

def quat_rotate_batch(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_rotate(q, v):
    q = np.array(q)
    v = np.array(v)
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def ang_vel_to_ang_forward(ang_vel, projected_gravity, theta, alpha, onedir=True, return_y_vec=False):
    # this is for the new module model
    l = 1
    v0 = np.array([0,l*np.cos(theta),l*np.sin(theta)])
    v1 = np.array([l*np.sin(alpha),l*np.cos(alpha),-l*np.sin(theta)])
    # v0 /= np.linalg.norm(v0)
    # v1 /= np.linalg.norm(v1)
    forward_vec = (v0+v1) / np.linalg.norm((v0+v1))
    virtual_y = np.cross(forward_vec, projected_gravity)
    if virtual_y[1] < 0 and onedir:
        virtual_y *= -1

    ang_vel_forward = np.dot(virtual_y, ang_vel)
    if not return_y_vec:
        return ang_vel_forward
    else:
        return ang_vel_forward, virtual_y

def rotate_vector2D(v, theta):
    assert len(v) == 2, "Input vector must be 2D"
    
    # Define the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    
    # Perform the rotation
    return np.dot(rotation_matrix, v)

def xyzw_to_wxyz(q):
    return np.array([q[-1], q[0], q[1], q[2]])

def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def calculate_transformation_matrix(point_A, orientation_A, point_B, orientation_B):
    """
    Calculates the transformation matrix that aligns point_A in A's frame with point_B in B's frame,
    including their orientations.
    Args:
        point_A (numpy.ndarray): Connection point in A's frame (3x1).
        orientation_A (numpy.ndarray): Orientation matrix in A's frame (3x3).
        point_B (numpy.ndarray): Connection point in B's frame (3x1).
        orientation_B (numpy.ndarray): Orientation matrix in B's frame (3x3).

    Returns:
        numpy.ndarray: Transformation matrix (4x4).
    """
    # Transformation matrix for A's frame
    T_A = np.eye(4)
    T_A[:3, :3] = orientation_A
    T_A[:3, 3] = np.array(point_A)

    # Transformation matrix for B's frame
    T_B = np.eye(4)
    T_B[:3, :3] = orientation_B
    T_B[:3, 3] = np.array(point_B)

    # Invert the transformation matrix of part B to get the transformation from B to the connection point
    T_B_inv = np.linalg.inv(T_B)

    # The transformation matrix from A's frame to B's frame
    T_A_B = np.dot(T_A, T_B_inv)
    return T_A_B

def transform_point(T, point):
    """
    Transforms a point using a given transformation matrix.
    Args:
        T (numpy.ndarray): Transformation matrix (4x4).
        point (numpy.ndarray): Point to be transformed (3x1).

    Returns:
        numpy.ndarray: Transformed point (3x1).
    """
    point_homogeneous = np.append(point, 1)  # Convert to homogeneous coordinates
    transformed_point_homogeneous = np.dot(T, point_homogeneous)
    return transformed_point_homogeneous[:3]  # Convert back to Cartesian coordinates

def rotation_matrix(axis, angle):
    # Normalize the axis vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    u_x, u_y, u_z = axis
    
    # Compute the components of the rotation matrix
    cos_alpha = np.cos(angle)
    sin_alpha = np.sin(angle)
    one_minus_cos = 1 - cos_alpha
    
    R = np.array([
        [
            cos_alpha + u_x**2 * one_minus_cos,
            u_x * u_y * one_minus_cos - u_z * sin_alpha,
            u_x * u_z * one_minus_cos + u_y * sin_alpha
        ],
        [
            u_y * u_x * one_minus_cos + u_z * sin_alpha,
            cos_alpha + u_y**2 * one_minus_cos,
            u_y * u_z * one_minus_cos - u_x * sin_alpha
        ],
        [
            u_z * u_x * one_minus_cos - u_y * sin_alpha,
            u_z * u_y * one_minus_cos + u_x * sin_alpha,
            cos_alpha + u_z**2 * one_minus_cos
        ]
    ])
    
    return R

def rotation_matrix_multiply2(R1, R2):
    return np.dot(R2, R1)

def rotation_matrix_sequence(r_list):
    R = np.eye(3)
    for r in r_list:
        R = rotation_matrix_multiply2(R, r)
    return R


def rotation_matrix_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion.
    Args:
        R (numpy.ndarray): Rotation matrix (3x3).

    Returns:
        numpy.ndarray: Quaternion (4,).
    """
    q = np.empty((4,))
    t = np.trace(R)
    if t > 0:
        t = np.sqrt(t + 1.0)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = np.argmax(np.diagonal(R))
        j = (i + 1) % 3
        k = (i + 2) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    return q

def matrix_to_pos_quat(T_A_B):
    origin_B_in_A = transform_point(T_A_B, np.zeros(3))
    R_A_B = T_A_B[:3, :3]
    quaternion_A_B = rotation_matrix_to_quaternion(R_A_B) # [1,0,0,0] # 
    return origin_B_in_A, quaternion_A_B

def normalize_angle(angle):
    return angle % (2 * np.pi)

if __name__ == "__main__":

    # q = quat_rotate_inverse_batch(torch.tensor([[1,2,3,9.1]]), torch.tensor([[2,3,3.]]))
    # print(q)
    # q = quat_rotate_inverse(np.array([1,2,3,9.1]), np.array([2,3,3.]))
    # print(q)

    # q = quat_rotate_batch(torch.tensor([[1,2.6,3,9.1]]), torch.tensor([[2,3.6,3.]]))
    # print(q)
    # q = quat_rotate(np.array([1,2.6,3,9.1]), np.array([2,3.6,3.]))
    # print(q)


    q = torch.tensor([[1,-2,3,9.1]])
    v = torch.tensor([[2,3,3.]])
    result = quat_rotate_inverse_batch(q, v)
    print(result)

    q = jnp.array([[1,-2,3,9.1]])
    v = jnp.array([[2,3,3.]])
    result = quat_rotate_inverse_jax(q, v)
    print(result)

    q = jnp.array([[9.1, 1,-2,3]])
    result = quat_rotate_inverse_jax_wxyz(q, v)
    print(result)

    q = jnp.array([[1.1, 1,-2,3]])
    result = quat_rotate_inverse_jax_wxyz(q, v)
    print(result)


    q = jnp.array([[9.1, 1,-2,3], [1.1, 1,-2,3]])
    v = jnp.array([[2,3,3.]])
    result = quat_rotate_inverse_jax_wxyz(q, v)
    print(result)

