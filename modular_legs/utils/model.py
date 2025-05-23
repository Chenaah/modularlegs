



import os
from pathlib import Path
import pdb
import time
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
import numpy as np
from numpy import cos, sin 
import mujoco
from lxml import etree

from modular_legs import LEG_ROOT_DIR

def get_joint_pos_addr(model):
    '''
    Returns the joint position address of the robot.
    Joint name is assumed to be 'joint{i}' where i is the joint index.
    d.joint('foo').qpos also works.
    '''
    joint_idx = [model.joint(f'joint{i}').id for i in range(model.nu)]
    return model.jnt_qposadr[joint_idx]




def euler_angles_from_vectors(v1, v2):
    u = v1 / np.linalg.norm(v1)
    v = v2 / np.linalg.norm(v2)
    
    # Calculate the rotation axis
    axis = np.cross(u, v)
    
    # Calculate the rotation angle
    angle = np.arccos(np.dot(u, v))
    
    # Convert the rotation to Euler angles
    if np.allclose(axis, [0, 0, 0]):  # If rotation axis is zero (no rotation)
        return [0, 0, 0]
    else:
        rotation = R.from_rotvec(angle * axis)
        euler = rotation.as_euler('xyz', degrees=True)  # Adjust the rotation sequence as needed
        return euler

def create_color_spectrum(colormap_name='plasma', num_colors=100):
    """
    Create a color spectrum using Matplotlib colormap.

    Args:
        colormap_name (str): Name of the Matplotlib colormap.
        num_colors (int): Number of colors in the spectrum.

    Returns:
        List of RGB tuples representing the color spectrum.
    """
    colormap = get_cmap(colormap_name)
    colors = [colormap(i/num_colors) for i in range(num_colors)]
    return colors


def quat_rotate(q, v, order='xyzw'):
    if order == 'wxyz':
        q = np.array([q[1], q[2], q[3], q[0]])
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c

def quaternion_from_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Ensure input vectors are normalized
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate rotation axis
    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)  # Normalize axis

    # Calculate rotation angle
    angle = np.arccos(np.dot(v1, v2))

    # Construct quaternion
    w = np.cos(angle / 2)
    x, y, z = np.sin(angle / 2) * axis

    return np.array([w, x, y, z])

def vec2string(arr):
    if not isinstance(arr, str):
        arr = np.array(arr)
        pos_string = ' '.join(arr.astype(str))
    else:
        pos_string = arr
    return pos_string

def tree2string(tree):
    xml_string = ET.tostring(tree, encoding="utf-8", xml_declaration=False).decode()
    dom = xml.dom.minidom.parseString(xml_string)
    xml_string = dom.childNodes[0].toprettyxml(indent="  ")
    return xml_string


def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to a 3D rotation matrix.
    
    Parameters:
        euler_angles (array_like): Euler angles in the order of rotation (e.g., [roll, pitch, yaw]).
        
    Returns:
        rotation_matrix (ndarray): 3x3 rotation matrix.
    """
    roll, pitch, yaw = euler_angles
    
    # Convert Euler angles to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Calculate trigonometric values
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    # Calculate the rotation matrix
    rotation_matrix = np.array([[cp*cy, cp*sy, -sp],
                                [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
                                [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]])
    
    return rotation_matrix

def random_point_at_distance(origin, d):
    # Generate a random direction.
    direction = np.random.randn(3)  # a random vector
    direction /= np.linalg.norm(direction)  # normalize to make it a unit vector

    # Scale the direction by the desired distance.
    vector = d * direction

    # Add the original point to the vector to get the final point.
    return origin + vector


def lighten_color(rgb, factor):
    """
    Lightens an RGB color by blending it with white (normalized values).
    
    :param rgb: A tuple containing the RGB values (r, g, b) where each value is between 0 and 1.
    :param factor: A float between 0 and 1 representing the blend factor. 
                   0 returns the original color, 1 returns white.
    :return: A tuple containing the lightened RGB values.
    """
    # Ensure the blend factor is within the valid range
    if not (0 <= factor <= 1):
        raise ValueError("Factor must be between 0 and 1")

    # Calculate the new RGB values
    r, g, b = rgb[0], rgb[1], rgb[2]
    new_r = r + (1 - r) * factor
    new_g = g + (1 - g) * factor
    new_b = b + (1 - b) * factor

    # Ensure the RGB values are within the valid range
    new_r = min(1, max(0, new_r))
    new_g = min(1, max(0, new_g))
    new_b = min(1, max(0, new_b))

    return [new_r, new_g, new_b]


def get_jing_vector(alpha, theta):
    if alpha != 0:
        alpha = 0.00001 if alpha < 0.00001 and alpha > 0 else alpha
        alpha = -0.00001 if alpha > -0.00001 and alpha < 0 else alpha
        jing_vec = np.array([
            (cos(alpha)-1)/sin(theta),
            1,
            ((1-cos(alpha))**2 + sin(theta)*sin(alpha)) / (cos(theta)*sin(alpha))
        ])
    else:
        jing_vec = np.array([
            0,
            cos(theta),
            sin(theta)
        ])
    return jing_vec / np.linalg.norm(jing_vec)


def get_local_zvec(alpha, theta):
    jing_vec = get_jing_vector(alpha, theta)
    mid_vec = np.array([cos(theta)*sin(theta), cos(theta)*(1-cos(alpha)), 0])
    local_zvec = np.cross(jing_vec, mid_vec)
    return local_zvec / np.linalg.norm(local_zvec)


def fix_model_file_path(root):
    parts_dir = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "parts")
    mesh_files = [file.name for file in Path(parts_dir).rglob('*') if file.is_file()]
    mesh_file_paths = [str(file.resolve()) for file in Path(parts_dir).rglob('*') if file.is_file()]
    for mesh in root.findall('.//mesh') + root.findall('.//hfield'):
        if mesh.get('file') in mesh_files:
            mesh.set('file', mesh_file_paths[mesh_files.index(mesh.get('file'))])
            # print(f"Fixed mesh file path: {mesh.get('file')}")
    return root

def position_to_torque_control(root):
    for position in root.findall('.//position'):
        position.tag = 'motor'
        del position.attrib['kp']
        del position.attrib['kv']
        position.attrib['ctrlrange'] = position.attrib['forcerange']
        del position.attrib['forcerange']
    return root

def torque_to_position_control(root, kp=20, kd=0.5):
    for motor in root.findall('.//motor'):
        motor.tag = 'position'
        motor.attrib['kp'] = f"{kp}"
        motor.attrib['kv'] = f"{kd}"
        motor.attrib['forcerange'] = motor.attrib['ctrlrange']
        del motor.attrib['ctrlrange']
    return root

def update_xml_timestep(root, timestep):
    option = root.xpath('//option[@integrator="RK4"]')[0]
    option.attrib['timestep'] = f"{timestep}"
    return root

def compile_xml(xml_file, torque_control=False, timestep=None):
    # Load XML file, fix mesh file path, do other convertion, and return the XML string

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_file, parser)
    root = tree.getroot()

    # Fix mesh file path
    root = fix_model_file_path(root)
    # Convert position control to torque control
    if torque_control:
        root = position_to_torque_control(root)
    # Update timestep
    if timestep is not None:
        root = update_xml_timestep(root, timestep)

    xml_string = etree.tostring(root, pretty_print=True, xml_declaration=False, encoding='utf-8').decode()
    return xml_string

class XMLCompiler:
    def __init__(self, xml_file):
        parser = etree.XMLParser(remove_blank_text=True)
        # pdb.set_trace()
        if xml_file.endswith('.xml'):
            if not os.path.exists(xml_file):
                xml_file = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", xml_file)
            tree = etree.parse(xml_file, parser)
            self.root = tree.getroot()
        else:
            self.root = etree.fromstring(xml_file, parser)
        
        # Fix mesh file path
        self.root = fix_model_file_path(self.root)

    def torque_control(self):
        self.root = position_to_torque_control(self.root)

    def position_control(self, kp=20, kd=0.5):
        self.root = torque_to_position_control(self.root, kp=20, kd=0.5)

    def pyramidal_cone(self):
        option_element = self.root.find('.//option')
        if option_element is not None:
            option_element.set('cone', 'pyramidal')

    def update_timestep(self, timestep):
        self.root = update_xml_timestep(self.root, timestep)

    def get_string(self):
        xml_string = etree.tostring(self.root, pretty_print=True, xml_declaration=False, encoding='utf-8').decode()
        # pdb.set_trace()
        # DEBUG:
        # self.save(os.path.join(os.path.dirname(__file__), "test.xml"))
        return xml_string
    
    def save(self, file):
        tree = etree.ElementTree(self.root)
        tree.write(file, pretty_print=True, xml_declaration=False, encoding='utf-8')
    
    def get_mass_range(self, percentage=0.1):
        # Get the mass range of the stick, top_hemi, bottom_hemi, and motor
        mass_range = {}
        left_mass = self.root.xpath('//geom[@name="left0"]/@mass')[0]
        mass_range['left'] = [float(left_mass) * (1 - percentage), float(left_mass) * (1 + percentage)]
        right_mass = self.root.xpath('//geom[@name="right0"]/@mass')[0]
        mass_range['right'] = [float(right_mass) * (1 - percentage), float(right_mass) * (1 + percentage)]
        stick_mass = self.root.xpath('//geom[@name="stick0"]/@mass')[0]
        mass_range['stick'] = [float(stick_mass) * (1 - percentage), float(stick_mass) * (1 + percentage)]

        motor_list = self.root.xpath('//geom[@name="motor0"]/@mass')
        if motor_list:
            motor_mass = motor_list[0]
            mass_range['motor'] = [float(motor_mass) * (1 - percentage), float(motor_mass) * (1 + percentage)]
        battery_list = self.root.xpath('//geom[@name="battery0"]/@mass')
        if battery_list:
            battery_mass = battery_list[0]
            mass_range['battery'] = [float(battery_mass) * (1 - percentage), float(battery_mass) * (1 + percentage)]
        pcb_list = self.root.xpath('//geom[@name="pcb0"]/@mass')
        if pcb_list:
            pcb_mass = pcb_list[0]
            mass_range['pcb'] = [float(pcb_mass) * (1 - percentage), float(pcb_mass) * (1 + percentage)]

        return mass_range
    
    def update_mass(self, mass_dict):
        for key, mass in mass_dict.items():
            for geom in self.root.xpath(f'//geom[starts-with(@name, {key})]'):
                geom.set('mass', str(mass))

    def update_damping(self, armature, damping):
        # joints = self.root.find('default').findall('joint')
        # for joint in joints:
        #     joint.set('armature', str(armature))
        #     joint.set('damping', str(damping))
        for joint in self.root.xpath(f'//joint[starts-with(@name, "joint")]'):
            joint.set('armature', str(armature))
            joint.set('damping', str(damping))
            # print(f"Updated damping and armature of {joint.get('name')} to {damping} and {armature}")

    def add_walls(self, transparent=False, angle=0):
        world_body = self.root.findall("./worldbody")[0]
        wall = etree.SubElement(world_body, "body", name=f"boundary", pos="0 0 0", axisangle=f"0 0 1 {angle}")
        etree.SubElement(wall, "geom", name=f"boundary/right", pos="0 1 0.25", type="box", material="boundary", size="25 0.1 0.5", **({} if not transparent else {'rgba': "0.1 0.1 0.1 0.0"}))
        etree.SubElement(wall, "geom", name=f"boundary/left", pos="0 -1 0.25", type="box", material="boundary", size="25 0.1 0.5", **({} if not transparent else {'rgba': "0.1 0.1 0.1 0.0"}))

    def remove_walls(self):
        world_body = self.root.findall("./worldbody")[0]
        for wall in world_body.findall("./body[@name='boundary']"):
            world_body.remove(wall)

    def reset_hfield(self, radius_x, radius_y, elevation_z, base_z, hfield_file=None, size=None):
        '''
        If hfield_file is None, model.hfield_data should be provided.
        '''
        if hfield_file is not None:
            hfield_file = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "parts", hfield_file)
        hfields = self.root.findall('.//hfield')
        for hfield in hfields:
            if hfield_file is not None:
                hfield.set('file', hfield_file)
            else:
                hfield.attrib.pop('file', None)
            hfield.set('size', f"{radius_x} {radius_y} {elevation_z} {base_z}")
        if not hfields:
            world_body = self.root.findall("./worldbody")[0]
            floor_elem = world_body.find("geom[@name='floor']")
            assert floor_elem is not None, "No floor element found in the XML file."
            world_body.remove(floor_elem)

            assets = self.root.findall('.//asset')[0]
            etree.SubElement(world_body, "geom", name="floor", pos="0 0 0", type="hfield", material="hfield", conaffinity="1", condim="6", friction="1.0 .0 .0", hfield="rugged")
            if hfield_file is not None:
                etree.SubElement(assets, "hfield", name="rugged", size=f"{radius_x} {radius_y} {elevation_z} {base_z}", file=hfield_file)
            else:
                assert size is not None, "Size of the hfield is not provided."
                etree.SubElement(assets, "hfield", name="rugged", size=f"{radius_x} {radius_y} {elevation_z} {base_z}", nrow=f"{size}", ncol=f"{size}") 


    def reset_obstacles(self, terrain_params):
        world_body = self.root.findall("./worldbody")[0]
        elements = [geom for geom in world_body.findall("./geom") if geom.get("name", "").startswith("obstacle")]
        # assert elements, "No obstacle found in the old model."
        if not elements:
            num_bumps = terrain_params["num_bumps"] if "num_bumps" in terrain_params else 200
            for i in range(num_bumps):
                pos = np.random.uniform(-5, 5, 2)
                angle = np.random.uniform(0, 360)
                height = np.random.uniform(*terrain_params["height_range"])
                width = np.random.uniform(*terrain_params["width_range"])
                # self.builder.add_minibump(pos=pos, angle=angle, height=height, width=width, length=1)
                length = 1
                etree.SubElement(world_body, "geom", name=f"obstacle{i}", pos=f"{pos[0]} {pos[1]} {height/2}", axisangle=f"0 0 1 {angle}", type="box", material="boundary", size=f"{width/2} {length/2} {height/2}")
        else:
            for obstacle in elements:
                pos = np.random.uniform(-5, 5, 2)
                angle = np.random.uniform(0, 360)
                height = np.random.uniform(*terrain_params["height_range"])
                width = np.random.uniform(*terrain_params["width_range"])
                length = 1

                pos=f"{pos[0]} {pos[1]} {height/2}"
                axisangle=f"0 0 1 {angle}"
                size=f"{width/2} {length/2} {height/2}"
            
                obstacle.set('pos', pos)
                obstacle.set('axisangle', axisangle)
                obstacle.set('size', size)

    def update_mesh(self, mesh_dict, robot_cfg=None):
        # Update all the geom mesh in the model
        world_body = self.root.findall("./worldbody")[0]
        assets = self.root.findall('.//asset')[0]

        # Import the mesh files
        for key, mesh_file in mesh_dict.items():
            if mesh_file.endswith('.obj') or mesh_file.endswith('.stl'):
                meshes = assets.findall(f"./mesh[@name='{key}']")
                for mesh in meshes:
                    assets.remove(mesh)
                etree.SubElement(assets, "mesh", file=mesh_file, name=key, scale="1 1 1")


        if "up" in mesh_dict:
            lefts = [geom for geom in world_body.findall(".//geom") if geom.get("name", "").startswith("left")]
            for left_geom in lefts:
                name = left_geom.get("name")
                color = left_geom.get("rgba")
                mass = left_geom.get("mass")
                parent = left_geom.getparent()
                parent.remove(left_geom)  # Remove the old geom from its parent
                if mesh_dict["up"].endswith('.obj') or mesh_dict["up"].endswith('.stl'):
                    etree.SubElement(parent, "geom", type="mesh", name=name, mesh="up", rgba=color, mass=mass, material="metallic", friction="1.0 .0 .0", priority="2")
                elif mesh_dict["up"] == "SPHERE":
                    # Draft mode: Use sphere instead of mesh
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["R"]
                    etree.SubElement(parent, "geom", type="sphere", name=name, size=f"{radius}", rgba=color, mass=mass, friction="1.0 .0 .0", priority="2")
                else:
                    raise ValueError("The mesh should be either a .obj file or a SPHERE")

        if "bottom" in mesh_dict:
            rights = [geom for geom in world_body.findall(".//geom") if geom.get("name", "").startswith("right")]
            for right_geom in rights:
                name = right_geom.get("name")
                color = right_geom.get("rgba")
                mass = right_geom.get("mass")
                parent = right_geom.getparent()
                parent.remove(right_geom)

                if mesh_dict["bottom"].endswith('.obj') or mesh_dict["bottom"].endswith('.stl'):
                    etree.SubElement(parent, "geom", type="mesh", name=name, mesh="bottom", rgba=color, mass=mass, material="metallic", friction="1.0 .0 .0", priority="2")
                elif mesh_dict["bottom"] == "SPHERE":
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["R"]
                    etree.SubElement(parent, "geom", type="sphere", name=name, size=f"{radius}", rgba=color, mass=mass, friction="1.0 .0 .0", priority="2")
                else:
                    raise ValueError("The mesh should be either a .obj file or a SPHERE")
                
        if "battery" in mesh_dict:
            batteries = [geom for geom in world_body.findall(".//geom") if geom.get("name", "").startswith("battery")]
            assert batteries, "No battery found in the old model." # TODO
            for battery in batteries:
                name = battery.get("name")
                color = battery.get("rgba")
                mass = battery.get("mass")
                parent = battery.getparent()
                parent.remove(battery)

                if mesh_dict["battery"].endswith(".obj") or mesh_dict["battery"].endswith(".stl"):
                    etree.SubElement(parent, "geom", type="mesh", name=name, mesh="battery", rgba=color, mass=mass, material="metallic", contype="10", conaffinity="0")
                elif mesh_dict["battery"] == "NONE":
                    pass
                else:
                    raise ValueError("The mesh should be either a .obj file or NONE")

        if "pcb" in mesh_dict:
            pcbs = [geom for geom in world_body.findall(".//geom") if geom.get("name", "").startswith("pcb")]
            assert pcbs, "No PCB found in the old model." # TODO
            for pcb in pcbs:
                name = pcb.get("name")
                color = pcb.get("rgba")
                mass = pcb.get("mass")
                parent = pcb.getparent()
                parent.remove(pcb)

                if mesh_dict["pcb"].endswith(".obj") or mesh_dict["pcb"].endswith(".stl"):
                    etree.SubElement(parent, "geom", type="mesh", name=name, mesh="pcb", rgba=f"0 0 0 0.5", mass=mass, material="metallic", contype="10", conaffinity="0")
                elif mesh_dict["pcb"] == "NONE":
                    pass
                else:
                    raise ValueError("The mesh should be either a .obj file or NONE")
                
        if "motor" in mesh_dict:
            motors = [geom for geom in world_body.findall(".//geom") if geom.get("name", "").startswith("motor")]
            assert motors, "No motor found in the old model."
            for motor in motors:
                name = motor.get("name")
                color = motor.get("rgba")
                mass = motor.get("mass")
                parent = motor.getparent()
                parent.remove(motor)

                if mesh_dict["motor"].endswith(".obj") or mesh_dict["motor"].endswith(".stl"):
                    etree.SubElement(parent, "geom", type="mesh", name=name, mesh="motor", rgba=f"1 0 0 0.5", mass=mass, contype="10", conaffinity="0")
                elif mesh_dict["motor"] == "CYLINDER":
                    etree.SubElement(parent, "geom", type="cylinder", name=name, pos=vec2string([0,0,-0.015]), quat=vec2string([1,0,0,0 ]), size=f"{0.05} {0.03/2}", rgba=color, mass=mass, contype="10", conaffinity="0")
                elif mesh_dict["motor"] == "NONE":
                    pass
                else:
                    raise ValueError("The mesh should be either a .obj file or CYLINDER or NONE")

        if "stick" in mesh_dict:
            sticks = [geom for geom in world_body.findall(".//geom") if geom.get("name", "").startswith("stick")]
            assert sticks, "No stick found in the old model."
            for stick in sticks:
                name = stick.get("name")
                color = stick.get("rgba")
                mass = stick.get("mass")
                parent = stick.getparent()
                parent.remove(stick)

                if mesh_dict["stick"].endswith(".obj") or mesh_dict["stick"].endswith(".stl"):
                    etree.SubElement(parent, "geom", name=name, type="mesh", pos="0 0 0", quat="1 0 0 0", mesh="stick", rgba=color, mass=mass, friction="1.0 .0 .0", priority="2")
                elif mesh_dict["stick"] == "CYLINDER":
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["r"]
                    length = robot_cfg["l_"] 
                    broken = 0
                    etree.SubElement(parent, "geom", name=name, type="cylinder", pos="0 0 0", quat="1 0 0 0", size=f"{radius} {length/2 *(1-broken)}", rgba=color, mass=mass, friction="1.0 .0 .0", priority="2")
                elif mesh_dict["stick"] == "CAPSULE":
                    assert robot_cfg is not None, "Robot configuration is not provided."
                    radius = robot_cfg["r"]
                    length = robot_cfg["l_"] 
                    broken = 0
                    etree.SubElement(parent, "geom", name=name, type="capsule", pos="0 0 0", quat="1 0 0 0", size=f"{radius} {length/2 *(1-broken)}", rgba=color, mass=mass, friction="1.0 .0 .0", priority="2")
                else:
                    raise ValueError("The mesh should be either a .obj file or CYLINDER or CAPSULE")

    def recolor_floor(self, color, mark_color=".8 .8 .8"):
        floor = self.root.xpath('//texture[@name="texplane"]')[0]
        floor.set('rgb1', color[0])
        floor.set('rgb2', color[1])
        floor.set('markrgb', mark_color)

    def recolor_sky(self, color):
        sky = self.root.xpath('//texture[@type="skybox"]')[0]
        sky.set('rgb1', color[0])
        sky.set('rgb2', color[1])

    def remove_floor(self):
        world_body = self.root.findall("./worldbody")[0]
        floor_elem = world_body.find("geom[@name='floor']")
        assert floor_elem is not None, "No floor element found in the XML file."
        world_body.remove(floor_elem)


    def recolor_robot(self, colors, sphere_only=False):
        # lefts = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith("left")]
        # lefts = [body for body in self.root.findall(".//body") if body.get("name", "").startswith("l")]
        stick_idx = 0
        # pdb.set_trace()
        for i in range(len(colors)):
            left_geom = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"left{i}")]
            right_geom = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"right{i}")]
            stick_geom1 = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"stick{i*2}")]
            stick_geom2 = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"stick{i*2+1}")]
            battery_geom = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"battery{i}")]
            pcb_geom = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"pcb{i}")]
            motor_geom = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith(f"motor{i}")]
            imu_site = [geom for geom in self.root.findall(".//site") if geom.get("name", "").startswith(f"imu_site{i}")]
            back_imu_site = [geom for geom in self.root.findall(".//site") if geom.get("name", "").startswith(f"back_imu_site{i}")]
            if left_geom:
                left_geom[0].set("rgba", colors[i])
            if right_geom:
                right_geom[0].set("rgba", colors[i])
            if battery_geom:
                battery_geom[0].set("rgba", colors[i])
            if pcb_geom:
                pcb_geom[0].set("rgba", colors[i])
            if motor_geom:
                motor_geom[0].set("rgba", colors[i])
            if imu_site:
                imu_site[0].set("rgba", colors[i])
            if back_imu_site:
                back_imu_site[0].set("rgba", colors[i])
            if not sphere_only:
                if stick_geom1:
                    stick_geom1[0].set("rgba", colors[i])
                if stick_geom2:
                    stick_geom2[0].set("rgba", colors[i])
        # for right_geom, color in zip(rights, colors):
        #     right_geom.set("rgba", color)
        # sticks = [geom for geom in self.root.findall(".//geom") if geom.get("name", "").startswith("stick")]
        # for i, stick_geom in enumerate(sticks):
        #     stick_geom.set("rgba", colors[int(i/2)])

    def remove_shadow(self):
        quality = self.root.find(".//visual/quality")
        if quality is not None:
            # Set the shadowsize attribute to "0"
            quality.set("shadowsize", "0")
        else:
            # If the <quality> element doesn't exist, create it under <visual>
            visual = self.root.find(".//visual")
            if visual is None:
                # If there's no <visual> element, create one
                visual = etree.SubElement(self.root, "visual")
            quality = etree.SubElement(visual, "quality")
            quality.set("shadowsize", "0")



def is_headless():
    return not os.getenv('DISPLAY')


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

def numpy_to_native(obj):
    """
    Recursively convert numpy objects in a dictionary to their native Python equivalents.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, dict):
        return {key: numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(numpy_to_native(value) for value in obj)
    else:
        return obj