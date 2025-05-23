
from collections import defaultdict
import copy
import csv
import os
import pdb

import numpy as np
from omegaconf import OmegaConf
import omegaconf
import torch
from modular_legs import LEG_ROOT_DIR
from modular_legs.sim.robot_designer import RobotDesigner
from modular_legs.sim.robot_metadesigner import MetaDesignerAsym
from modular_legs.utils.others import is_list_like
# from modular_legs.sim.robot_metadesigner import MetaDesigner





def update_cfg_with_pipeline(cfg, design_pipeline, robot_cfg, mesh_dict, run_name=None, wandb_run_name=None, evolve_log_dir=None, init_pose_type="default"):
    '''
    Update the config with the design pipeline (TODO: input designer to save decoding time)
    @param cfg: Original OmegaConf config
    @param design_pipeline: List of integers representing the design (the ASYM version)
    @param robot_cfg: The robot config used for creating the robot; This is required as it will sinificantly affect the dynamics of the robot
    @param mesh_dict: The mesh dictionary used for creating the robot; This is required as it will sinificantly affect the dynamics of the robot
    @param run_name: The run name used for creating logging directories
    @param wandb_run_name: The run name used for easily identifying the run in wandb
    @param evolve_log_dir: The directory where the logs are saved
    @param init_pose_type: The initial pose type
    '''
    designer = MetaDesignerAsym(design_pipeline, robot_cfg=robot_cfg, mesh_dict=mesh_dict, mesh_mode="pretty")

    dark_grey = (0.15,0.15,0.15)
    black = (0.1,0.1,0.1)
    designer.builder.change_color_name("l", black)
    designer.builder.change_color_name("r", dark_grey)
    designer.builder.change_color_name("s", dark_grey)

    designer.compile(self_collision_test=False, stable_state_test=False) # Necessary?
    if cfg.sim.terrain is not None:
        designer.set_terrain(cfg.sim.terrain)
    xml, robot_properties = designer.get_xml(), designer.robot_properties
    xml_file, yaml_file = designer.save(gen_log_dir_asym(design_pipeline), render=False)

    num_joints = robot_properties["num_joints"]
    num_act = num_joints
    if cfg.agent.obs_version == "robust_proprioception":
        num_obs = 6 + num_joints*3 # for robust_proprioception
    elif cfg.agent.obs_version == "cheat_robust_proprioception":
        num_obs = 8 + num_joints*3 # for robust_proprioception
    elif cfg.agent.obs_version == "sensed_proprioception":
        num_obs = num_joints*9
    else:
        raise NotImplementedError

    conf = copy.deepcopy(cfg)

    if init_pose_type == "default":
        init_pos = robot_properties["stable_pos"][0].tolist()
        init_quat = robot_properties["stable_quat"][0].tolist()
    elif init_pose_type == "highest":
        highest_idx = np.argmax(robot_properties["stable_height"])
        init_pos = robot_properties["stable_pos"][highest_idx].tolist()
        init_quat = robot_properties["stable_quat"][highest_idx].tolist()
    elif init_pose_type == "optimized":
        # init_pos, init_quat, init_joint = optimize_pose(design_pipeline)
        # Optimized on the server side
        # if conf.sim.init_pos.startswith("?+"):
        #     add_h = float(conf.sim.init_pos[2:])
        # init_pos = f"optimized+{add_h}"
        # init_quat = "optimized"
        # conf.agent.default_dof_pos = "optimized"
        init_pos = conf.sim.init_pos
        init_quat = conf.sim.init_quat
        
    elif init_pose_type == "original":
        assert is_list_like(conf.sim.init_pos) and is_list_like(conf.sim.init_quat)
        init_pos = conf.sim.init_pos
        init_quat = conf.sim.init_quat

    conf.agent.num_act = num_act
    conf.agent.num_obs = num_obs
    conf.sim.asset_file = xml_file
    conf.sim.init_pos = init_pos
    conf.sim.init_quat = init_quat
    OmegaConf.update(conf, "trainer.evolution.design_pipeline", f"{[int(i) for i in design_pipeline]}") # for logging
    OmegaConf.update(conf, "trainer.evolution.design_pipeline_list", [int(i) for i in design_pipeline]) # for logging
    if wandb_run_name is not None:
        OmegaConf.update(conf, "trainer.evolution.run_name", wandb_run_name) # for logging

    if run_name is not None and evolve_log_dir is not None:
        train_log_dir = os.path.join(evolve_log_dir, run_name)
        conf.logging.data_dir = train_log_dir
    return conf





def split_list(input_list, max_length):
    return [input_list[i:i + max_length] for i in range(0, len(input_list), max_length)]



def is_metapipeline_valid(design_pipeline, level=0, conf_dict=None):
    """
    Check if the design pipeline is valid (5x+1 Version)
    """
    design_pipeline = design_pipeline.copy()

    if (len(design_pipeline)-1) % 5 != 0:
        print("Invalid: Unbuildable pipeline")
        return False
    
    # Config
    self_collision_threshold = conf_dict["self_collision_threshold"] if conf_dict is not None else 0.5
    ave_speed_threshold = conf_dict["ave_speed_threshold"] if conf_dict is not None else 0.1

    # LEVEL 0: Check if the design pipeline is buildable
    designer = MetaDesigner()
    designer.reset(design_pipeline.pop(0))
    design_pipeline = np.reshape(design_pipeline, (-1, 5))
    for step in design_pipeline:
        module = step[0]
        if module not in [0, 1]:
            pdb.set_trace()
            return False
        parent = step[1]
        if parent not in designer.node_ids:
            pdb.set_trace()
            return False
        pos_list = designer.get_pos_id_list(module, parent)
        position = step[2]
        if position not in pos_list:
            pdb.set_trace()
            return False
        rotation = step[3]
        orientation_list = designer.get_rotation_id_list(module, parent, position)
        if rotation not in orientation_list:
            pdb.set_trace()
            return False
        sibling_pose = step[4]
        sibling_id_list = designer.get_sibling_id_list(module)
        if sibling_pose not in sibling_id_list:
            pdb.set_trace()
            return False
        designer.step(step)
    if level == 0:
        return True
    
    # LEVEL 1: Check if the design pipeline is self-collision free
    designer.compile(self_collision_test=True, 
                     stable_state_test=True, 
                     movability_test=True if level > 1 else False,
                     config_dict={"init_pos": 0, "init_quat": 0} if level > 1 else None)
    pass_self_collision_test = designer.robot_properties["self_collision_rate"] < self_collision_threshold
    init_self_collision = designer.robot_properties["init_self_collision"]
    if not pass_self_collision_test:
        return False
    elif level == 1:
        return True
    
    # LEVEL 2: Check if the design pipeline is actively moveable
    easy_to_move = designer.robot_properties["ave_speed"] > ave_speed_threshold 
    if not easy_to_move:
        return False

    return True

def is_pipeline_valid(design_pipeline, level=0, conf_dict=None):
    """
    Check if the design pipeline is valid
    """
    if len(design_pipeline) % 4 != 0:
        return False
    
    # Config
    self_collision_threshold = conf_dict["self_collision_threshold"] if conf_dict is not None else 0.5
    ave_speed_threshold = conf_dict["ave_speed_threshold"] if conf_dict is not None else 0.1

    # LEVEL 0: Check if the design pipeline is buildable
    designer = RobotDesigner()
    designer.reset()
    design_pipeline = np.reshape(design_pipeline, (-1, 4))
    for step in design_pipeline:
        module = step[0]
        if module not in [0, 1]:
            pdb.set_trace()
            return False
        parent = step[1]
        if parent not in designer.node_ids:
            pdb.set_trace()
            return False
        pos_list = designer.get_pos_id_list(module, parent)
        position = step[2]
        if position not in pos_list:
            pdb.set_trace()
            return False
        rotation = step[3]
        orientation_list = designer.get_rotation_id_list(module, parent, position)
        if rotation not in orientation_list:
            pdb.set_trace()
            return False
        designer.step(step)
    if level == 0:
        return True
    
    # LEVEL 1: Check if the design pipeline is self-collision free
    designer.compile(self_collision_test=True, 
                     stable_state_test=True, 
                     movability_test=True if level > 1 else False,
                     config_dict={"init_pos": 0, "init_quat": 0} if level > 1 else None)
    pass_self_collision_test = designer.robot_properties["self_collision_rate"] < self_collision_threshold
    init_self_collision = designer.robot_properties["init_self_collision"]
    if not pass_self_collision_test or init_self_collision:
        return False
    elif level == 1:
        return True
    
    # LEVEL 2: Check if the design pipeline is actively moveable
    easy_to_move = designer.robot_properties["ave_speed"] > ave_speed_threshold 
    if not easy_to_move:
        return False

    return True


def decode(designer, design_pipeline):
    """
    Decode the design pipeline into a xml file
    """
    assert (len(design_pipeline)) % 4 == 0, "Invalid design pipeline"
    # assert is_pipeline_valid(design_pipeline), "Invalid design pipeline"

    designer.reset()
    for i in range(0,len(design_pipeline),4):
        designer.step(design_pipeline[i:i+4])
    designer.compile()

    return designer.get_xml(), designer.robot_properties


def gen_log_dir(design_pipeline):
    """
    Get the log directory
    """
    design_pipeline = np.reshape(design_pipeline, (-1, 4))
    asset_dir = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", "factory")
    for p in design_pipeline:
        p_dir = "-".join([str(i) for i in p])
        asset_dir = os.path.join(asset_dir, p_dir)
        
    return asset_dir

def gen_log_dir_5x1(design_pipeline):
    """
    Get the log directory for 5x+1 version
    """
    asset_dir = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", "factory")
    for p in [design_pipeline[:6]] + [design_pipeline[i:i+5] for i in range(6, len(design_pipeline), 5)]:
        p_dir = "-".join([str(i) for i in p])
        asset_dir = os.path.join(asset_dir, p_dir)
        
    return asset_dir

def gen_log_dir_asym(design_pipeline):
    """
    Get the log directory for 4x ASYM version
    """
    asset_dir = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", "factory", "asym")
    for p in np.reshape(design_pipeline, (-1, 4)):
        p_dir = "-".join([str(i) for i in p])
        asset_dir = os.path.join(asset_dir, p_dir)
        
    return asset_dir


def index_of_first_greater_than(lst, value):
    for i, v in enumerate(lst):
        if v > value:
            return i
    return len(lst)  # Return -1 if no such element is found


def find_closest_value(value, value_list):
    return min(value_list, key=lambda x:abs(x-value))

class CSVLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "log.csv")
        header = ['Index', 'Design Pipeline', 'Fitness', "Best Design Pipeline", "Best Fitness"]

        # Create a new CSV file and write the header
        with open(self.log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        self.best_design_pipeline, self.best_fitness = None, -np.inf

    def log(self, index, design_pipeline, fitness):
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_design_pipeline = design_pipeline
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([index, design_pipeline, fitness, self.best_design_pipeline, self.best_fitness])

class CSVLoggerPro:
    def __init__(self, filename, fieldnames=None):
        self.filename = filename
        self.fieldnames = fieldnames if fieldnames else ["timestep"]
        self.file_exists = os.path.exists(filename)

        self.rows = []
        self.file = open(self.filename, mode="a", newline="")
        self.writer = None
        self._initialize_writer()

    def _initialize_writer(self):
        """Initializes the CSV writer and writes headers if the file is new."""
        if not self.file_exists:
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        else:
            self.writer = csv.DictWriter(self.file, fieldnames=self._get_existing_headers())
            self._load_existing_data()

    def _get_existing_headers(self):
        """Retrieves headers from an existing file to maintain consistency."""
        with open(self.filename, mode="r", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
        return headers if headers else self.fieldnames

    def _load_existing_data(self):
        """Loads existing rows from the CSV file into memory."""
        with open(self.filename, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            self.rows = [row for row in reader]

    def _convert_list_like(self, value):
        """Auto-flattens and converts list-like objects into a standardized string format."""
        if isinstance(value, (list, tuple, np.ndarray, omegaconf.listconfig.ListConfig)):
            value_ = np.array(copy.deepcopy(value)).flatten().tolist()  # Flatten first
            return ",".join(map(str, value_))  # Convert to CSV-friendly format
        elif isinstance(value, (torch.Tensor)):
            value = np.array(value.detach().cpu()).flatten().tolist()  # Flatten first
            return ",".join(map(str, value))  # Convert to CSV-friendly format

        return str(value)  # Ensure all values are stored as strings

    def log(self, row_identifier, key, value, allow_new_row=True):
        """
        Logs a value under a specific key.
        
        row_identifier can be:
        - An integer (step index)
        - A condition in the format: (column_name, value)
        
        If `allow_new_row` is False and no match is found, raises a ValueError.
        """
        row = None
        value = self._convert_list_like(value)  # Auto-flatten list-like values

        if isinstance(row_identifier, int):
            row = next(
                (r for r in reversed(self.rows) if r.get("timestep") is not None and str(r["timestep"]).isdigit() and int(r["timestep"]) == row_identifier),
                None
            )
        elif isinstance(row_identifier, tuple):
            column_name, match_value = row_identifier
            match_value = self._convert_list_like(match_value)

            # Check if the column even exists in any row
            if column_name not in self.writer.fieldnames:
                self.writer.fieldnames.append(column_name)  # Add new column dynamically
                self._rewrite_file_with_new_headers()

            # Find the last row that matches the condition
            row = next((r for r in reversed(self.rows) if r.get(column_name) == match_value), None)

        if row is None:
            if allow_new_row:
                row = {row_identifier[0]: self._convert_list_like(row_identifier[1])} if isinstance(row_identifier, tuple) else {"timestep": row_identifier}
                self.rows.append(row)
            else:
                raise ValueError(f"No matching row found for {row_identifier}")

        row[key] = value

        # Update headers if new keys are added
        if key not in self.writer.fieldnames:
            self.writer.fieldnames.append(key)
            self._rewrite_file_with_new_headers()

    def _rewrite_file_with_new_headers(self):
        """Rewrites the file when new headers are introduced."""
        self.file.close()
        self.file = open(self.filename, mode="w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.writer.fieldnames)
        self.writer.writeheader()
        self.writer.writerows(self.rows)

    def flush(self):
        """Writes all stored rows to the file and clears the buffer."""
        self.file.seek(0)
        self.file.truncate()
        self.writer.writeheader()
        self.writer.writerows(self.rows)
        self.file.flush()

    def close(self):
        """Flushes any remaining data and closes the file properly."""
        self.flush()
        self.file.close()

