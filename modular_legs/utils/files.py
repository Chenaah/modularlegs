import datetime
import glob
import os
import pdb

from omegaconf import OmegaConf

from modular_legs import LEG_ROOT_DIR
from modular_legs.utils.others import is_list_like


def get_latest_file(path):
    # Initialize variables to hold the latest file and its modification time
    latest_file = None
    latest_mtime = 0
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Get the file's information
            file_stat = os.stat(file_path)
            # Check if it's the latest modification time so far
            if file_stat.st_mtime > latest_mtime:
                latest_mtime = file_stat.st_mtime
                latest_file = file_path
    
    return latest_file

def get_latest_ver_files(path, startwith):
    latest_file = None
    latest_mtime = 0
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.startswith(startwith):
                file_path = os.path.join(root, file_name)
                file_stat = os.stat(file_path)
                if file_stat.st_mtime > latest_mtime:
                    latest_mtime = file_stat.st_mtime
                    latest_file = file_path
    
    return latest_file


def update_cfg(cfg, alg="droq"):
    exp_name=f"{cfg.robot.mode}_{alg}"
    if cfg.logging.data_dir is None:

        cfg.logging.data_dir = os.path.join(LEG_ROOT_DIR,
                                            "exp", 
                                            exp_name, 
                                            get_log_name(cfg)
                                            )

    if not is_list_like(cfg.trainer.load_run):
        if cfg.trainer.load_run is not None and not os.path.isfile(cfg.trainer.load_run):
            if os.path.isfile(os.path.join(LEG_ROOT_DIR, cfg.trainer.load_run)):
                cfg.trainer.load_run = os.path.join(LEG_ROOT_DIR, cfg.trainer.load_run)
    else:
        for i, run in enumerate(cfg.trainer.load_run):
            if run is not None and not os.path.isfile(run):
                if os.path.isfile(os.path.join(LEG_ROOT_DIR, run)):
                    cfg.trainer.load_run[i] = os.path.join(LEG_ROOT_DIR, run)

    if not is_list_like(cfg.sim.asset_file):
        dir_path = os.path.join(LEG_ROOT_DIR, "modular_legs", "sim", "assets", "robots", cfg.sim.asset_file)

        if os.path.isdir(dir_path):
            cfg.sim.asset_file = glob.glob(os.path.join(dir_path, "*.xml"))


    return cfg

def get_log_name(cfg):

    date = datetime.datetime.now().strftime("%m%d%H%M%S")

    obs_name = ''.join([i[0] for i in cfg.agent.obs_version.split("_")])
    rwd_name = ''.join([i[0] for i in cfg.agent.reward_version.split("_")])
    asset_file = cfg.sim.asset_file if not is_list_like(cfg.sim.asset_file) else cfg.sim.asset_file[0]
    robot_name = asset_file.split("/")[-1].split(".")[0]
    if cfg.sim.reset_terrain and cfg.sim.reset_terrain_type is not None:
        add_info = ''.join([i[0] for i in cfg.sim.reset_terrain_type.split("_")])
        if cfg.sim.reset_terrain_params is not None:
            add_info += f"{cfg.sim.reset_terrain_params[0]}"
    else:
        add_info = ""
    return f"{robot_name}-{add_info}-{obs_name}-{rwd_name}-{date}".replace("--", "-")

def get_cfg_path(name="default"):
    if not name.endswith(".yaml") and "config/" not in name:
        yaml_file = os.path.join(LEG_ROOT_DIR, "config", f"{name}.yaml")
    elif name.endswith(".yaml") and "config/" not in name:
        yaml_file = os.path.join(LEG_ROOT_DIR, "config", name)
    elif "config/" in name and LEG_ROOT_DIR not in name:
        yaml_file = os.path.join(LEG_ROOT_DIR, name)
    elif LEG_ROOT_DIR in name:
        yaml_file = name

    return yaml_file

def get_cfg_name(path):
    return os.path.basename(path).split(".")[0]


def get_latest_model(folder_path):
    # Find all model files matching the pattern
    model_files = glob.glob(os.path.join(folder_path, "rl_model_*_steps.zip"))
    
    if not model_files:
        # raise FileNotFoundError("No model files found in the specified folder.")
        print("No model files found in the specified folder.")
        return None
    
    # Extract the numeric step count and sort the files by it
    model_files.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)
    
    # Return the path of the latest model
    return model_files[0]

    
def load_cfg(name="default", alg="sbx"):
    yaml_file = get_cfg_path(name)

    conf = OmegaConf.load(yaml_file)

    default_conf = OmegaConf.load(os.path.join(LEG_ROOT_DIR, "config", f"default.yaml"))
    merged_config = OmegaConf.merge(default_conf, conf)

    conf = update_cfg(merged_config, alg)

    return conf

def get_curriculum_cfg_paths(name):
    assert "curriculum" in name and ".yaml" not in name, "Curriculum config name must be a folder containing 'curriculum' and not end with '.yaml'"
    if not "config/" in name:
        name = os.path.join("config", name)
    curriculum_folder = os.path.join(LEG_ROOT_DIR, name)
    # files = [f for f in os.listdir(curriculum_folder) if f.endswith(".yaml")]
    files = [f for f in glob.glob(os.path.join(curriculum_folder, "*.yaml"), recursive=True)]
    return files


def generate_unique_filename(base_name, separator=""):
    """
    Generate a unique filename by appending a number if the base file exists.
    
    Parameters:
        base_name (str): The base name of the file (e.g., "xxx.pkl").
        
    Returns:
        str: A unique filename.
    """
    if not os.path.isfile(base_name):
        return base_name
    
    name, ext = os.path.splitext(base_name)
    counter = 0
    while True:
        new_name = f"{name}{separator}{counter}{ext}"
        if not os.path.isfile(new_name):
            return new_name
        counter += 1

