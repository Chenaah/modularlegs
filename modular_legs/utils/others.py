


import ast
from collections.abc import Sequence
import os
import shlex
from subprocess import Popen, PIPE, STDOUT
import numpy as np

from modular_legs import LEG_ROOT_DIR


def is_list_like(variable):
    return isinstance(variable, (Sequence, np.ndarray)) and not isinstance(variable, (str, bytes))


def convert_np_arrays_to_lists(input_dict):
    converted_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            converted_dict[key] = value.tolist()
        else:
            converted_dict[key] = value
    return converted_dict

def is_number(variable):
    try:
        number = float(variable)
        return True
    except (ValueError, TypeError):
        return False
    
def get_freer_gpu():
    os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >{os.path.join(LEG_ROOT_DIR, "utils/temp")}')
    memory_available = [int(x.split()[2]) for x in open(os.path.join(LEG_ROOT_DIR, "utils/temp"), 'r').readlines()]
    return np.argmax(memory_available)

def get_freer_gpus(n):
    os.system(f'nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >{os.path.join(LEG_ROOT_DIR, "utils/temp")}')
    memory_available = [int(x.split()[2]) for x in open(os.path.join(LEG_ROOT_DIR, "utils/temp"), 'r').readlines()]
    return (-memory_available).argsort()[:n]


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
    

def string_to_list(s):
    if isinstance(s, str):
        return ast.literal_eval(s)
    else:
        return s
    


def get_simple_cmd_output(cmd, stderr=STDOUT):
    """
    Execute a simple external command and get its output.
    """
    args = shlex.split(cmd)
    return Popen(args, stdout=PIPE, stderr=stderr).communicate()[0]


def get_ping_time(host):
    host = host.split(':')[0]
    cmd = "fping {host} -C 3 -q".format(host=host)
    # result = str(get_simple_cmd_output(cmd)).replace('\\','').split(':')[-1].split() if x != '-']
    result = str(get_simple_cmd_output(cmd)).replace('\\', '').split(':')[-1].replace("n'", '').replace("-",
                                                                                                        '').replace(
        "b''", '').split()
    res = [float(x) for x in result]
    if len(res) > 0:
        return sum(res) / len(res)
    else:
        return 9999
