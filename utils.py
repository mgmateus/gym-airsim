#!/usr/bin/env python3

import os
import subprocess
import sys
import time

import numpy as np

from gymnasium import spaces
sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))




from airsim_base.types import ImageType
from ros_helper import ActPosition


def container_ip(container_name : str):
    """Find the named container's ip.

    Args:
        container_name (str): container's name

    Returns:
        str: container's ip
    """
    byte_host_ip = subprocess.check_output(f'ping {container_name}' + ' -c1 | head -1 | grep -Eo "[0-9.]{4,}"', shell=True)
    return  byte_host_ip.decode('utf-8').replace('\n', "")

def subprocess_launch(cmd : str):
    """Execute a command in another terminal as a python subprocess.

    Args:
        cmd (str): The expecified command.

    Returns:
        Popen: The subprocess contained opened terminal resources.
    """
    launch = subprocess.Popen(['xterm', '-e', f'bash -c "{cmd}; exec bash"'],
                        preexec_fn=os.setpgrp)
    return launch


def _make_2d_observation_space(observation_type : str, camera_dim : str):
    w, h = camera_dim[0], camera_dim[1]
    return spaces.Dict(
                {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, w, h), dtype=int),
                    "depth": spaces.Box(low = 0, high = 255, shape=(1, w, h), dtype=int),
                    "tf": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(3,), dtype=np.float32),
                }
            )  if observation_type == 'stereo' else spaces.Dict(
                {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, w, h), dtype=int),
                    "depth": spaces.Box(low = 0, high = 255, shape=(1, w, h), dtype=int),
                    "segmentation": spaces.Box(low = 0, high = 255, shape=(3, w, h), dtype=int),
                    "tf": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(6,), dtype=np.float32),
                }
            )
            
def _make_3d_observation_space(observation_type : str, camera_dim : str):
    return None
            
def make_observation_space(observation_type : str, camera_dim : str, _2d : bool = True):
    return _make_2d_observation_space(observation_type, camera_dim) if _2d else _make_3d_observation_space(observation_type, camera_dim)


def normalize_value(x, min_val, max_val, a, b):
    return ((x - min_val) / (max_val - min_val)) * (b - a) + a
    
        

    
     
      

