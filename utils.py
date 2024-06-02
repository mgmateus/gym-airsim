#!/usr/bin/env python3

import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np

from gymnasium import spaces
sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))




from airsim_base.types import ImageType

def json_content(path : str):
    with open(path, 'r') as file:
        return json.load(file)

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


def process_cfg(mode_ : str, observation_ : str, env_ : str):
    settings_path = os.path.abspath(__file__).replace('utils.py', 'settings/settings.json')
    config_path = os.path.abspath(__file__).replace('utils.py', 'config.json')
    
    settings = json_content(settings_path)
    config = json_content(config_path)
    
    svehicle_names = list(settings['Vehicles'].keys())
    
    svehicle = settings['Vehicles'][svehicle_names[0]]
    svehicle_name = svehicle_names[0]
    gx, gy, gz, groll, gpitch, gyaw = svehicle['X'], svehicle['Y'], svehicle['Z'], \
                                svehicle['Roll'], svehicle['Pitch'], svehicle['Yaw']
    svehicle_camera_name = list(svehicle['Cameras'].keys())[0]
    
    svehicle_camera_dim = svehicle['Cameras'][svehicle_camera_name]['CaptureSettings'][0]['Width'],\
                          svehicle['Cameras'][svehicle_camera_name]['CaptureSettings'][0]['Height']  
                          
    svehicle_camera_fov = svehicle['Cameras'][svehicle_camera_name]['CaptureSettings'][0]["FOV_Degrees"]
    
    sshadow = settings['Vehicles'][svehicle_names[1]]
    sshadow_name = svehicle_names[1]
    gsx, gsy, gsz, gsroll, gspitch, gsyaw = sshadow['X'], sshadow['Y'], sshadow['Z'], \
                                sshadow['Roll'], sshadow['Pitch'], sshadow['Yaw']
    sshadow_camera_name = list(sshadow['Cameras'].keys())[0]
    sshadow_camera_dim = sshadow['Cameras'][sshadow_camera_name]['CaptureSettings'][0]['Width'],\
                          sshadow['Cameras'][sshadow_camera_name]['CaptureSettings'][0]['Height']
    sshadow_camera_fov = sshadow['Cameras'][sshadow_camera_name]['CaptureSettings'][0]["FOV_Degrees"]
                          
    mode = config['mode'][mode_]                      
    env = config['envs'][env_]
    env['observation'] = observation_ 
    
    env['name'] = env_
    env['vehicle']['name'] = svehicle_name
    env['vehicle']['global_pose'] = [gx, gy, gz, groll, gpitch, gyaw]
    env['vehicle']['camera']['name'] = svehicle_camera_name
    env['vehicle']['camera']['dim'] = svehicle_camera_dim
    env['vehicle']['camera']['fov'] = svehicle_camera_fov
    
    env['shadow']['name'] = sshadow_name
    env['shadow']['global_pose'] = [gsx, gsy, gsz, gsroll, gspitch, gsyaw]
    env['shadow']['camera']['name'] = sshadow_camera_name
    env['shadow']['camera']['dim'] = sshadow_camera_dim
    env['shadow']['camera']['fov'] = sshadow_camera_fov
    
    return mode, env

def airsim_launch(ip : str):
    launch = f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={ip}"
    s = subprocess_launch(launch)
    time.sleep(4)
    return s

def rtabmap_launch(vehicle_name : str, camera_name : str):
    launch = f"roslaunch rtabmap_launch rtabmap.launch delete_db_on_start:=true rgb_topic:=/airsim_node/{vehicle_name}/{camera_name}/Scene \
                        depth_topic:=/airsim_node/{vehicle_name}/{camera_name}/DepthPerspective \
                        camera_info_topic:=/airsim_node/{vehicle_name}/{camera_name}/Scene/camera_info \
                        odom_topic:=/airsim_node/{vehicle_name}/odom_local_ned \
                        imu_topic:=/airsim_node/{vehicle_name}/imu/Imu visual_odometry:=false \
                        frame_id:={camera_name}_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 \
                        scan_cloud_topic:=\points gen_cloud_voxel:=0.5"
    s = subprocess_launch(launch)
    time.sleep(15)
    return s

def normalize_value(x, min_val, max_val, a, b):
    return ((x - min_val) / (max_val - min_val)) * (b - a) + a

def random_choice(min_range, max_range):
    random.seed()
    a, b = min_range
    c, d = max_range
    return random.uniform(a, b) if random.choice([True, False]) else random.uniform(c, d)

def theta(vehicle_pose : list, target_position : list):
    vx, vy, vz, _, _, yaw = vehicle_pose
    tx, ty, _ = target_position
    heading = np.arctan2(ty - vy, tx - vx)
    
    if heading > math.pi:
        heading -= 2 * math.pi

    elif heading < -math.pi:
        heading += 2 * math.pi
    return heading

def quaternion_to_euler(q):
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]
        

    
     
      

