import cv2
import json
import os
import subprocess
import time

import numpy as np

from numpy.typing import NDArray


def normalize_value(x, min_val, max_val, a, b):
    return ((x - min_val) / (max_val - min_val)) * (b - a) + a

def pre_aug_obs_shape(img : NDArray, dim : tuple, type= 'int'):
        if type.endswith('float'):
            img_ = img.copy()
            nan_location = np.isnan(img_)
            img_[nan_location] = np.nanmax(img_)
            norm_image =  (img_)*255./5.
            norm_image[0,0] = 255.
            norm_image = norm_image.astype('uint8')
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR) #cv2.resize(norm_image.copy(), dim, interpolation = cv2.INTER_AREA)
            return cv2.resize(norm_image.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1) #cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)

        return cv2.resize(img.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1)

def json_content(path : str):
    with open(path, 'r') as file:
        return json.load(file)

def parse_cfg(env_ : str, observation_ : str):
    settings_path = os.path.abspath(__file__).replace('utils.py', 'settings/settings.json')
    config_path = os.path.abspath(__file__).replace('utils.py', 'config/config.json')
    
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
                                     
    env = config[env_]
    env['observation'] = observation_ 
    
    env['name'] = env_
    env['simulation']['vehicle']['name'] = svehicle_name
    env['simulation']['vehicle']['global_pose'] = [gx, gy, gz, groll, gpitch, gyaw]
    env['simulation']['vehicle']['camera']['name'] = svehicle_camera_name
    env['simulation']['vehicle']['camera']['dim'] = svehicle_camera_dim
    env['simulation']['vehicle']['camera']['fov'] = svehicle_camera_fov
    
    env['simulation']['shadow']['name'] = sshadow_name
    env['simulation']['shadow']['global_pose'] = [gsx, gsy, gsz, gsroll, gspitch, gsyaw]
    env['simulation']['shadow']['camera']['name'] = sshadow_camera_name
    env['simulation']['shadow']['camera']['dim'] = sshadow_camera_dim
    env['simulation']['shadow']['camera']['fov'] = sshadow_camera_fov
    
    return env

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

def airsim_launch(ip : str):
    if ip:
        launch = f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={ip}"
    else:
        launch = f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen"
    s = subprocess_launch(launch)
    time.sleep(4)
    return s

def rtabmap_launch(vehicle_name : str, camera_name : str, db_path : str):
    launch = f"roslaunch rtabmap_launch rtabmap.launch rtabmap_args:=--delete_db_on_start database_path:={db_path} rgb_topic:=/airsim_node/{vehicle_name}/{camera_name}/Scene \
                        depth_topic:=/airsim_node/{vehicle_name}/{camera_name}/DepthPerspective \
                        camera_info_topic:=/airsim_node/{vehicle_name}/{camera_name}/Scene/camera_info \
                        odom_topic:=/airsim_node/{vehicle_name}/odom_local_ned \
                        imu_topic:=/airsim_node/{vehicle_name}/imu/Imu visual_odometry:=false \
                        frame_id:={camera_name}_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 \
                        scan_cloud_topic:=\points gen_cloud_voxel:=0.5"
    s = subprocess_launch(launch)
    time.sleep(15)
    return s

def shapes(obs):
    return [tp.shape for k, tp in obs.items()]
    