import cv2
import json
import os
import subprocess
import time

import numpy as np

from numpy.typing import NDArray

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
    command = ['nohup', 'bash', '-c', f'{cmd}']
    launch = subprocess.Popen(command, preexec_fn=os.setpgrp, stdout=subprocess.PIPE)
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


    