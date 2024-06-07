import json
import os
import subprocess
import time

def json_content(path : str):
    with open(path, 'r') as file:
        return json.load(file)
    
def parse_setup_env(setup_env : dict):
    ip = container_ip(setup_env['ue4_container']) if setup_env['ue4_container'] else setup_env['ue4_container']
    cfg = parse_cfg(setup_env['env'], setup_env['observation'])
    env_name = setup_env['env']
    
    return ip, cfg, env_name

def parse_setup_agent(setup_agent : dict):
    start_step = setup_agent['start_step']
    steps = setup_agent['steps']
    eval_freq = setup_agent['eval_freq']
    num_eval_episodes = setup_agent['num_eval_episodes']
    agent = setup_agent['agent']
    replay_buffer = setup_agent['replay_buffer']
    
    return start_step, steps, eval_freq, num_eval_episodes, agent, replay_buffer

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
    