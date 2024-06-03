
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import wandb
import yaml

import gymnasium as gym
import numpy as np

from gymnasium.envs.registration import register

register(
    id='gym-airsim/aereo-position-nbv',
    entry_point='environments:AereoPositionNBV'
)

register(
    id='gym-airsim/underwater-position-nbv',
    entry_point='environments:UnderwaterPositionNBV'
)

def json_content(path : str):
    with open(path, 'r') as file:
        return json.load(file)

def process_cfg(mode_ : str, observation_ : str, env_ : str):
    settings_path = os.path.abspath(__file__).replace('exec.py', 'settings/settings.json')
    config_path = os.path.abspath(__file__).replace('exec.py', 'config.json')
    
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
    
def process_env(env_name : str):
    split = env_name.rsplit('-')
    _ = split.pop(0)
    split.append('nbv')
    env_name = '-'.join(split)
    return env_name
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration')
    
    parser.add_argument('--mode', type= str, required=False, default='train', help='Execution mode.')
    parser.add_argument('--agent', type= str, required=False, default='curl', help='Agent class.')
    parser.add_argument('--observation', type= str, required=False, default='rgb', help='Agent class.')
    parser.add_argument('--env', type= str, required=False, default='eolic-aereo-position', help='Gym environment name.')
    parser.add_argument('--container_simulation', type= str, required=False, default='ue4', help='Docker container name to Unreal.')
    args = parser.parse_args()
    
    
    
    ip = container_ip(args.container_simulation)      
    mode, env_config = process_cfg(args.mode, args.observation, args.env) 
    env_name = process_env(env_config['name'])
    
    # wandb.init(
    #     project=f"{args.agent}-{env_config['name']}-{args.observation}",
    #     config= mode
    # )
    
    airsim = airsim_launch(ip) # wait to connect into airsim...
    
    # env = gym.make(f'gym-airsim/{env_name}', ip=ip, config=env_config, node=env_name)
    # observation = env.reset()
    # for i in range(20):
    #     print(f"step - {i}")
    #     action = np.array([0, 0.05, .1, 0, .001], dtype=np.float32)
    #     env.step(action)
    #     time.sleep(0.5)
    # os.killpg(airsim.pid, signal.SIGINT)
    # sys.exit()
    
    env = gym.make(f'gym-airsim/{env_name}', ip=ip, config=env_config, node=env_name)
    observation = env.reset()
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    
    
    for step in range(50):
        # evaluate agent periodically

        if step % mode['eval_freq'] == 0:
            # wandb.log({"episode": episode, "step": step})
            pass
            # evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            # if args.save_model:
            #     agent.save_curl(model_dir, step)
            # if args.save_buffer:
            #     replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                # wandb.log({"duration": time.time() - start_time, "step": step})
                start_time = time.time()
                # wandb.log({"episode_reward": episode_reward, "step": step})
                
            obs, _= env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            
            # wandb.log({"episode": episode, "step": step})


        # sample action for data collection
        if step < mode['init_steps']:
            action = env.action_space.sample()
        else:
            # with utils.eval_mode(agent):
            #     action = agent.sample_action(obs)
            pass

        # run training update
        if step >= mode['init_steps']:
            num_updates = 1 
            for _ in range(num_updates):
                # agent.update(replay_buffer, L, step)
                pass

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env.unwrapped.max_episode_steps else float(done)
        episode_reward += reward
        # replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
    
    _ = env.reset()
    os.killpg(airsim.pid, signal.SIGINT)
    sys.exit()