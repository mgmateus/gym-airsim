
import argparse
import os
import signal
import sys
import time
import yaml


from utils import container_ip, subprocess_launch
from environments import PositionNBV
   
def get_config():
    config_path = os.path.abspath(__file__).replace('exec.py', 'config.yaml')
    print(config_path)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration')
    
    parser.add_argument('--env', type= str, required=False, default='semisub-position', help='Gym environment name.')
    parser.add_argument('--container_simulation', type= str, required=False, default='ue4', help='Docker container name to Unreal.')
    args = parser.parse_args()
    
    config = get_config()
    ip = container_ip(args.container_simulation)      
    env_config = config['env'][args.env] 
    

    simu = f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={ip}"
    reconstruction = f"roslaunch rtabmap_launch rtabmap.launch rgb_topic:=/airsim_node/{env_config['vehicle_name']}/{env_config['camera_name']}/Scene \
                        depth_topic:=/airsim_node/{env_config['vehicle_name']}/{env_config['camera_name']}/DepthPerspective \
                        camera_info_topic:=/airsim_node/{env_config['vehicle_name']}/{env_config['camera_name']}/Scene/camera_info \
                        odom_topic:=/airsim_node/{env_config['vehicle_name']}/odom_local_ned \
                        imu_topic:=/airsim_node/{env_config['vehicle_name']}/imu/Imu visual_odometry:=false \
                        frame_id:={env_config['camera_name']}_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 \
                        scan_cloud_topic:=\points gen_cloud_voxel:=0.5"
    

    airsim = subprocess_launch(simu)
    time.sleep(4)
    # wait to connect into airsim...

    #Write main loop here...
    print(args.env)
    env = PositionNBV(ip, env_config, args.env)
    for i in range(3):
        print(f"step - {i}")
        env.step([.3, .1, .1, 0, 0])

#     rtabmap = subprocess_launch(reconstruction)
#     os.killpg(rtabmap.pid, signal.SIGINT)
    os.killpg(airsim.pid, signal.SIGINT)