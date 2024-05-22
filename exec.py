
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
    
    parser.add_argument('--mode', type= str, required=False, default='train', help='Execution mode.')
    parser.add_argument('--env', type= str, required=False, default='semisub-position-aereo', help='Gym environment name.')
    parser.add_argument('--container_simulation', type= str, required=False, default='ue4', help='Docker container name to Unreal.')
    args = parser.parse_args()
    
    config = get_config()
    ip = container_ip(args.container_simulation)      
    mode = config['mode'][args.mode]
    env_config = config['env'][args.env] 
    vehicle = env_config['vehicle']
    
   
    

    simu = f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={ip}"
    reconstruction = f"roslaunch rtabmap_launch rtabmap.launch delete_db_on_start:=true rgb_topic:=/airsim_node/{vehicle['name']}/{vehicle['camera']['name']}/Scene \
                        depth_topic:=/airsim_node/{vehicle['name']}/{vehicle['camera']['name']}/DepthPerspective \
                        camera_info_topic:=/airsim_node/{vehicle['name']}/{vehicle['camera']['name']}/Scene/camera_info \
                        odom_topic:=/airsim_node/{vehicle['name']}/odom_local_ned \
                        imu_topic:=/airsim_node/{vehicle['name']}/imu/Imu visual_odometry:=false \
                        frame_id:={vehicle['camera']['name']}_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 \
                        scan_cloud_topic:=\points gen_cloud_voxel:=0.5"
    

    airsim = subprocess_launch(simu)
    time.sleep(4)
    # wait to connect into airsim...

    #Write main loop here...
    print(args.env)
    env = PositionNBV(ip, env_config, args.env)
    # rtabmap = subprocess_launch(reconstruction)
    # time.sleep(15)
    for i in range(30):
        print(f"step - {i}")
        env.step([0, 0, .1, 0, 0])

    # os.killpg(rtabmap.pid, signal.SIGINT)
    os.killpg(airsim.pid, signal.SIGINT)