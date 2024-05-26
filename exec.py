
import argparse
import json
import os
import signal
import sys
import time
import yaml


from utils import container_ip, process_cfg, airsim_launch, rtabmap_launch
from environments import PositionNBV
   
    
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration')
    
    parser.add_argument('--mode', type= str, required=False, default='train', help='Execution mode.')
    parser.add_argument('--env', type= str, required=False, default='semisub-position-aereo', help='Gym environment name.')
    parser.add_argument('--container_simulation', type= str, required=False, default='ue4', help='Docker container name to Unreal.')
    args = parser.parse_args()
    
    ip = container_ip(args.container_simulation)      
    mode, env_config = process_cfg(args.mode, args.env) 
    
    airsim = airsim_launch(ip)
    # wait to connect into airsim...

    #Write main loop here...

    env = PositionNBV(ip, env_config)
    env._random_vehicle_pose()
    
    # rtabmap = subprocess_launch(reconstruction)
    # time.sleep(15)
    for i in range(3):
        print(f"step - {i}")
        # env.step([0, 0, .1, 0, 0])
        env._random_vehicle_pose()
        time.sleep(1)

    # os.killpg(rtabmap.pid, signal.SIGINT)
    os.killpg(airsim.pid, signal.SIGINT)