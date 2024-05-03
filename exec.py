
import os
import signal
import sys
import time
import yaml


from utils import container_ip, subprocess_launch
from ros_environments import PositionNBV
   

if __name__ == "__main__":
    config_path = os.path.abspath(__file__).replace('exec.py', 'config.yaml')
    print(config_path)
    with open(config_path, 'r') as file:
        r = yaml.safe_load(file)
        settings = r['settings']
        gym = r['gym']

    ip = container_ip(settings['ue4_container_name'])       

    simu = f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={ip}"
    reconstruction = f"roslaunch rtabmap_launch rtabmap.launch rgb_topic:=/airsim_node/{gym['env']['vehicle_name']}/{gym['env']['camera_name']}/Scene \
                        depth_topic:=/airsim_node/{gym['env']['vehicle_name']}/{gym['env']['camera_name']}/DepthPerspective \
                        camera_info_topic:=/airsim_node/{gym['env']['vehicle_name']}/{gym['env']['camera_name']}/Scene/camera_info \
                        odom_topic:=/airsim_node/{gym['env']['vehicle_name']}/odom_local_ned \
                        imu_topic:=/airsim_node/{gym['env']['vehicle_name']}/imu/Imu visual_odometry:=false \
                        frame_id:={gym['env']['camera_name']}_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 \
                        scan_cloud_topic:=\points gen_cloud_voxel:=0.5"
    

    airsim = subprocess_launch(simu)
    time.sleep(4)
    # wait to connect into airsim...

    #Write main loop here...
    print(gym['env'])
    env = PositionNBV(ip, gym['env'])
    for i in range(2):
        env.step([.1, .1, .1, 0, 0])


#     rtabmap = subprocess_launch(reconstruction)
#     os.killpg(rtabmap.pid, signal.SIGINT)
    os.killpg(airsim.pid, signal.SIGINT)