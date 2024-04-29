
import libtmux
import os
import roslaunch
import sys
import subprocess
import time
import yaml

from subprocess import call

from utils import container_ip

sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))

class SettingsConfig:
    def __init__(self):
        config_path = os.path.abspath(__file__).replace('exec.py', 'config.yaml')
        with open(config_path, 'r') as file:
            settings = yaml.safe_load(file)['settings']
            
        self.airsim_launch_file = settings['airsim_launch_path_file']
        self.ip = container_ip(settings['ue4_container_name'])
        

def launch_core():
    subprocess.Popen('roscore')
    time.sleep(3)  # Delay to initialize the roscore
    
def launch_sim(host_ip : str):
    """
    Function to launch a turtle-sim node
    """
    package = 'airsim_ros_pkgs'
    executable = 'airsim_node'
    node = roslaunch.core.Node(package, executable, args=f"_host_ip:={host_ip}", output='log')
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    launch.launch(node)
    
    return launch

def launch_rtabmap():
    """
    Function to launch a turtle-sim node
    """
    package = 'rtabmap_launch'
    executable = 'rtabmap_node'
    node = roslaunch.core.Node(package, executable, args=f"_rtabmap_args:=--delete_db_on_start _rgb_topic:=/airsim_node/Hydrone/stereo/Scene _depth_topic:=/airsim_node/Hydrone/stereo/DepthPerspective _camera_info_topic:=/airsim_node/Hydrone/stereo/Scene/camera_info _odom_topic:=/airsim_node/Hydrone/odom_local_ned _imu_topic:=/airsim_node/Hydrone/imu/Imu _visual_odometry:=false _frame_id:=stereo_optical _approx_sync:=false _rgbd_sync:=true _queue_size:=1000 _scan_cloud_topic:=\points _gen_cloud_voxel:=0.5")
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    launch.launch(node)
    
    return launch
   

if __name__ == "__main__":
    settings = SettingsConfig()
    # print(settings.airsim_launch_file, settings.ip)
    # launch_core()
    # airsim = launch_sim(settings.ip)   
    # rtabmap = launch_rtabmap()
    # for i in range(100): 
    #     if i == 20:
    #         rtabmap.stop()
    #     time.sleep(2)
    #     print('finish')
    # airsim.stop()
    # os.system('pkill -9 rosmaster')
    
    # call(["gnome-terminal", "-x", "sh", "-c", f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={settings.ip}"])
    # for i in range(30): 
    #     print(i)
    #     if i == 10:
    #         call(["gnome-terminal", "-x", "sh", "-c", f"roslaunch rtabmap_launch rtabmap.launch rgb_topic:=/airsim_node/Hydrone/stereo/Scene depth_topic:=/airsim_node/Hydrone/stereo/DepthPerspective camera_info_topic:=/airsim_node/Hydrone/stereo/Scene/camera_info odom_topic:=/airsim_node/Hydrone/odom_local_ned imu_topic:=/airsim_node/Hydrone/imu/Imu visual_odometry:=false frame_id:=stereo_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 scan_cloud_topic:=\points gen_cloud_voxel:=0.5"])
    #     time.sleep(1)
    # print('finish')
    # os.system('pkill -9 rosmaster')
    import signal
    airsim = subprocess.Popen(['gnome-terminal', '--disable-factory', "gnome-terminal", "-x", "sh", "-c", f"roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:={settings.ip}"],
                     preexec_fn=os.setpgrp)
    # do something here...
    time.sleep(4)
    rtabmap = subprocess.Popen(['gnome-terminal', '--disable-factory', "gnome-terminal", "-x", "sh", "-c", f"roslaunch rtabmap_launch rtabmap.launch rgb_topic:=/airsim_node/Hydrone/stereo/Scene depth_topic:=/airsim_node/Hydrone/stereo/DepthPerspective camera_info_topic:=/airsim_node/Hydrone/stereo/Scene/camera_info odom_topic:=/airsim_node/Hydrone/odom_local_ned imu_topic:=/airsim_node/Hydrone/imu/Imu visual_odometry:=false frame_id:=stereo_optical approx_sync:=false rgbd_sync:=true queue_size:=1000 scan_cloud_topic:=\points gen_cloud_voxel:=0.5"],
                     preexec_fn=os.setpgrp)
    time.sleep(10)
    os.killpg(rtabmap.pid, signal.SIGINT)
    os.killpg(airsim.pid, signal.SIGINT)