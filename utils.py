#!/usr/bin/env python3
import roslaunch
import rospy
import subprocess
import os


def container_ip(container_name : str):
    byte_host_ip = subprocess.check_output(f'ping {container_name}' + ' -c1 | head -1 | grep -Eo "[0-9.]{4,}"', shell=True)
    return  byte_host_ip.decode('utf-8').replace('\n', "")

def airsim_launch(launch_file : str, host_ip : str):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    cli_args = [launch_file, 'output:=screen', f'host:={host_ip}']
    roslaunch_args = cli_args[2:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

    return roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, verbose=True)

def rtabmap_launch(launch_file : str, host_ip : str):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    cli_args = [launch_file, 'output:=screen', f'host:={host_ip}']
    roslaunch_args = cli_args[2:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

    return roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file, verbose=True)

# def background_node()



    
     
      

