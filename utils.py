#!/usr/bin/env python3
import subprocess
import os


def container_ip(container_name : str):
    byte_host_ip = subprocess.check_output(f'ping {container_name}' + ' -c1 | head -1 | grep -Eo "[0-9.]{4,}"', shell=True)
    return  byte_host_ip.decode('utf-8').replace('\n', "")

def subprocess_launch(cmd : str):
    """
    Function to launch the airsim simulation node
    """
    launch = subprocess.Popen(['gnome-terminal', '--disable-factory', "gnome-terminal", "-x", "sh", "-c", cmd],
                     preexec_fn=os.setpgrp)
    
    return launch







    
     
      

