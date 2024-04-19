import yaml
import os

from utils import get_docker_ip, get_launch

if __name__ == '__main__':
    config_path = os.path.abspath(__file__).replace('settings.py', 'config.yaml')
    with open(config_path, 'r') as file:
            settings = yaml.safe_load(file)['settings']
            
    docker_container = settings['docker_container']
    ip = settings['ip']
    airsim_launch_path_file = settings['airsim_launch_path_file']
    
    if not ip:
        ip = get_docker_ip(docker_container)
        
    airsim_launch = get_launch(airsim_launch_path_file, ip)
    
    
    airsim_launch.start()
    airsim_launch.spin()