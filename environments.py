import rospy
import cv2
import os
import yaml
import sys
import time

import numpy as np

from math import dist
from numpy.typing import NDArray
from typing import List, Tuple
from gymnasium import Env, spaces

sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))

from airsim_base.types import ImageType
from ros_helper import ActPosition
from utils import make_observation_space, normalize_value
 
class LoggerFiles:
    def __init__(self, files_path : str) -> None:
        self.__dataset_path = files_path + '/dataset'
        self.__count_name = 0
        
        self.views_paths = []
        
    def mkdir_dataset_path(self, episode : int, views : List[str]):
        ep_path = self.__dataset_path + '/ep_' + str(episode)
        for view in views:
            view_path =  ep_path + '/' + view
            self.views_paths.append(view_path)
            if not os.path.exists(view_path):
                os.makedirs(view_path)
        
    def store_views(self, views : List[NDArray]):
        name =  str(self.__count_name) + ".png"
        # rospy.logwarn(f"{self.views_paths}")

        for i, view_path in enumerate(self.views_paths):
            # cv2.imwrite(view_path + '/' + name, views[i])
            if i == 1:
                cv2.imwrite(view_path + '/' + name, 255 - views[i].astype(np.uint16))
            else:
                cv2.imwrite(view_path + '/' + name, views[i])
            
        self.__count_name += 1




class PositionNBV(Env):
   
    def __init__(self, ip : str, 
                 config : dict,
                 env_name : str):
        
        rospy.init_node(f'gym-{env_name}-{config["observation"]}')

        self.views = ['rgb', 'depth'] if config['observation'] == 'stereo' else ['rgb', 'depth', 'segmentation']

        self.observation_space = make_observation_space(config['observation'], config['vehicle']['camera']['dim'])
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float64)  
        self.n_steps = 0
        self.max_steps = config['max_steps']

        self.action_range = config['action_range']
        
        self.vehicle = ActPosition(ip, 
                                   config['vehicle']['name'], 
                                   config['vehicle']['camera']['name'], 
                                   config['vehicle']['start_pose'],
                                   config['observation'])   
        self.vehicle.enableApiControl(True, config['shadow']['name'])
        self.vehicle.armDisarm(True, config['shadow']['name'])
        self.shadow = config['shadow']
        self.vehicle.simSetDetectionFilterRadius(config['shadow']['camera']['name'], 
                                                 ImageType.Scene, 200 * 100, 
                                                 vehicle_name=config['shadow']['name']) 
        self.vehicle.simAddDetectionFilterMeshName(config['shadow']['camera']['name'], 
                                              ImageType.Scene, f"{config['markers']['name']}*", 
                                              vehicle_name=config['shadow']['name']) 
        
        self.markers = {'Cube'}
        self.original_len = config['markers']['num']
        self.past_len = config['markers']['num']

        self.take_off()
        
    def take_off(self):
        self.vehicle.take_off()
        self.vehicle.take_off(self.shadow['name'])
        
    def _moveon(self, action):
        norm_action = self._normalize_action(action)
        self.vehicle.moveon(norm_action)
        
        pose = self.vehicle.pose
        self.vehicle.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.shadow['name'])
        
        self.vehicle.pgimbal(self.shadow['name'], self.shadow['camera']['name'])
        
        
    def _reward(self):
      
        viewd_markers, distances = self.vehicle.simGetDetectedMeshesDistances('shadow', ImageType.Scene,vehicle_name="Shadow")
        
        if distances:
            if (8053.1 - min(distances)) < 30 or (8053.1 - max(distances)) > 120:
                return -10
        
        self.markers -= set(viewd_markers)
        len_markers = len(self.markers)

        if len_markers == self.past_len:
            return 0
        
        return self.original_len - len_markers
    
    
    def _normalize_action(self, action):
        x, y, z, yaw_angle, gimbal_pitch_angle = action
        xmin, xmax = self.action_range['x']
        ymin, ymax = self.action_range['y']
        zmin, zmax = self.action_range['z']
        yaw_min, yaw_max = self.action_range['yaw']
        gimbal_pitch_min, gimbal_pitch_max = self.action_range['gimbal_pitch']
        
        px = normalize_value(x, -1, 1, xmin, xmax)
        py = normalize_value(y, -1, 1, ymin, ymax)
        pz = normalize_value(z*-1, -1, 1, zmin, zmax)
        yaw = normalize_value(yaw_angle, -1, 1, yaw_min, yaw_max)
        gimbal_pitch = normalize_value(gimbal_pitch_angle, -1, 1, gimbal_pitch_min, gimbal_pitch_max)
        
        return px, py, pz, yaw, gimbal_pitch


    def reset(self):
        
        return self.observation_space
    
    def step(self, action):
        # norm_action = self._normalize_action(action)
        # pose, pgimbal = self.vehicle.moveon(norm_action)
        # self._move_shadow(pose, pgimbal)
        self._moveon(action)
        reward = self._reward()
        observation = self.vehicle.get_observation()
        
        
        done = False # condition to finish
        if done:
            self.vehicle.reset() #reset of airsim
            return observation, done
     
        return observation, done