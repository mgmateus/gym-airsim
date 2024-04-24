import rospy
import cv2
import os
import yaml
import sys

import numpy as np

from math import dist
from numpy.typing import NDArray
from typing import List, Tuple
from gym import Env, spaces


sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))
from ros_api import Position, Trajectory

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
                 vehicle_name : str, 
                 camera_name : str, 
                 domain : str,
                 observation_type : str,
                 space_observation_dim : int, 
                 space_action_dim : int,
                 max_steps : int):

        rospy.init_node(f'gym-{vehicle_name}-{domain}-{observation_type}')
        
        self.observation_space = np.zeros(shape=(space_observation_dim,)) #mudar tipo gym nativo
        self.action_space = np.zeros(shape=(space_action_dim,))  #mudar tipo para gym nativo
        self.n_steps = 0
        self.max_steps = max_steps
        self.ep = 0
        
        self.vehicle = Position(ip, vehicle_name, camera_name, observation_type)
        
        self.views = ['rgb', 'depth'] if observation_type == 'stereo' else ['rgb', 'depth', 'segmentation']
        
        
    def step(self, action):
        state = self.vehicle.get_state(action)
        return state
    
    def reset(self, curr_ep : int):
        self.ep = curr_ep
        
        return self.observation_space
    