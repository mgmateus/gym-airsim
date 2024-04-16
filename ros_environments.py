import rospy
import cv2
import os


import numpy as np

from math import dist
from numpy.typing import NDArray
from typing import List, Tuple

from gym import Env, spaces

from .airsim_resources.ros_api import Position, Trajectory


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
        
        
        

class EnvironmentPositionNBV(Env):
    def __init__(self, ip : str, 
                 vehicle_name : str, 
                 camera_name : str, 
                 observation_type : str,
                 space_observation_dim : int, 
                 space_action_dim : int,
                 max_steps : int):
        
        
        self.observation_space = np.zeros(shape=(space_observation_dim,))
        self.action_space = np.zeros(shape=(space_action_dim,))
        self.n_steps = 0
        self.max_steps = max_steps
        self.ep = 0
        
        self.vehicle = Position(ip, vehicle_name, camera_name, observation_type)
        
        self.views = ['rgb', 'depth'] if observation_type == 'stereo' else ['rgb', 'depth', 'segmentation']
        
    # def _store(self):
    #     if not self.vehicle.past_position or dist(self.vehicle.position, self.vehicle.past_position) > 0:
    #         views = self.vehicle.get_views()
    #         self.store_views(views)
    #         self.vehicle.past_position = self.vehicle.position

           
        
    def step(self, action):
        state = self.vehicle.get_state(action)
        return state
    
    def reset(self, curr_ep : int):
        self.ep = curr_ep
        # self.mkdir_dataset_path(curr_ep, self.views)
        
        return self.observation_space
    
class EnvironmentTrajectoryNBV(Env, LoggerFiles):
    def __init__(self, ip : str, 
                 vehicle_name : str, 
                 camera_name : str, 
                 observation_type : str,
                 files_path : str, 
                 space_observation_dim : int, 
                 space_action_dim : int,
                 max_steps : int):
        LoggerFiles.__init__(self, files_path)
        
        self.observation_space = np.zeros(shape=(space_observation_dim,))
        self.action_space = np.zeros(shape=(space_action_dim,))
        self.n_steps = 0
        self.max_steps = max_steps
        
        self.vehicle = Trajectory(ip, vehicle_name, camera_name, observation_type)
        
        self.views = ['rgb', 'depth'] if observation_type == 'stereo' else ['rgb', 'depth', 'segmentation']
        
    def _store(self):
        if not self.vehicle.past_position or dist(self.vehicle.position, self.vehicle.past_position) > 0:
            views = self.vehicle.get_views()
            self.store_views(views)
            self.vehicle.past_position = self.vehicle.position

           
        
    def step(self, action):
        state = self.vehicle.get_state(action)
        self._store()
        return state
    
    def reset(self, curr_ep : int):
        self.mkdir_dataset_path(curr_ep, self.views)
        
        return self.observation_space

