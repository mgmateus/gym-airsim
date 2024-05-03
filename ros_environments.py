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
from ros_api import ActPosition

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

def make_observation_space(observation_type : str, camera_dim : str):
    return spaces.Dict(
                {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, camera_dim[0], camera_dim[1]), dtype=int),
                    "depth": spaces.Box(low = 0, high = 255, shape=(1, camera_dim[0], camera_dim[1]), dtype=int),
                    "tf": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(3,), dtype=np.float32),
                }
            )  if observation_type == 'stereo' else spaces.Dict(
                {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, camera_dim[0], camera_dim[1]), dtype=int),
                    "depth": spaces.Box(low = 0, high = 255, shape=(1, camera_dim[0], camera_dim[1]), dtype=int),
                    "segmentation": spaces.Box(low = 0, high = 255, shape=(3, camera_dim[0], camera_dim[1]), dtype=int),
                    "tf": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(6,), dtype=np.float32),
                }
            )

def stop(target_pos, tf):
    x, y, z, yaw, pitch = tf
    return True

        

class PositionNBV(Env):
    def __init__(self, ip : str, 
                 config : dict):
        
        print(config['vehicle_name'])
        rospy.init_node(f'gym-{config["vehicle_name"]}-{config["domain"]}-{config["observation_type"]}')

        vehicle_name = config['vehicle_name']
        camera_name = config['camera_name']
        camera_dim = config['camera_dim']
        observation_type = config['observation_type']
        max_steps = config['max_steps']

        self.observation_space = make_observation_space(observation_type, camera_dim)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float64)  #mudar tipo para gym nativo
        self.n_steps = 0
        self.max_steps = max_steps
        self.ep = 0
        
        self.vehicle = ActPosition(ip, vehicle_name, camera_name, observation_type)    
        self.vehicle.simSetDetectionFilterRadius("shadow", ImageType.Scene, 200 * 100) 
        self.vehicle.simAddDetectionFilterMeshName("shadow", ImageType.Scene, "Cube*")    
        
        self.views = ['rgb', 'depth'] if observation_type == 'stereo' else ['rgb', 'depth', 'segmentation']

        self.markers = []
        self.original_len = len(self.markers)

        self.vehicle.take_off() 

        time.sleep(3)
        
    
    def _reward(self):
        self.vehicle.simSetVehiclePose(self.vehicle.vehicle_pose, True, 'shadow')
        # detections = self.vehicle.simGetDetectedMeshes('shadow', ImageType.Scene, vehicle_name = 'shadow')
        
        # self.markers -= detections
        # len_markers = len(self.markers)
        



    def reset(self, curr_ep : int):
        self.ep = curr_ep
        
        return self.observation_space
    
    def step(self, action):
        _ = self.vehicle.moveon(action)
        # observation = self.vehicle.get_observation()
        # print(self.vehicle.simGetDetectedMeshesDistances('shadow', ImageType.Scene, vehicle_name = 'shadow'))
        # print(self.vehicle.simGetDetections('shadow', ImageType.Scene))
        # done = False # condition to finish
        # if done:
        #     self.vehicle.reset() #reset of airsim
        #     return observation, done
        
        # done = False
        # return observation, done