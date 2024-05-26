
import cv2
import math
import os
import rospy
import random
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
from utils import normalize_value, random_choice, theta, quaternion_to_euler
 
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


def make_2d_observation_space(observation_type : str, camera_dim : str):
    w, h = camera_dim[0], camera_dim[1]
    return spaces.Dict(
                {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, w, h), dtype=int),
                    "depth": spaces.Box(low = 0, high = 255, shape=(1, w, h), dtype=int),
                    "tf": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(3,), dtype=np.float32),
                }
            )  if observation_type == 'stereo' else spaces.Dict(
                {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, w, h), dtype=int),
                    "depth": spaces.Box(low = 0, high = 255, shape=(1, w, h), dtype=int),
                    "segmentation": spaces.Box(low = 0, high = 255, shape=(3, w, h), dtype=int),
                    "tf": spaces.Box(low = -2**63, high = 2**63 - 2, shape=(6,), dtype=np.float32),
                }
            )
            
def make_3d_observation_space(observation_type : str, camera_dim : str):
    return None
            
def make_observation_space(observation_type : str, camera_dim : str, _2d : bool = True):
    return make_2d_observation_space(observation_type, camera_dim) if _2d else make_3d_observation_space(observation_type, camera_dim)

class PositionNBV(Env):
   
    @staticmethod
    def set_markers(markers_name, num):
        m = [markers_name]
        m = m + [f"{markers_name}{i}" for i in range(1, num+1)]
        return set(m)
    
    @staticmethod
    def vs_start(ip : str, config : dict):
        vehicle_ = config['vehicle']
        shadow = config['shadow']
        observation = config['observation']
        markers = config['markers']
        
        vehicle = ActPosition(ip, 
                              vehicle_['name'], 
                              vehicle_['camera']['name'], 
                              vehicle_['global_pose'], 
                              vehicle_['start_pose'],
                              observation)
        
        vehicle.enableApiControl(True, shadow['name'])
        vehicle.armDisarm(True, shadow['name'])
        vehicle.simSetDetectionFilterRadius(shadow['camera']['name'], 
                                                 ImageType.Scene, 200 * 100, 
                                                 vehicle_name=shadow['name'])
        
        vehicle.simAddDetectionFilterMeshName(shadow['camera']['name'], 
                                              ImageType.Scene, f"{markers['name']}*", 
                                              vehicle_name=shadow['name'])
        return vehicle, shadow
        
    @staticmethod
    def quadrant(px : float, py : float):
        """
        The objective is turn around of center (0,0) represented for the structure center.
        
                 X
        --------------------
        |        |         |
        |   Q3   |    Q2   |                  
        |        |         |   
        -------(0,0)-------- Y              
        |        |         |   
        |   Q1   |    Q0   |   
        |        |         |   
        --------------------   

        """
        if px < 0 and py > 0:
            #Q0
            return np.radians(-67.5), np.radians(-22.5)
        
        if px < 0 and py < 0:
            #Q1
            return np.radians(22.5), np.radians(67.5)
        
        if px > 0 and py > 0:
            #Q2
            return np.radians(-112.5), np.radians(-157.5)
        return np.radians(112.5), np.radians(157.5)
        
    def __init__(self, ip : str, 
                 config : dict):
        
        rospy.init_node(f"gym-{config['name']}-{config['observation']}")

        self.observation_space = make_observation_space(config['observation'], config['vehicle']['camera']['dim'])
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float64)  
        self.n_steps = 0
        
        self.domain = config['domain']
        self.vehicle, self.shadow = self.vs_start(ip, config)
        self.vehicle_cfg = config['vehicle']
        self.action_range = config['action_range']
        self.target_range = config['target_range']
        self.altitude_range = config['altitude_range']
        self.centroide = config['centroide']
        self.markers = self.set_markers(config['markers']['name'], config['markers']['num'])
        self.original_len = config['markers']['num']
        self.past_len = config['markers']['num']
        
        self._start_pose()
        self._take_off()
        
    def _start_pose(self):        
        self.vehicle.set_start_pose()
        self.vehicle.set_start_pose(self.shadow['name'])
        
        if self.domain == 'aereo':
            position, _= self.vehicle.get_start_pose()
            position[1] -= 13 
            position[2] = self.altitude_range[0]
            self.vehicle.set_object_pose(position, [0, np.deg2rad(-90), 0], self.vehicle_cfg['base_name'])
            global_position = self.shadow['global_pose'][:3]
            shadow_position, _= self.vehicle.get_pose(self.shadow['name'])
            current_position = np.array(global_position) + np.array(shadow_position)
            current_position = current_position.tolist()
            current_position[1] -= 13 
            current_position[2] = self.altitude_range[0]
            self.vehicle.set_object_pose(current_position, [0, np.deg2rad(-90), 0], self.shadow['base_name'])
            
    def _take_off(self):
        self.vehicle.take_off()
        self.vehicle.take_off(self.shadow['name'])
        
        return True
    
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
    
    
    def _moveon(self, action):
        norm_action = self._normalize_action(action)
        self.vehicle.moveon(norm_action)
        
        pose = self.vehicle.pose
        self.vehicle.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.shadow['name'])
        
        self.vehicle.pgimbal(self.shadow['name'], self.shadow['camera']['name'])
        
        return True
        
    def _random_vehicle_pose(self, randoz_yaw : bool = False, randon_z : bool = False):
        random.seed()
        axmin, axmax = self.action_range['x']
        aymin, aymax = self.action_range['y']
        
        txmin, txmax = self.target_range['x']
        tymin, tymax = self.target_range['y']
        
        zmin, zmax = self.altitude_range
        
        px = random_choice((axmin - txmin, txmin), (txmax, txmax + axmax))
        py = random_choice((aymin - tymin, tymin), (tymax, tymax + aymax))
        pz = random.uniform(zmin, zmax) if randon_z else zmin-2
        

        centroide_pose = self.vehicle.objectp2list(self.centroide)
        tf = self.vehicle.tf()
        vehicle_position = [px, py, tf[2]]
        vehicle_e_orientation = quaternion_to_euler(tf[3:])
        vehicle_pose = vehicle_position + vehicle_e_orientation
        
        a, b = self.quadrant(px, py)
        t = theta(vehicle_pose, centroide_pose[:3])
        yaw = random.uniform(a, b) + t if randoz_yaw else t
        self.vehicle.set_start_pose([px, py, pz], [0, 0, yaw])
        self.vehicle.set_start_pose([px, py, pz], [0, 0, yaw], self.shadow['name'])
        
        if self.domain == 'aereo':
            position = [px, py, pz]
            position[1] -= 13 
            position[2] = self.altitude_range[0]
            self.vehicle.set_object_pose(position, [0, np.deg2rad(-90), 0], self.vehicle_cfg['base_name'])
            
            global_position = self.shadow['global_pose'][:3]
            current_position = np.array(global_position) + np.array(position)
            current_position = current_position.tolist()
            self.vehicle.set_object_pose(current_position, [0, np.deg2rad(-90), 0], self.shadow['base_name'])
            
    def _wshadow_distance(self):
        wx, wy, wz, _, _, _ = self.shadow['global_pose']

        pose = self.vehicle.simGetVehiclePose(vehicle_name=self.shadow['name'])
        rx, ry, rz = pose.position.x_val, pose.position.y_val, pose.position.z_val
        
        x, y, z = wx + rx, wy + ry, wz + rz
        
        return np.sqrt(x**2 + y**2 + z**2)
        

    def _get_state(self):
        observation = self.vehicle.get_observation()
        #implemtents pack observation
        
        viewd_markers, distances = self.vehicle.simGetDetectedMeshesDistances(self.shadow['camera']['name'], ImageType.Scene,vehicle_name=self.shadow['name'])
        d = self._wshadow_distance()
        done = False
        reset_pose = False
        distance = 0
        len_markers = self.past_len
        if distances:
            self.markers -= set(viewd_markers)
            len_markers = len(self.markers)
            
            if (d - min(distances)) < 30 or (d - max(distances)) > 120:
                distance = 1
                reset_pose = True
                
            if not len_markers or (self.original_len - len_markers) >= .97*self.original_len :
                done = True
                
        else:
            reset_pose = True
        
        return observation, len_markers, distance, reset_pose, done
    
    def _reward(self, len_markers, distance, reset_pose, done):
        if done:
            return self.original_len, done
        
        if reset_pose:
            self._start_pose()
            if distance:
                return -10, done
            return -20, done
        
        if len_markers == self.past_len:
            return 0, done
        
        self.past_len = len_markers
        return self.original_len - len_markers, done

    def reset(self):
        
        return self.observation_space
    
    def step(self, action):
        self._moveon(action)
        observation, len_markers, distance, reset_pose, done = self._get_state()
        reward, done = self._reward(len_markers, distance, reset_pose, done)
        print(f'reward : {reward}')
        
        if done:
            self.reset() #reset of airsim
            return observation, done
     
        return observation, done