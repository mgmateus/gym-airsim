
import cv2
import math
import os
import rospy
import random
import sys
import time

import numpy as np

from collections import deque
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


class StockPile:
    
    def __init__(self, observation_type : str , image_dim : tuple, pack_len : int) -> None:
        self.observation_type = observation_type
        self.pack_len = pack_len
        w, h = image_dim
        self.__observation_space = {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3, h*3, w), dtype=np.uint8),
                    "depth": spaces.Box(low = -2**63, high = 2**63, shape=(1, h*3, w), dtype=np.float32),
                    "segmentation": spaces.Box(low = 0, high = 255, shape=(3, h*3, w), dtype=np.uint8),
                    "point_cloud": spaces.Box(low = -2**63, high = 2**63, shape=(1, ), dtype=np.float32),
                    "tf": spaces.Box(low = -2**63, high = 2**63, shape=(7*3,), dtype=np.float64)
                }
            
        self.__rgb = {'rgb' : deque([], maxlen=pack_len), 'tf' : deque([], maxlen=pack_len)}
        self.__stereo = {'rgb' : deque([], maxlen=pack_len), 
                         'depth' : deque([], maxlen=pack_len), 
                         'tf' : deque([], maxlen=pack_len)}
        self.__panoptic = {'rgb' : deque([], maxlen=pack_len), 
                           'depth' : deque([], maxlen=pack_len), 
                           'segmentation' : deque([], maxlen=pack_len), 
                           'tf' : deque([], maxlen=pack_len)}
        self.__stereo_occupancy = {'rgb' : deque([], maxlen=pack_len), 
                                                                      'depth' : deque([], maxlen=pack_len), 
                                                                      'point_cloud' : deque([], maxlen=pack_len), 
                                                                      'tf' : deque([], maxlen=pack_len)}        
        self.__panoptic_occupancy = {'rgb' : deque([], maxlen=pack_len), 
                                                                          'depth' : deque([], maxlen=pack_len), 
                                                                          'segmentation' : deque([], maxlen=pack_len), 
                                                                          'point_cloud' : deque([], maxlen=pack_len),
                                                                          'tf' : deque([], maxlen=pack_len)}
        
    @property
    def observation_space(self):
        to_remove =  self.__getattribute__(self.observation_type).keys() ^ self.__observation_space.keys()
        _ = [self.__observation_space.pop(k) for k in to_remove]
        return spaces.Dict(self.__observation_space)
        
    @property
    def rgb(self):
        return self.__rgb
    
    @rgb.setter
    def rgb(self, observation):
        if not self.__rgb['rgb']:
            self.__rgb['rgb'].append(observation['rgb'])
            self.__rgb['tf'].append(observation['tf'])
            self.__rgb['rgb'] = self.__rgb['rgb']*3
            self.__rgb['tf'] = self.__rgb['tf']*3
            
        self.__rgb['rgb'].append(observation['rgb'])
        self.__rgb['tf'].append(observation['tf'])
        
    @property
    def stereo(self):
        return self.__stereo
    
    @stereo.setter
    def stereo(self, observation):
        if not self.__stereo['rgb']:
            self.__stereo['rgb'].append(observation['rgb'])
            self.__stereo['depth'].append(observation['depth'])
            self.__stereo['tf'].append(observation['tf'])
            self.__stereo['rgb'] = self.__stereo['rgb']*3
            self.__stereo['depth'] = self.__stereo['depth']*3
            self.__stereo['tf'] = self.__stereo['tf']*3
            
        self.__stereo['rgb'].append(observation['rgb'])
        self.__stereo['depth'].append(observation['depth'])
        self.__stereo['tf'].append(observation['tf'])
        
    @property
    def stereo_occupancy(self):
        return self.__stereo_occupancy
    
    @stereo_occupancy.setter
    def stereo_occupancy(self, observation):
        if not self.__rgb['rgb']:
            self.__stereo_occupancy['rgb'].append(observation['rgb'])
            self.__stereo_occupancy['depth'].append(observation['depth'])
            self.__stereo_occupancy['point_cloud'].append(observation['point_cloud'])
            self.__stereo_occupancy['tf'].append(observation['tf'])
            self.__stereo_occupancy['rgb'] = self.__stereo_occupancy['rgb']*3
            self.__stereo_occupancy['depth'] = self.__stereo_occupancy['depth']*3
            self.__stereo_occupancy['point_cloud'] = self.__stereo_occupancy['point_cloud']*3
            self.__stereo_occupancy['tf'] = self.__stereo_occupancy['tf']*3
            
        self.__stereo_occupancy['rgb'].append(observation['rgb'])
        self.__stereo_occupancy['depth'].append(observation['depth'])
        self.__stereo_occupancy['point_cloud'].append(observation['point_cloud'])
        self.__stereo_occupancy['tf'].append(observation['tf'])
    
        
    @property
    def panoptic(self):
        return self.__panoptic
    
    @panoptic.setter
    def panoptic(self, observation):
        if not self.__panoptic['rgb']:
            self.__panoptic['rgb'].append(observation['rgb'])
            self.__panoptic['depth'].append(observation['depth'])
            self.__panoptic['segmentation'].append(observation['segmentation'])
            self.__panoptic['tf'].append(observation['tf'])
            self.__panoptic['rgb'] = self.__panoptic['rgb']*3
            self.__panoptic['depth'] = self.__panoptic['depth']*3
            self.__panoptic['segmentation'] = self.__panoptic['segmentation']*3
            self.__panoptic['tf'] = self.__panoptic['tf']*3
            
        self.__panoptic['rgb'].append(observation['rgb'])
        self.__panoptic['depth'].append(observation['depth'])
        self.__panoptic['segmentation'].append(observation['segmentation'])
        self.__panoptic['tf'].append(observation['tf'])
        
    @property
    def panoptic_occupancy(self):
        return self.__panoptic_occupancy
    
    @panoptic_occupancy.setter
    def panoptic_occupancy(self, observation):
        if not self.__rgb['rgb']:
            self.__panoptic_occupancy['rgb'].append(observation['rgb'])
            self.__panoptic_occupancy['depth'].append(observation['depth'])
            self.__panoptic_occupancy['segmentation'].append(observation['segmentation'])
            self.__panoptic_occupancy['point_cloud'].append(observation['point_cloud'])
            self.__panoptic_occupancy['tf'].append(observation['tf'])
            self.__panoptic_occupancy['rgb'] = self.__panoptic_occupancy['rgb']*3
            self.__panoptic_occupancy['depth'] = self.__panoptic_occupancy['depth']*3
            self.__panoptic_occupancy['segmentation'] = self.__panoptic_occupancy['segmentation']*3
            self.__panoptic_occupancy['point_cloud'] = self.__panoptic_occupancy['point_cloud']*3
            self.__panoptic_occupancy['tf'] = self.__panoptic_occupancy['tf']*3
            
        self.__panoptic_occupancy['rgb'].append(observation['rgb'])
        self.__panoptic_occupancy['depth'].append(observation['depth'])
        self.__panoptic_occupancy['segmentation'].append(observation['segmentation'])
        self.__panoptic_occupancy['point_cloud'].append(observation['point_cloud'])
        self.__panoptic_occupancy['tf'].append(observation['tf'])
        
        
    def packing(self, observation : dict):
        self.__setattr__(self.observation_type, observation)
        
    def concat(self):
        pack = self.__getattribute__(self.observation_type)
        return dict(map(lambda k, obs : (k, np.concatenate(list(obs), axis=0)), pack.keys(), pack.values()))
    
    def __call__(self) -> dict:
        return self.__getattribute__(self.observation_type)
    
    

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
                              vehicle_['camera']['dim'],
                              vehicle_['camera']['fov'], 
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
        super(PositionNBV, self).__init__()
        node = f"gym-{config['name']}-{config['observation']}"
        rospy.init_node(node)
        
        self.domain = config['domain']
        self.vehicle, self.shadow = self.vs_start(ip, config)
        self.vehicle_cfg = config['vehicle']
        self.action_range = config['action_range']
        self.target_range = config['target_range']
        self.altitude_range = config['altitude_range']
        self.centroide = config['centroide']
        self.markers = config['markers']
        self.original_len = config['markers']['num']
        self.len_markers = config['markers']['num']
        self.past_len = config['markers']['num']
        
        self.markers_need_to_visit = self.set_markers(self.markers['name'], self.markers['num'])
        self.pack = StockPile(config['observation'], self.vehicle_cfg['camera']['dim'], 3)
        self.max_episode_steps = config['max_episode_steps']
        
        self.observation_space = self.pack.observation_space
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
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
        
        zmin, _ = self.altitude_range
        
        px = random_choice((axmin - txmin, txmin), (txmax, txmax + axmax))
        py = random_choice((aymin - tymin, tymin), (tymax, tymax + aymax))
        pz = zmin-3
        

        centroide_pose = self.vehicle.objectp2list(self.centroide)
        tf = self.vehicle.tf()
        vehicle_position = [px, py, tf[2]]
        vehicle_e_orientation = quaternion_to_euler(tf[3:])
        vehicle_pose = vehicle_position + vehicle_e_orientation
        
        
        yaw = theta(vehicle_pose, centroide_pose[:3])
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
        
        viewd_markers, distances = self.vehicle.simGetDetectedMeshesDistances(self.shadow['camera']['name'], ImageType.Scene,vehicle_name=self.shadow['name'])
        d = self._wshadow_distance()
        done = False
        reset_pose = False
        distance = 0
        self.len_markers = self.past_len
        if distances:
            self.markers_need_to_visit -= set(viewd_markers)
            self.len_markers = len(self.markers_need_to_visit)
            
            if (d - min(distances)) < 30 or (d - max(distances)) > 120:
                distance = 1
                reset_pose = True
                
            if not self.len_markers or (self.original_len - self.len_markers) >= .97*self.original_len :
                done = True
                
        else:
            reset_pose = True
        
        return self._get_obs(), self.len_markers, distance, reset_pose, done, self._get_info()
    
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
    
    def _get_obs(self):
        observation = self.vehicle.get_observation()
        self.pack.packing(observation)
        return self.pack.concat()
    
    def _get_info(self):
        info = f"Current markers lenght : {self.len_markers}"
        return {'info' : info}

    def reset(self, seed= None, options= None):
        seed = np.random.seed()
        super().reset(seed=seed)
        
        self._start_pose()
        self.markers_need_to_visit = self.set_markers(self.markers['name'], self.markers['num']) 
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        self._moveon(action)
        observation, len_markers, distance, reset_pose, done, info = self._get_state()
        reward, done = self._reward(len_markers, distance, reset_pose, done)
        
        return observation, reward, done, info