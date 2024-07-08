
import cv2
import math
import os
import rospy
import random
import signal
import sys
import time

import numpy as np

from collections import deque
from math import dist
from numpy.typing import NDArray
from typing import List, Tuple
from gymnasium import Env, spaces

from .utils import container_ip, parse_cfg, airsim_launch

sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))

from airsim_base.types import ImageType
from ros_helper import DualActPose


def normalize_value(x, min_val, max_val, a, b):
    return ((x - min_val) / (max_val - min_val)) * (b - a) + a

def pre_aug_obs_shape(img : NDArray, dim : tuple):
        return cv2.resize(img.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1)

class ObservationSpace:
    def __init__(self, observation_type : str , image_dim : tuple, pack_len : int, pre_aug : tuple = None) -> None:
        self.observation_type = observation_type
        self.pack_len = pack_len
        self.pre_aug = pre_aug
        w, h = image_dim
        self.__observation_space = {
                    "rgb": spaces.Box(low = 0, high = 255, shape=(3*pack_len, 100, 100), dtype=np.uint8),
                    "depth": spaces.Box(low = -2**63, high = 2**63, shape=(pack_len, h*3, w), dtype=np.float32),
                    "segmentation": spaces.Box(low = 0, high = 255, shape=(3*pack_len, h*3, w), dtype=np.uint8),
                    "point_cloud": spaces.Box(low = -2**63, high = 2**63, shape=(1, ), dtype=np.float32),
                    "tf": spaces.Box(low = -2**63, high = 2**63, shape=(1, 7*pack_len), dtype=np.float64)
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
            self.__rgb['rgb'].append(pre_aug_obs_shape(observation['rgb'], self.pre_aug) 
                                     if self.pre_aug else observation['rgb'])
            self.__rgb['tf'].append(observation['tf'])
            self.__rgb['rgb'] = self.__rgb['rgb']*3
            self.__rgb['tf'] = self.__rgb['tf']*3
            
        self.__rgb['rgb'].append(pre_aug_obs_shape(observation['rgb'], self.pre_aug) 
                                 if self.pre_aug else observation['rgb'])
        self.__rgb['tf'].append(observation['tf'])

    @property
    def stereo(self):
        return self.__stereo
    
    @stereo.setter
    def stereo(self, observation):
        if not self.__stereo['rgb']:
            self.__stereo['rgb'].append(pre_aug_obs_shape(observation['rgb'], self.pre_aug) 
                                     if self.pre_aug_obs_shape else observation['rgb'] )
            self.__stereo['depth'].append(pre_aug_obs_shape(observation['depth'], self.pre_aug) 
                                     if self.pre_aug_obs_shape else observation['depth'])
            self.__stereo['tf'].append(observation['tf'])
            self.__stereo['rgb'] = self.__stereo['rgb']*3
            self.__stereo['depth'] = self.__stereo['depth']*3
            self.__stereo['tf'] = self.__stereo['tf']*3
            
        self.__stereo['rgb'].append(pre_aug_obs_shape(observation['rgb'], self.pre_aug) 
                                     if self.pre_aug_obs_shape else observation['rgb'] )
        self.__stereo['depth'].append(pre_aug_obs_shape(observation['depth'], self.pre_aug) 
                                     if self.pre_aug_obs_shape else observation['depth'])
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
    
class PointOfView:
    @staticmethod
    def set_markers(markers : dict):
        m = [markers['name']]
        m = m + [f"{markers['name']}{i}" for i in range(1, markers['num']+1)]
        return set(m)
    
    def __init__(self, ue4 : str, config : dict, node : str):        
        ip = container_ip(ue4)
        self.__airsim = airsim_launch(ip)
        rospy.init_node(node)
        
        self.__action_range = config['action_range']
        self.__target_range = config['simulation']['target_range']
        self.__centroide = config['simulation']['centroide']
        self.__markers = config['markers']
        
        self.pov = DualActPose(ip, config['simulation']['vehicle'], config['simulation']['shadow'], config['observation'])
        self.original_len = self.__markers['num']
        self.len_markers = self.__markers['num']
        self.past_len = self.__markers['num']
        self.markers_need_to_visit = self.set_markers(self.__markers)
        
        self.pov.set_detection(self.__markers['name'])
        
    @property
    def airsim(self):
        return self.__airsim    
    
    @property
    def action_range(self):
        return self.__action_range
    
    @property
    def target_range(self):
        return self.__target_range
        
    @property
    def centroide(self):
        return self.__centroide
    
    @property
    def markers(self):
        return self.__markers
    
    def _normalize_action(self, action):
        x, y, z, yaw_angle, gimbal_pitch_angle = action
        xmin, xmax = self.__action_range['x']
        ymin, ymax = self.__action_range['y']
        zmin, zmax = self.__action_range['z']
        yaw_min, yaw_max = self.__action_range['yaw']
        gimbal_pitch_min, gimbal_pitch_max = self.__action_range['gimbal_pitch']
        
        px = normalize_value(x, -1, 1, xmin, xmax)
        py = normalize_value(y, -1, 1, ymin, ymax)
        pz = normalize_value(-1 * z, -1, 1, zmin, zmax)
        yaw = normalize_value(yaw_angle, -1, 1, yaw_min, yaw_max)
        gimbal_pitch = normalize_value(gimbal_pitch_angle, -1, 1, gimbal_pitch_min, gimbal_pitch_max)
        
        return px, py, pz, yaw, gimbal_pitch

class AereoPointOfView(PointOfView, Env):
    def __init__(self, ue4: str, env_name : str, observation : str, pre_aug : tuple = (100,100)):
        config = parse_cfg(env_name, observation)
        PointOfView.__init__(self, ue4, config, env_name)
        Env.__init__(self)
        
        self.pack = ObservationSpace(config['observation'], config['simulation']['vehicle']['camera']['dim'], 3, pre_aug)
        self.observation_space = self.pack.observation_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.max_episode_steps = config['max_episode_steps']
        self.current_step = 0
        
    def _get_obs(self):
        observation = self.pov.observation
        self.pack.packing(observation)
        return self.pack.concat()
    
    def _get_info(self):
        info = f"Current markers lenght : {self.len_markers}"
        return {'info' : info}
    
    def _get_state(self):
        
        viewd_markers, distances = self.pov.detections()
        d = self.pov.detection_distance()
        done = False
        reset_pose = False
        distance = 0
        self.len_markers = self.past_len
        # if self.pov.altitude <= self.pov.altitude_min:
        #     return self._get_obs(), self.len_markers, distance, True, done, self._get_info() 
       
        if distances:
            self.markers_need_to_visit -= set(viewd_markers)
            self.len_markers = len(self.markers_need_to_visit)
            dmin, dmax = self.markers['d']
            
            if (abs(d - min(distances))) < dmin or abs(max(distances) - d) > dmax:
                distance = 1
                reset_pose = True
                
            if not self.len_markers or (self.original_len - self.len_markers) >= .97*self.original_len :
                done = True
                
        else:
            reset_pose = True
        
        return self._get_obs(), self.len_markers, distance, reset_pose, done, self._get_info()
    
    def reset_random(self):
        pose = self.pov.random_pose(self.action_range['x'], self.action_range['y'], self.target_range['x'], self.target_range['y'], self.centroide)
        self.pov.random_base_pose(pose)
        self.pov.home = pose
        self.pov.go_home()
    
    def _reward(self, len_markers, distance, reset_pose, done):
        if done:
            return self.original_len, done
        
        if self.current_step == self.max_episode_steps:
            return 0, True
        
        if reset_pose:
            self.pov.go_home()
            if distance:
                return -10, done
            return -20, done
        
        if len_markers == self.past_len:
            return 0, done
        
        self.past_len = len_markers
        return self.original_len - len_markers, done
    
    def reset(self, seed= None, options= None):
        seed = np.random.seed()
        super().reset(seed=seed)
        self.current_step = 0
        
        self.pov.go_home()
        self.markers_need_to_visit = self.set_markers(self.markers) 
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        norm_action = self._normalize_action(action)
        self.pov.next_point_of_view(norm_action)
        observation, len_markers, distance, reset_pose, done, info = self._get_state()
        reward, done = self._reward(len_markers, distance, reset_pose, done)
        self.current_step += 1
        return observation, reward * 0.01, done, info
    
    def close(self):
        os.killpg(self.airsim, signal.SIGINT)
        sys.exit()  
    
class UnderwaterPointOfView(PointOfView, Env):
    def __init__(self, ue4: str, env_name : str, observation : str):
        config = parse_cfg(env_name, observation)
        PointOfView.__init__(self, ue4, config, env_name)
        Env.__init__(self)
        
        self.pack = ObservationSpace(config['observation'], config['simulation']['vehicle']['camera']['dim'], 3)
        self.observation_space = self.pack.observation_space
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.max_episode_steps = config['max_episode_steps']
        self.current_step = 0
        
    def _get_obs(self):
        observation = self.pov.observation
        self.pack.packing(observation)
        return self.pack.concat()
    
    def _get_info(self):
        info = f"Current markers lenght : {self.len_markers}"
        return {'info' : info}
    
    def _get_state(self):
        viewd_markers, distances = self.pov.detections()
        d = self.pov.detection_distance()
        
        done = False
        reset_pose = False
        distance = 0
        self.len_markers = self.past_len
        
        # if self.pov.altitude <= self.pov.altitude_min or self.pov.altitude >= self.pov.altitude_max:
        #     return self._get_obs(), self.len_markers, distance, True, done, self._get_info() 
       
        if distances:
            self.markers_need_to_visit -= set(viewd_markers)
            self.len_markers = len(self.markers_need_to_visit)
            dmin, dmax = self.markers['d']
            
            if (abs(d - min(distances))) < dmin or abs(max(distances) - d) > dmax:
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
        
        if self.current_step == self.max_episode_steps:
            return 0, True
        
        if reset_pose:
            self.pov.go_home()
            if distance:
                return -10, done
            return -20, done
        
        if len_markers == self.past_len:
            return 0, done
        
        self.past_len = len_markers
        return self.original_len - len_markers, done
    
    def reset_random(self):
        pose = self.pov.random_pose(self.action_range['x'], self.action_range['y'], self.target_range['x'], self.target_range['y'], self.centroide)
        self.pov.home = pose
        self.pov.go_home()
    
    def reset(self, seed= None, options= None):
        seed = np.random.seed()
        super().reset(seed=seed)
        self.current_step = 0
        self.pov.go_home()
        self.markers_need_to_visit = self.set_markers(self.markers) 
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        norm_action = self._normalize_action(action)
        self.pov.next_point_of_view(norm_action)
        observation, len_markers, distance, reset_pose, done, info = self._get_state()
        reward, done = self._reward(len_markers, distance, reset_pose, done)
        self.current_step += 1
        return observation, reward * 0.01, done, info
    
    def close(self):
        os.killpg(self.airsim, signal.SIGINT)
        sys.exit() 