
import cv2
import importlib  
import os 
import signal
import sys

import numpy as np

from collections import deque
from gymnasium import Env


from gymnasium.spaces import Space, Box
from numpy.typing import NDArray

from .utils import dict_object, container_ip, airsim_launch

airsim_base = importlib.import_module("airsim-base")



class Stack(Space):    
        
    def __init__(self, obs_type : str, stack : int, pre_aug : tuple):
        self.__obs_type = obs_type
        img_shape = (3*stack, *pre_aug)
        tf_shape = (1, 7*stack)

        self.__obs_space = {
                    "rgb": Box(low = 0, high = 255, shape= img_shape, dtype=np.uint8),
                    "depth": Box(low = -2**63, high = 2**63, shape=img_shape, dtype=np.float32),
                    "segmentation": Box(low = 0, high = 255, shape=img_shape, dtype=np.uint8),
                    "point_cloud": Box(low = -2**63, high = 2**63, shape=(1, ), dtype=np.float32),
                    "tf": Box(low = -2**63, high = 2**63, shape=tf_shape, dtype=np.float64)
                }
        
        self.__stack = self._stack(stack)
        Space.__init__(self, shape=[*((img_shape, ) * (len(self.__obs_space.keys()) -1)),tf_shape])
        
    @property
    def stack(self):
        stack = dict()
        for k, obs in self.__stack.items():
            stack[k] = np.concatenate(list(obs), axis=0)
        return stack.values()
    
    @stack.setter
    def stack(self, obs : dict):
        for k, v in obs.items():
            if self.__stack[k]:
                self.__stack[k].append(v)
            else:
                self.__stack[k] = v*3

    
    def _obs_space(self):
        stereo = set({'segmentation', 'point_cloud'})
        stereo_occupancy = set({'segmentation'})
        panoptic = set({'point_cloud'})
        leftover = None
        if self.__obs_type.endswith('stereo'):
            leftover = list(set(self.__obs_space) - stereo)
        elif self.__obs_type.endswith('stereo_occupancy'):
            leftover = list(set(self.__obs_space) - stereo_occupancy)
        elif self.__obs_type.endswith('panoptic'):
            leftover = list(set(self.__obs_space) - panoptic)

        if not leftover:
            obs = dict()
            obs[self.__obs_type] = self.__obs_space[self.__obs_type]
            obs['tf'] = self.__obs_space['tf']
            self.__obs_space = obs
            return self.__obs_space
        
        obs = dict()

        for k in leftover:
            obs[k] = self.__obs_space[k]

        self.__obs_space = obs
        return self.__obs_space
    
    def _stack(self, stack : int):
        dstack = dict()
        for k in self._obs_space():
            dstack[k] = deque([], maxlen=stack)

        return dstack
    

class Simulation:
    action_range = dict_object({
                    "x" : [-60, 60],
                    "y" : [-155, 155],
                    "z" : [-60, 60],
                    "yaw" : [-45, 45],
                    "gimbal_pitch" : [-45, 45]
                })
    
    vehicle = dict_object({
                "start_pose" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "base" : {
                    "name" : "boat_vehicle",
                    "altitude" : 3.0
                }
            })
    twin = dict_object({
                "start_pose" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "base" : {
                    "name" : "boat_vehicle",
                    "altitude" : 3.0
                }
            })
    
    centroide = "centroide"
    
    def __init__(self, observation_type : str, ue4 : str, markers_name : dict_object):        
        self.__ip = container_ip(ue4)
        self.__airsim = airsim_launch(self.__ip)

        self.__twins = airsim_base.PointOfViewTwins(self.ip, self.vehicle, self.twin, observation_type)
        self.__twins.set_detection(markers_name)
        self._parse_settings()

    @property
    def client(self):
        return self.__twins

    @property
    def airsim(self):
        return self.__airsim
    
    @property
    def ip(self):
        return self.__ip

    def _parse_settings(self):
    
        settings = dict({self.__twins.getSettingsString()})
        svehicle_names = list(settings['Vehicles'].keys())
        
        svehicle = settings['Vehicles'][svehicle_names[0]]
        svehicle_name = svehicle_names[0]
        gx, gy, gz, groll, gpitch, gyaw = svehicle['X'], svehicle['Y'], svehicle['Z'], \
                                    svehicle['Roll'], svehicle['Pitch'], svehicle['Yaw']
        svehicle_camera_name = list(svehicle['Cameras'].keys())[0]
        
        svehicle_camera_dim = svehicle['Cameras'][svehicle_camera_name]['CaptureSettings'][0]['Width'],\
                            svehicle['Cameras'][svehicle_camera_name]['CaptureSettings'][0]['Height']  
                            
        svehicle_camera_fov = svehicle['Cameras'][svehicle_camera_name]['CaptureSettings'][0]["FOV_Degrees"]
        
        stwin = settings['Vehicles'][svehicle_names[1]]
        stwin_name = svehicle_names[1]
        gsx, gsy, gsz, gsroll, gspitch, gsyaw = stwin['X'], stwin['Y'], stwin['Z'], \
                                    stwin['Roll'], stwin['Pitch'], stwin['Yaw']
        stwin_camera_name = list(stwin['Cameras'].keys())[0]
        stwin_camera_dim = stwin['Cameras'][stwin_camera_name]['CaptureSettings'][0]['Width'],\
                            stwin['Cameras'][stwin_camera_name]['CaptureSettings'][0]['Height']
        stwin_camera_fov = stwin['Cameras'][stwin_camera_name]['CaptureSettings'][0]["FOV_Degrees"]
                                        
        
        self.vehicle.update({'name' : svehicle_name})
        self.vehicle.update({'global_pose' : [gx, gy, gz, groll, gpitch, gyaw]})
        self.vehicle.update({'camera' : {
                                    'name' : svehicle_camera_name,
                                    'dim' : svehicle_camera_dim,
                                    'fov' : svehicle_camera_fov
                                    }
                            })
        
        self.twin.update({'name' : stwin_name})
        self.twin.update({'global_pose' : [gsx, gsy, gsz, gsroll, gspitch, gsyaw]})
        self.twin.update({'camera' : {
                                    'name' : stwin_camera_name,
                                    'dim' : stwin_camera_dim,
                                    'fov' : stwin_camera_fov
                                    }
                            })


class GymPointOfView(Simulation, Env):
    task_name = "point-of-view"
    max_episode_steps = 200

    @staticmethod
    def set_markers(markers_name : float, markers_num : int):
        m = [ markers_name ]
        m = m + [f"{markers_name}{i}" for i in range(1, markers_num+1)]
        return set(m)

    def __init__(self, observation_type : str, 
                 observation_stack : int, 
                 pre_aug : tuple, 
                 domain : str,
                 ue4 : str, 
                 markers : dict_object,
                 target_range : dict_object):
        
        Simulation.__init__(self, observation_type, ue4, markers.name)
        Env.__init__(self)

        self.observation_space = Stack(observation_type, observation_stack, pre_aug)
        self.action_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.current_step = 0

        self.original_markers_len = markers.num
        self.markers_len =  markers.num
        self.past_markers_len =  markers.num
        self.markers_need_to_visit = self.set_markers(markers.name, markers.num)
        self.markers_backup = self.set_markers(markers.name, markers.num)
        self.range_to_get_target = markers.range_to_get
        self.target_range = target_range
        self.domain = domain
        self.heliport = True if domain.endswith('air') else False

        self.__pre_aug = pre_aug

    def _normalize_action(self, action):

        def _normalize_value(x, min_val, max_val, a, b):
            return ((x - min_val) / (max_val - min_val)) * (b - a) + a
        
        x, y, z, yaw_angle, gimbal_pitch_angle = action
        xmin, xmax = self.action_range.x
        ymin, ymax = self.action_range.y
        zmin, zmax = self.action_range.z
        yaw_min, yaw_max =self.action_range.yaw
        gimbal_pitch_min, gimbal_pitch_max = self.action_range.gimbal_pitch
        
        px = _normalize_value(x, -1, 1, xmin, xmax)
        py = _normalize_value(y, -1, 1, ymin, ymax)
        pz = _normalize_value(-1 * z, -1, 1, zmin, zmax)
        yaw = _normalize_value(yaw_angle, -1, 1, yaw_min, yaw_max)
        gimbal_pitch = _normalize_value(gimbal_pitch_angle, -1, 1, gimbal_pitch_min, gimbal_pitch_max)
        
        return px, py, pz, yaw, gimbal_pitch
    
    def _get_obs(self):
        def _pre_aug_obs_shape(img : NDArray, dim : tuple, type= int):
            if isinstance(type, float):
                img_ = img.copy()
                nan_location = np.isnan(img_)
                img_[nan_location] = np.nanmax(img_)
                norm_image =  (img_)*255./5.
                norm_image[0,0] = 255.
                norm_image = norm_image.astype('uint8')
                norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)
                return cv2.resize(norm_image.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1) 

            return cv2.resize(img.copy(), dim, interpolation = cv2.INTER_AREA).transpose(2, 0, 1)
        
        def _parse_obs(obs : dict):
            _obs = dict()
            for k, v in obs.items():
                if k.endswith('tf'):
                    _obs[k] = v

                elif k.endswith('depth'):
                    _obs[k] = _pre_aug_obs_shape(v, self.__pre_aug, float)

                else:
                    _obs[k] = _pre_aug_obs_shape(v, self.__pre_aug)
            
            return _obs


        observation = _parse_obs(self.client.observation)
        self.observation_space.stack = observation
        return self.observation_space.stack
    
    def _get_info(self):
        info = f"Current markers lenght : {self.markers_len}"
        return {'info' : info}
    
    def _get_state(self):
        
        viewd_markers, distances = self.client.detections()
        d = self.client.detection_distance()
        done = False
        reset_pose = False
        distance = 0
        self.markers_len = self.past_markers_len
       
        if distances:
            self.markers_need_to_visit -= set(viewd_markers)
            self.markers_len = len(self.markers_need_to_visit)
            dmin, dmax = self.range_to_get_target
            
            if (abs(d - min(distances))) < dmin or abs(max(distances) - d) > dmax:
                distance = 1
                reset_pose = True
                
            if not self.markers_len or (self.original_markers_len - self.markers_len) >= .97*self.original_markers_len :
                done = True
                
        else:
            reset_pose = True
        
        return self._get_obs(), self.markers_len, distance, reset_pose, done, self._get_info()
    
    def _reward(self, markers_len, distance, reset_pose, done):
        if done:
            return self.original_markers_len, done
        
        if self.current_step == self.max_episode_steps:
            return 0, True
        
        if reset_pose:
            self.client.go_home()
            if distance:
                return -10, done
            return -20, done
        
        if markers_len == self.past_markers_len:
            return 0, done
        
        self.past_markers_len = markers_len
        return self.original_markers_len - markers_len, done

    def reset_random(self):
        pose = self.client.random_pose(self.action_range.x, self.action_range.y, 
                                       self.target_range.x, self.target_range.y, 
                                       self.centroide, self.heliport)
        self.client.home = pose
        self.client.go_home()

    def reset(self, seed= None, options= None):
        seed = np.random.seed()
        super().reset(seed=seed)
        self.current_step = 0
        
        self.client.go_home()
        self.markers_need_to_visit = self.markers_backup
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        norm_action = self._normalize_action(action)
        self.client.next_point_of_view(norm_action)
        observation, len_markers, distance, reset_pose, done, info = self._get_state()
        reward, done = self._reward(len_markers, distance, reset_pose, done)
        self.current_step += 1
        return observation, reward * 0.01, done, info
    
    def close(self):
        os.killpg(self.airsim, signal.SIGINT)
        sys.exit()  

class AirPointOfView:
    domain = 'air'
    markers = dict_object({'name' : 'Cube'})
    altitude = -1.0

class UnderwaterPointOfView:
    domain_name = 'underwater'
    markers = dict_object({'name' : 'Sphere'})
    altitude = [-30, -6]

class BasicAirPOV(AirPointOfView, GymPointOfView):
    markers = AirPointOfView.markers.update({
                "num" : 79,
                "range_to_get" : [2, 120]
            })
    target_range = dict_object({
                "x" : [50, 170],
                "y" : [50, 170]
            })

    def __init__(self, observation_type : str, 
                 observation_stack : int, 
                 pre_aug : tuple, 
                 max_episode_steps : int,
                 domain : str,
                 ue4 : str) -> None:
        
        AirPointOfView.__init__(self)
        GymPointOfView.__init__(self, observation_type, observation_stack, pre_aug, 
                                domain, ue4, self.markers, self.target_range)
        
        self.max_episode_steps = max_episode_steps

class BasicUnderwaterPOV(UnderwaterPointOfView, GymPointOfView):
    markers = AirPointOfView.markers.update({
                "num" : 73,
                "range_to_get" : [2, 120]
            })
    target_range = dict_object({
                "x" : [50, 170],
                "y" : [50, 170]
            })

    def __init__(self, observation_type : str, 
                 observation_stack : int, 
                 pre_aug : tuple, 
                 max_episode_steps : int,
                 domain : str,
                 ue4 : str) -> None:
        
        AirPointOfView.__init__(self)
        GymPointOfView.__init__(self, observation_type, observation_stack, pre_aug, 
                                domain, ue4, self.markers, self.target_range)
        
        self.max_episode_steps = max_episode_steps