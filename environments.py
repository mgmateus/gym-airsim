from gym import Env, spaces

# from airsim_resources.simulation import ComputerVision, QuadrotorClient
from .airsim_resources.noetic import QuarotorStereoROS


# class EnvironmentCV(Env):
#     def __init__(self, ip : str):
#         self.vehicle = ComputerVision(ip)
        
#     def step(self, action):
#         self.vehicle.get_state(action)

# class EnvironmentQuadrotor(Env):
#     def __init__(self, ip : str, vehicle_name : str, camera_name : str, observation : str):
#         self.vehicle = QuadrotorClient(ip, vehicle_name, camera_name, observation)
        
#     def step(self, action):
#         self.vehicle.get_state(action)


class EnvironmentROS(Env):
    def __init__(self, ip : str, 
                 vehicle_name : str, 
                 camera_name : str, 
                 observation : str,
                 vertices_path : str, 
                 vertices_name : str):
        
        self.vehicle = QuarotorStereoROS(ip, 
                                         vehicle_name, 
                                         camera_name, 
                                         observation, 
                                         vertices_path, 
                                         vertices_name)
        
    def step(self, action):
        state = self.vehicle.get_state(action)
        return state

