from gym import Env, spaces

# from airsim_resources.simulation import ComputerVision, QuadrotorClient
from .airsim_resources.noetic import Position, Trajectory


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


class EnvironmentPositionNBV(Env):
    def __init__(self, ip : str, 
                 vehicle_name : str, 
                 camera_name : str, 
                 observation_type : str):
        
        self.vehicle = Position(ip, vehicle_name, camera_name, observation_type)   
        
    def step(self, action):
        state = self.vehicle.get_state(action)
        return state

