from gym import Env, spaces

from airsim_resources.simulation import ComputerVision, QuadrotorClient
# from airsim_resources.noetic import QuarotorROS


class EnvironmentCV(Env):
    def __init__(self, ip : str):
        self.vehicle = ComputerVision(ip)
        
    def step(self, action):
        self.vehicle.get_state(action)

class EnvironmentQuadrotor(Env):
    def __init__(self, ip : str):
        self.vehicle = QuadrotorClient(ip)
        
    def step(self, action):
        self.vehicle.get_state(action)




# class EnvironmentROS(Env):
#     def __init__(self, vehicle_name):
#         self.vehicle = QuarotorROS(vehicle_name)
        
#     def step(self, action):
#         self.vehicle.get_state(action)

