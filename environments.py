from gym import Env, spaces

from airsim_resources.simulation import ComputerVision, QuadrotorClient


class EnvironmentCV(Env):
    def __init__(self, ip : str):
        self.vehicle = ComputerVision(ip)
        
    def step(self, action):
        self.vehicle.get_state(action)

class EnvironmentQuadrotor(Env):
    def __init__(self, ip : str, vehicle_name : str, camera_name : str, observation : str):
        self.vehicle = QuadrotorClient(ip, vehicle_name, camera_name, observation)
        
    def step(self, action):
        self.vehicle.get_state(action)




