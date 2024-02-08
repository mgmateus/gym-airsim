from gym import Env, spaces

from airsim_resources.simulation import QuarotorROS

class Environment(Env):
    def __init__(self, vehicle_name):
        self.vehicle = QuarotorROS(vehicle_name)
        
    def step(self, action):
        self.vehicle.get_state(action)
