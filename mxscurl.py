#!/usr/bin/env python3

import time
import os

import numpy as np

from environments import EnvironmentCV, EnvironmentQuadrotor

if __name__ == "__main__":
    ip = os.environ['UE4_IP']
    env = EnvironmentQuadrotor(ip, 'Hydrone', 'stereo', 'depth')
    
    env.vehicle.takeoffAsync().join()
    action = np.array([-.5, .9, -1, 0.01, 0.001, -0.001])
    for i in range(30000):
        env.step(action)

    
        