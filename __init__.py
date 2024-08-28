
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'airsim-helper'))
from ros_helper import PointOfViewTwins, DictToClass
from gymnasium.envs.registration import register

register(
    id='gym-airsim/basic-air-pov',
    entry_point='gym-airsim.environments:BasicAirPOV'
)

register(
    id='gym-airsim/basic-underwater-pov',
    entry_point='gym-airsim.environments:BasicUnderwaterPOV'
)

register(
    id='gym-airsim/basic-hybrid-pov',
    entry_point='gym-airsim.environments:BasicHybridPOV'
)