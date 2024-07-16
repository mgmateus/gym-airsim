
from gymnasium.envs.registration import register


register(
    id='gym-airsim/basic-air-pov',
    entry_point='gym-airsim.environments:BasicAirPOV'
)

register(
    id='gym-airsim/basic-underwater-point-of-view',
    entry_point='gym-airsim.environments:BasicUnderwaterPOV'
)
