
from gymnasium.envs.registration import register

from .utils import shapes

register(
    id='gym-airsim/eolic-aereo-point-of-view',
    entry_point='gym-airsim.environments:AereoPointOfView'
)

register(
    id='gym-airsim/eolic-underwater-point-of-view',
    entry_point='gym-airsim.environments:UnderwaterPointOfView'
)

register(
    id='gym-airsim/basic-aereo-point-of-view',
    entry_point='gym-airsim.environments:AereoPointOfView'
)

register(
    id='gym-airsim/basic-underwater-point-of-view',
    entry_point='gym-airsim.environments:UnderwaterPointOfView'
)
