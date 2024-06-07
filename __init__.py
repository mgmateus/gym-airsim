from gymnasium.envs.registration import register

register(
    id='gym-airsim/eolic-aereo-point-of-view',
    entry_point='environments:AereoPointOfView'
)

register(
    id='gym-airsim/eolic-underwater-point-of-view',
    entry_point='environments:UnderwaterPointOfView'
)
