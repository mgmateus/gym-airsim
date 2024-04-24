from gymnasium.envs.registration import register

register(
    id='gym-airsim/aereo_nbv',
    entry_point='gym_hydrone.envs:HydroneEnv'
)
