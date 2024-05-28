
import argparse
import json
import os
import signal
import sys
import time
import wandb
import yaml

import gymnasium as gym
import numpy as np

from utils import container_ip, process_cfg, airsim_launch, rtabmap_launch
from environments import PositionNBV

from gymnasium.envs.registration import register

register(
    id='gym-airsim/hybrid-position-nbv',
    entry_point='environments:PositionNBV'
)
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration')
    
    parser.add_argument('--mode', type= str, required=False, default='train', help='Execution mode.')
    parser.add_argument('--agent', type= str, required=False, default='curl', help='Agent class.')
    parser.add_argument('--env', type= str, required=False, default='semisub-position-aereo', help='Gym environment name.')
    parser.add_argument('--container_simulation', type= str, required=False, default='ue4', help='Docker container name to Unreal.')
    args = parser.parse_args()
    
    ip = container_ip(args.container_simulation)      
    mode, env_config = process_cfg(args.mode, args.env) 
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{args.agent}-{env_config['name']}-{env_config['observation']}",

        # track hyperparameters and run metadata
        config= mode
    )
    
    airsim = airsim_launch(ip)
    # wait to connect into airsim...
    
    # # rtabmap = subprocess_launch(reconstruction)
    # # time.sleep(15)
    # for i in range(10):
    #     print(f"step - {i}")
    #     action = np.array([0, 0, .1, 0, 0], dtype=np.float32)
    #     env.step(action)
    #     # env._random_vehicle_pose(True, True)
    #     time.sleep(0.5)

    # # os.killpg(rtabmap.pid, signal.SIGINT)
    
    
    #################################################################
    env = gym.make('gym-airsim/hybrid-position-nbv', ip=ip, config=env_config)
    observation = env.reset()
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    
    for step in range(6):
        # evaluate agent periodically

        if step % mode['eval_freq'] == 0:
            wandb.log({"episode": episode, "step": step})
            pass
            # evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            # if args.save_model:
            #     agent.save_curl(model_dir, step)
            # if args.save_buffer:
            #     replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                wandb.log({"duration": time.time() - start_time, "step": step})
                start_time = time.time()
                wandb.log({"episode_reward": episode_reward, "step": step})
                
            obs, _= env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            
            wandb.log({"episode": episode, "step": step})


        # sample action for data collection
        if step < mode['init_steps']:
            action = env.action_space.sample()
        else:
            # with utils.eval_mode(agent):
            #     action = agent.sample_action(obs)
            pass

        # run training update
        if step >= mode['init_steps']:
            num_updates = 1 
            for _ in range(num_updates):
                # agent.update(replay_buffer, L, step)
                pass

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env.unwrapped.max_episode_steps else float(done)
        episode_reward += reward
        # replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
        
    os.killpg(airsim.pid, signal.SIGINT)