import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__) + '/../'))

import numpy as np
import torch
import random

import gym 
import gym_pathplan
gym.logger.set_level(40) 

import fixed_seed
import trainer
import sac

def main():
    ENV_ID = 'STLPathPlan-v0' # nopreprocess
    #ENV_ID = 'STLPathPlan-v1' # preprocess
    SEED = 14 # 0,1,...,14

    NUM_STEPS = 6 * 10 ** 5
    EVAL_INTERVAL = 10 ** 4
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    REPLAY_BUFFER_SIZE = 10**5
    TAU = 0.01
    REWARD_SCALE = 1.0
    NUM_EVAL_EPISODES = 100

    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)

    # set seed for numpy, random, pytorch 
    fixed_seed.fixed_seed_function(SEED)
    # set for Gym 
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env_test.seed(2**31-SEED)
    env_test.action_space.seed(2**31-SEED)
    env_test.observation_space.seed(2**31-SEED)

    print(env.observation_space.shape)
    print(env.extended_state_space.shape)
    algo = sac.SAC(
        state_shape=env.extended_state_space.shape,
        action_shape=env.action_space.shape,
        seed=SEED,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr_actor=LEARNING_RATE_ACTOR,
        lr_critic=LEARNING_RATE_CRITIC,
        replay_size=REPLAY_BUFFER_SIZE,
        tau=TAU,
        reward_scale=REWARD_SCALE,
        auto_coef=True,
    )

    # Define Trainer
    SAC_trainer = trainer.Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        seed=SEED,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL,
        num_eval_episodes=NUM_EVAL_EPISODES,
    )

    SAC_trainer.train() # Learning
    SAC_trainer.plot() # Result

    env.close()
    env_test.close()

if __name__ == "__main__":
    main()