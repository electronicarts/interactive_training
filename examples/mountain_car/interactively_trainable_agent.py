""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
The module demonstrates interactively trainable Mountain Car agent based on Markov Ensemble."""
import gym
import numpy as np
from pynput.keyboard import Key
import sys
sys.path.append('../../')

from common.action_info_types import ActionInfoType
from common.box_quantizer import BoxQuantizer
from common.interactive_markov_agent import InteractiveMarkovAgent
from common.markov_ensemble import MarkovEnsembleStack
from common.random_agent import RandomAgent


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")

    # RandomAgent provides default actions:
    random_agent = RandomAgent(env)

    # Ensemble and its parameters:
    min_order = 0
    max_order = 6
    box = [np.array([-1.3, -0.1]), np.array([0.7, 0.1])]  # The actual box: position -1.2	0.6; velocity -0.07	0.07
    quantizers = [BoxQuantizer(box, np.array([2**k, 2**k])).quantize for k in range(3, 10)]
    mes = MarkovEnsembleStack(min_order, max_order, quantizers)

    key_to_action = {
        Key.left: 0,
        Key.space: 1,
        Key.right: 2
    }
    interactive_agent = InteractiveMarkovAgent(key_to_action, random_agent, mes)

    observation, reward, done, info = env.reset(), None, False, None
    interactive_agent.demos.update(None, ActionInfoType.agent, observation, reward, done)

    episode_id = 0
    for _ in range(15000):
        env.render()
        action, _ = interactive_agent.get_action(observation, reward, done, info)
        observation, reward, done, info = env.step(action)

        if done:
            episode_id += 1
            print("Episode", episode_id, "ended.")
            interactive_agent.demos.end_episode(reward, 'mountain_car.pickle')
            observation, reward, done, info = env.reset(), None, False, None
            interactive_agent.demos.update(None, ActionInfoType.agent, observation, reward, done)

    env.close()
