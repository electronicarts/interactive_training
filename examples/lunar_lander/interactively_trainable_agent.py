""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
The module demonstrates interactively trainable Lunar Lander agent based on Markov Ensemble."""
import gym
import time
import numpy as np
import sys
sys.path.append('../../')

from common.action_info_types import ActionInfoType
from common.box_quantizer import BoxQuantizer
from common.interactive_markov_agent import InteractiveMarkovAgent
from common.markov_ensemble import MarkovEnsembleStack
from common.random_agent import RandomAgent


if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    print(env.observation_space)

    # Provides default actions:
    random_agent = RandomAgent(env)

    # Ensemble and its parameters:
    min_order = 0
    max_order = 6
    box = [np.array([-10]*8), np.array([10]*8)]
    quantizers = [BoxQuantizer(box, np.array([2**k]*8)).quantize for k in range(3, 12)]
    mes = MarkovEnsembleStack(min_order, max_order, quantizers)

    # See OpenAI documentation for the definition of the state-action space for the lander environment.
    # https://gym.openai.com/envs/LunarLander-v2/
    key_to_action = {
        32: 0,  # No action, stops all thrusters. Space.
        65363: 1,  # Right thruster. Right arrow.
        65362: 2,  # Main thruster. Up arrow.
        65361: 3  # Left thruster. Left Arrow.
    }
    interactive_agent = InteractiveMarkovAgent(key_to_action, random_agent, mes)

    observation, reward, done, info = env.reset(), 0, False, None
    interactive_agent.demos.update(None, ActionInfoType.agent, observation, reward, done)

    episode_id = 0
    for _ in range(15000):
        env.render()
        env.unwrapped.viewer.window.on_key_press = interactive_agent.on_press
        env.unwrapped.viewer.window.on_key_release = interactive_agent.on_release

        action, _ = interactive_agent.get_action(observation, reward, done, info)
        observation, reward, done, info = env.step(action)
        time.sleep(0.1)  # Simplifies human interaction.

        if done:
            episode_id += 1
            print("Episode", episode_id, "ended.")
            interactive_agent.demos.end_episode(reward, 'lunar_lander.pickle')
            time.sleep(1)  # Simplifies human interaction.
            observation, reward, done, info = env.reset(), 0, False, None
            interactive_agent.demos.update(None, ActionInfoType.agent, observation, reward, done)

    env.close()
