"""The module demonstrates interactively trainable Lunar Lander agent based on Markov Ensemble."""
import gym
import numpy as np
from pynput.keyboard import Key
from common.action_info_types import ActionInfoType
from common.box_quantizer import BoxQuantizer
from common.interactive_markov_agent import InteractiveMarkovAgent
from common.markov_ensemble import MarkovEnsembleStack
from common.random_agent import RandomAgent
import time

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
    Key.space: 0,  # No action, stops all thrusters.
    Key.right: 1,  # Right thruster
    Key.up: 2,  # Main thruster
    Key.left: 3  # Left thruster
}
interactive_agent = InteractiveMarkovAgent(key_to_action, random_agent, mes)

observation, reward, done, info = env.reset(), 0, False, None
interactive_agent.demos.update(None, ActionInfoType.agent, observation, reward, done)

episode_id = 0
for _ in range(15000):
    env.render()
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
