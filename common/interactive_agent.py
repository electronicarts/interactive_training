""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
An interactive agent allows a human player to override the actions of an agent 
supplied in the class instance constructor. """
from common.action_info_types import ActionInfoType


class InteractiveAgent:
    def __init__(self, key_to_action, agent):
        self.key_to_action = key_to_action
        self.auto_agent = agent
        self.latest_key = None

    def get_action(self, observation, reward, done, info):
        action = self.key_to_action.get(self.latest_key)
        if action is not None:
            return action, ActionInfoType.human
        else:
            return self.auto_agent.get_action(observation, reward, done, info)

    def on_press(self, key, mod):
        self.latest_key = key

    def on_release(self, key, mod):
        self.latest_key = None
