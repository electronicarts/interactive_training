""" Random agent is the base for other agents and allows base line random action sampling."""
from common.action_info_types import ActionInfoType


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation, reward, done, info):
        """ Can return an additional object describing the action. """
        return self.env.action_space.sample(), ActionInfoType.agent

