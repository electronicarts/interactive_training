""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
DemoRecorder instance records observations and actions, keeps separate 
records for human demonstrations. It can save episodes as native pickled files 
with references only to numpy and defaultdict. We use these demonstrations 
to build Markov Ensembles. """
import pickle
from collections import defaultdict
from common.action_info_types import ActionInfoType


class DemoRecorder:
    ended = 1
    in_progress = 2
    no_update = 3

    def __init__(self):
        self.curr_episode = defaultdict(list)
        self.episodes = []
        self.curr_demo = defaultdict(list)
        self.demos = []

    def end_episode(self, reward, save_file="episodes.pickle"):
        self.rewards.append(reward)
        print('Total reward:', sum(self.rewards))
        if self.curr_demo:
            self.demos.append(self.curr_demo)
            self.curr_demo = defaultdict(list)
        self.episodes.append(self.curr_episode)
        self.curr_episode = defaultdict(list)
        if save_file:
            with open(save_file, "wb") as outfile:
                pickle.dump(self.episodes, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def observations(self):
        return self.curr_episode['observations']

    @property
    def actions(self):
        return self.curr_episode['actions']

    @property
    def action_infos(self):
        return self.curr_episode['action_infos']

    @property
    def rewards(self):
        return self.curr_episode['rewards']

    def update(self, action, action_info, observation, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.action_infos.append(action_info)
        self.rewards.append(reward)

        human_input_stopped = len(self.action_infos) > 1 and \
                              self.action_infos[-2] == ActionInfoType.human and \
                              action_info == ActionInfoType.agent

        if self.curr_demo and human_input_stopped:
            self.demos.append(self.curr_demo)
            self.curr_demo = defaultdict(list)
            return self.ended

        if action_info is not None and not done:
            self.curr_demo["observations"].append(observation)
            self.curr_demo["actions"].append(action)
            return self.in_progress

        return self.no_update
