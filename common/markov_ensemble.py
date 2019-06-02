""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
The classes cover a range of Markov model orders from min_order to max_order, 
constructed from the same sequences of observations and raw actions. Such models 
represent the first step in building models ensemble (see Algorithm 1 in the paper). """
from common.action_info_types import ActionInfoType
from common.markov_model import MarkovModel


class MarkovEnsemble:
    def __init__(self, observations, actions, min_order, max_order, quantizers):
        self.models = [MarkovModel(observations, actions, m, q)
                       for m in range(min_order, max_order)
                       for q in quantizers]

    def get_action(self, observations, actions):
        for mcm in self.models[::-1]:
            action, found = mcm.next_action(observations, actions)
            if found is True:
                return action, ActionInfoType.found
        return None, ActionInfoType.not_found


class MarkovEnsembleStack:
    def __init__(self, min_order, max_order, quantizers):
        self.min_order = min_order
        self.max_order = max_order
        self.quantizers = quantizers
        self.stack = []

    def add_demo(self, observations, actions):
        self.stack.append(MarkovEnsemble(
            observations, actions, self.min_order, self.max_order, self.quantizers
        ))

    def get_action(self, observations, actions):
        for model in self.stack:
            action, action_info = model.get_action(observations, actions)
            if action_info:
                return action, action_info
        return None, ActionInfoType.not_found

