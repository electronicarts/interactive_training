""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
InteractiveMarkovAgent class combines interactive player input with the input from the ensemble model. 
Also, it updates the ensemble as needed to capture new demonstrations from the player. """
from common.action_info_types import ActionInfoType
from common.interactive_agent import InteractiveAgent
from common.demo_recorder import DemoRecorder


class InteractiveMarkovAgent(InteractiveAgent):
    def __init__(self, key_to_action, agent, ensemble_stack):
        super().__init__(key_to_action, agent)
        self.demos = DemoRecorder()
        self.ensemble = ensemble_stack

    def get_action(self, observation, reward, done, info):
        action = None
        action_info = None

        # Check if there is human input and record that as the next action to take
        human_action = self.key_to_action.get(self.kbd_listener.latest_key)
        if human_action is not None:
            action = human_action
            action_info = ActionInfoType.human

        # If there is no human input, check for an action computed by the ensemble model
        if action is None:
            ensemble_action, action_found = self.ensemble.get_action(self.demos.observations, self.demos.actions)
            if action_found:
                action = ensemble_action
                action_info = ActionInfoType.found

        # If ensemble didn't produce an action, let the auto play agent to produce one:
        if action is None:
            action, action_info = self.auto_agent.get_action(observation, reward, done, info)

        # Record action regardless of its origin and update ensemble as needed
        demo_status = self.demos.update(action, action_info, observation, reward, done)
        if DemoRecorder.ended == demo_status or (action_info == "human" and done):
                self.ensemble.add_demo(self.demos.demos[-1]["observations"],
                                       self.demos.demos[-1]["actions"])

        return action, action_info
