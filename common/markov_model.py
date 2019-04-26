"""
A Markov model defines probabilities of transitions in a stochastic system.
Here, using demonstration episodes (games played by a human player), 
we compute the probabilities of the next action in the currently observed state 
and with N actions already taken. The model may not necessarily find such an 
action if the game state together with the sequence of previous actions 
was never observed in the human play-through.

This model doesn't build an explicit dictionary of frequencies; instead, it keeps 
the original user input and collects pointers into it. This way we simplify 
sampling of the original continuous channels which helps to preserve temporal 
coherence of the model playback better and reproduce the distribution of 
the observed inputs.
"""
from collections import defaultdict
import numpy as np
from common.action_info_types import ActionInfoType


class MarkovModel:
    def __init__(self, states, actions, model_order, state_preproc=lambda x: x):
        self.order = model_order
        self.state_preproc = state_preproc
        self.states = [self.state_preproc(s) for s in states]
        self.actions = actions
        self.ngram_pointers = defaultdict(list)
        self.last_action_idx = 0
        self._build_dictionary()

    def _build_dictionary(self):
        """Key is last state and N actions preceding it. """
        sequence_length = len(self.actions)
        for ii in range(sequence_length - self.order):
            ngram = tuple([self.states[ii + self.order - 1]] + self.actions[ii: ii + self.order])
            self.ngram_pointers[ngram].append(ii)

    def next_action(self, states, actions):
        """ Expects states and actions already being preprocessed. """
        last_state = self.state_preproc(states[-1])
        ngram = tuple([last_state] + list(actions[-self.order:]))
        occurrences = self.ngram_pointers.get(ngram, [])
        if not occurrences:
            return None, ActionInfoType.not_found
        pick_id = 0 if len(occurrences) == 1 else np.random.choice(range(len(occurrences)))
        idx = occurrences[pick_id]
        self.last_action_idx = idx
        return self.actions[idx + self.order], ActionInfoType.found
