""" Enumeration of possible action info types."""


class ActionInfoType:
    human = "human"  # Human provided the action
    agent = None  # An agent other than Markov agent provided the action
    found = True  # Markov agent provided the action
    not_found = False  # Markov agent couldn't provide the action
