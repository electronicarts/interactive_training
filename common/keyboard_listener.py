""" Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
The module provides a minimal wrapper around pyinput keyboard input. 
See https://pythonhosted.org/pynput/keyboard.html for the original keyboard 
listener usage example. We are interested only in a single key and do not expect 
any multiple keys input. """
from pynput.keyboard import Key, Listener


class KeyboardListener:
    def __init__(self):
        self.latest_key = None
        self.listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release)
        self.listener.start()

    def __del__(self):
        self.listener.join()

    def _on_press(self, key):
        self.latest_key = key

    def _on_release(self, key):
        self.latest_key = None
