""" The module provides simple uniform quantization under the assumption that 
the range of state variables is known. We expect observations represented 
as a numpy array and quantize them into bins of size defined by fidelity. """
import numpy as np


class BoxQuantizer:
    def __init__(self, box, fidelity):
        self.box_low = box[0]
        self.box_high = box[1]
        self.size = box[1] - box[0]
        self.fidelity = fidelity

    def quantize(self, vec):
        return tuple(np.floor((vec - self.box_low)/self.size * self.fidelity))

if __name__ == "__main__":
    box = [np.array([-2, -1]), np.array([2, 1])]
    fidelity = np.array([20, 5])
    bq = BoxQuantizer(box, fidelity)
    vec = np.array([-0, 0.2])
    print(bq.quantize(vec))
