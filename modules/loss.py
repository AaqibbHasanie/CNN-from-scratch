import numpy as np


class CrossEntropy:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.inputs = None
        self.targets = None

    def forward(self, inputs, targets):

        self.inputs = inputs
        self.targets = targets
        minInput=np.minimum(inputs,1-self.eps)
        SafeInputs = np.maximum(self.eps,minInput)
        return -np.sum(targets * np.log(SafeInputs)) / len(inputs)


    def backward(self):

        return {"d_out": self.inputs - self.targets}
