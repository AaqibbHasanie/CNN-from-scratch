import numpy as np


class ReLU:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):

        self.inputs = inputs
        
        return np.maximum(0, inputs)


    def backward(self, d_outputs):

        result = (self.inputs > 0) * d_outputs

        return {"d_out": result}



class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):

        self.inputs = inputs

        return 1 / (1 + np.exp(-inputs))


    def backward(self, d_outputs):

        sigmFun = 1 / (1 + np.exp(-self.inputs))
        derivatived = sigmFun * (1 - sigmFun)

        return {"d_out": derivatived*d_outputs}


class Softmax:

    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):

        self.inputs = inputs 
        maxValue = np.max(inputs, axis=-1, keepdims=True)
        exponentt = np.exp(inputs - maxValue)
        den = np.sum(exponentt, axis=-1, keepdims=True)
        return exponentt / den


    def backward(self, d_outputs):

        return {"d_out": d_outputs}
