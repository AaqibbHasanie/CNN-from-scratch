import numpy as np


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weights = np.random.rand(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32)
        self.bias = np.random.rand(out_channels).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):

        self.inputs = inputs

        x = (inputs.shape[2] - self.kernel_size) // self.stride + 1    # using ((h-f+2p) / s) + 1 -> gives height
        y = (inputs.shape[3] - self.kernel_size) // self.stride + 1    # -> gives weight

        result = np.zeros((inputs.shape[0], self.out_channels, x, y))

        for i in range(x):
            for j in range(y):
                inputTemp = inputs[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]
                result[:, :, i, j] = np.tensordot(inputTemp, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.bias

        return result

    def backward(self, d_outputs):
    
        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )

        dWweights = np.zeros_like(self.weights)
        dWbias = np.sum(d_outputs, axis=(0, 2, 3))
        dWinputs = np.zeros_like(self.inputs)

        for i in range(d_outputs.shape[2]):
            for j in range(d_outputs.shape[3]):

                inputs_slice = self.inputs[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]

                for k in range(self.out_channels):

                    dWweights[k] += np.sum(inputs_slice * d_outputs[:, k, i, j][:, None, None, None], axis=0)
                    dWinputs[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += self.weights[k] * d_outputs[:, k, i, j][:, None, None, None]

        return {"d_weights": dWweights, "d_bias": dWbias,  "d_out": dWinputs}


    def update(self, d_weights, d_bias, learning_rate):

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias


class Flatten:

    def __init__(self):
        self.inputs_shape = None
        self.has_weights = False

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, d_outputs):
        return {"d_out": d_outputs.reshape(self.inputs_shape)}


class LinearLayer:

    def __init__(self, in_features, out_features):
        
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.rand(out_features, in_features).astype(np.float32)
        self.bias = np.random.rand(out_features).astype(np.float32)
        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):

        self.inputs = inputs
        y = np.dot(inputs, self.weights.T) + self.bias

        return y

    def backward(self, d_outputs):

        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )

        derivWinput = np.dot(d_outputs, self.weights)  
        derivWweights = np.dot(d_outputs.T, self.inputs)
        derivWbias = np.sum(d_outputs, axis=0)

        return {"d_weights": derivWweights, "d_bias": derivWbias, "d_out": derivWinput}

    def update(self, d_weights, d_bias, learning_rate):

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
