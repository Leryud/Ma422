import numpy as np
from src.activation import Softmax
from src.layers.layer import Layer

class Dense(Layer):

    def __init__(self, size, activation):
        super().__init__()
        self.size = size
        self.activation = activation
        self.is_softmax = isinstance(self.activation, Softmax)
        self.cache = {}
        self.w = None
        self.b = None

    def init(self, input_dim):
        self.w = np.random.randn(self.size, input_dim) * np.sqrt(2 / input_dim) # weights
        self.b = np.zeros((1, self.size)) # and biases

    def forward(self, prev, training):
        z = np.dot(prev, self.w.T) + self.b
        a = self.activation.f(z)

        if training:
            self.cache.update({'prev': prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        prev, z, a = (self.cache[key] for key in ('prev', 'z', 'a')) ### DEBUG HERE
        batch_size = prev.shape[0]

        if self.is_softmax:
            y = da * (-a)# TODO: Check if this is correct
            # Because of softmax, da is the derivative of the loss function with respect to the output of the layer

            dz = a - y

        else:
            dz1 = self.activation.df(z, cached_y=a)
            dz = da * dz1

        dw = 1 / batch_size * np.dot(dz.T, prev)
        db = 1 / batch_size * dz.sum(axis=0, keepdims=True)
        prev = np.dot(dz, self.w)

        return prev, dw, db

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.size