import numpy as np

epsilon = 1e-20


class CostFunction:
    def f(self, last, y):
        raise NotImplementedError

    def grad(self, last, y):
        raise NotImplementedError


class SigmoidCrossEntropy(CostFunction):
    def f(self, last, y):
        batch_size = y.shape[0]
        last = np.clip(last, epsilon, 1 - epsilon)  # min(max)
        cost = -np.sum(y * np.log(last) + (1 - y) * np.log(1 - last)) / batch_size
        return cost

    def grad(self, last, y):
        last = np.clip(last, epsilon, 1 - epsilon)  # min(max)
        return -(y / last - (1 - y) / (1 - last))


class SoftmaxCoressentropy(CostFunction):
    def f(self, last, y):
        batch_size = y.shape[0]
        clip = np.clip(last, epsilon, 1 - epsilon)
        cost = -np.sum(y * np.log(clip)) / batch_size
        return cost

    def grad(self, last, y):
        last = np.clip(last, epsilon, 1 - epsilon)
        return -(y / last)


softmax_cross_entropy = SoftmaxCoressentropy()
sigmoid_cross_entropy = SigmoidCrossEntropy()
