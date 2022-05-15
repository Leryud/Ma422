import numpy as np
from src.optimizer import gradient_descent


class NeuralNetwork:

    def __init__(self, input_dim, layers, cost_function, optimizer=gradient_descent, l2_lambda=0):
        self.layers = layers
        self.w_grads = {}
        self.b_grads = {}
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

        self.layers[0].init(input_dim)
        for prev_layer, curr_layer in zip(self.layers, self.layers[1:]):
            curr_layer.init(prev_layer.get_output_dim())

        self.trainable_layers = set(layer for layer in self.layers if layer.get_params() is not None)
        self.optimizer = optimizer(self.trainable_layers)
        self.optimizer.initialize()

    def forward_prop(self, x, training=True):

        a = x
        for layer in self.layers:
            a = layer.forward(a, training)

        return a

    def backward_prop(self, last, y):

        da = self.cost_function.grad(last, y)
        batch_size = da.shape[0]

        for layer in reversed(self.layers):
            da_prev, dw, db = layer.backward(da)
            if layer in self.trainable_layers:
                if self.l2_lambda != 0:
                    self.w_grads[layer] = dw + (self.l2_lambda / batch_size) * layer.get_params()[0]
                else:
                    self.w_grads[layer] = dw

                self.b_grads[layer] = db

            da = da_prev

    def predict(self, x):
        return self.forward_prop(x, training=False)

    def update_params(self, learning_rate, step):
        self.optimizer.update(learning_rate, self.w_grads, self.b_grads, step)

    def compute_cost(self, last, y):
        cost = self.cost_function.f(last, y)
        if self.l2_lambda != 0:
            batch_size = y.shape[0]
            weights = [layer.get_params()[0] for layer in self.trainable_layers]
            l2_cost = (self.l2_lambda / (2 * batch_size)) * NeuralNetwork.cumul_func(lambda ws, w: ws + np.sum(w ** 2),
                                                                                     weights, 0)
            return cost + l2_cost
        else:
            return cost

    def train(self, x_train, y_train, batch_size, learning_rate, num_epochs, validation_data):
        x_val, y_val = validation_data

        print(f"Started training with {num_epochs} epochs and batch size {batch_size}")
        step = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            epoch_cost = 0

            if batch_size == x_train.shape[0]:
                batches = (x_train, y_train)
            else:
                batches = NeuralNetwork.create_batches(x_train, y_train, batch_size)

            n_batch = len(batches)
            for i, batch in enumerate(batches, 1):
                batch_x, batch_y = batch
                step += 1
                epoch_cost = self.train_step(batch_x, batch_y, learning_rate, step) / batch_size
                print("\rProgress: {:.2f}%".format(100 * i / n_batch), end="")
            print(f"\rEpoch {epoch + 1} cost: {epoch_cost}")

            accuracy = np.sum(np.argmax(self.predict(x_val), axis=1) == y_val, axis=1) / x_val.shape[0]
            print(f"Validation accuracy: {accuracy}")
        print("Training finished")

    def train_step(self, x_train, y_train, learning_rate, step):
        last = self.forward_prop(x_train, training=True)
        self.backward_prop(last, y_train)
        cost = self.compute_cost(last, y_train)
        self.update_params(learning_rate, step)
        return cost

    @staticmethod
    def cumul_func(func, iterable, init=None):
        it = iter(iterable)
        if init is not None:
            value = init
        else:
            value = next(it)
        for element in it:
            value = func(value, element)
        return value

    @staticmethod
    def create_batches(x, y, mini_batch_size):
        """
        Creates sample mini batches from input and target labels batches.
        x : numpy.ndarray
            Input batch.
        y : numpy.ndarray
            Target labels batch.
        Returns
        -------
        list
            Mini batches pairs of input and target labels.
        """
        batch_size = x.shape[0]
        batches = []

        p = np.random.permutation(x.shape[0])
        x, y = x[p, :], y[p, :]
        n_batches = batch_size // mini_batch_size

        for k in range(0, n_batches):
            batches.append((
                x[k * mini_batch_size:(k + 1) * mini_batch_size, :],
                y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            ))

        # Fill with remaining data, if needed
        if batch_size % mini_batch_size != 0:
            batches.append((
                x[n_batches * mini_batch_size:, :],
                y[n_batches * mini_batch_size:, :]
            ))
        return batches
