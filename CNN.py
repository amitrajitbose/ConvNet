import autograd.numpy as np

from mla.neuralnet.layers import Layer, ParamMixin
from mla.neuralnet.parameters import Parameters


class Convolution(Layer, ParamMixin):
    def __init__(self, n_filters=8, filter_shape=(3, 3), padding=(0, 0), stride=(1, 1), parameters=None):
        """A 2D convolutional layer.
        Input shape: (n_images, n_channels, height, width)

        Parameters
        ----------
        n_filters : int, default 8
            The number of filters (kernels).
        filter_shape : tuple(int, int), default (3, 3)
            The shape of the filters. (height, width)
        parameters : Parameters instance, default None
        stride : tuple(int, int), default (1, 1)
            The step of the convolution. (height, width).
        padding : tuple(int, int), default (0, 0)
            The number of pixel to add to each side of the input. (height, weight)

        """
        self.padding = padding
        self._params = parameters
        self.stride = stride
        self.filter_shape = filter_shape
        self.n_filters = n_filters
        if self._params is None:
            self._params = Parameters()

    def setup(self, X_shape):
        n_channels, self.height, self.width = X_shape[1:]

        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = (self.n_filters)
        self._params.setup_weights(W_shape, b_shape)

    def forward_pass(self, X):
        n_images, n_channels, height, width = self.shape(X.shape)
        self.last_input = X
        self.col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.col_W = self._params['W'].reshape(self.n_filters, -1).T

        out = np.dot(self.col, self.col_W) + self._params['b']
        out = out.reshape(n_images, height, width, -1).transpose(0, 3, 1, 2)
        return out

    def backward_pass(self, delta):
        delta = delta.transpose(0, 2, 3, 1).reshape(-1, self.n_filters)

        d_W = np.dot(self.col.T, delta).transpose(1, 0).reshape(self._params['W'].shape)
        d_b = np.sum(delta, axis=0)
        self._params.update_grad('b', d_b)
        self._params.update_grad('W', d_W)

        d_c = np.dot(delta, self.col_W.T)
        return column_to_image(d_c, self.last_input.shape, self.filter_shape, self.stride, self.padding)

    def shape(self, x_shape):
        height, width = convoltuion_shape(self.height, self.width, self.filter_shape, self.stride, self.padding)
        return x_shape[0], self.n_filters, height, width
