from typing import Callable, Optional, Tuple, Union
import jax.numpy as jnp
from jax import random

class InputLayer:
    """
    An input layer for a neural network.

    Args:
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        dtype (str): The data type of the input tensor. Defaults to 'float32'.
        name (Optional[str]): The name of the input layer. Defaults to None.
    """
    def __init__(self, input_shape: Tuple[int, ...], dtype: str = 'float32', name: Optional[str] = None):
        self.input_shape = input_shape
        self.dtype = dtype
        self.name = name

    def __call__(self) -> jnp.ndarray:
        """
        Create an input tensor with the specified shape and data type.

        Returns:
            jnp.ndarray: Input tensor.
        """
        return jnp.zeros(self.input_shape, dtype=self.dtype)

class Dense:
    """
    A fully connected neural network layer.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        weight_init (Optional[Callable]): A function to initialize the weights. Defaults to jax.random.normal.
        bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
    """
    def __init__(self, input_dim: int, output_dim: int, weight_init: Optional[Callable] = None, bias_init: Optional[Callable] = None):
        key = random.PRNGKey(0)
        if weight_init is None:
            weight_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.weights = weight_init(key, (input_dim, output_dim))
        self.bias = bias_init(key, (output_dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Dense layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        return jnp.sum(x[:, jnp.newaxis, :] * self.weights[jnp.newaxis, :, :], axis=-1) + self.bias
    
class Conv1D:
    """
    A 1D convolutional neural network layer.

    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution. Defaults to 1.
        padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
        kernel_init (Optional[Callable]): A function to initialize the kernel. Defaults to jax.random.normal.
        bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding: str = 'VALID', kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None):
        key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(key, (kernel_size, input_channels, output_channels))
        self.bias = bias_init(key, (output_channels,))
        self.stride = stride
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        if self.padding == 'SAME':
            pad_size = (self.kernel.shape[0] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_size = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_shape = (x.shape[0], (x.shape[1] - self.kernel.shape[0]) // self.stride + 1, self.kernel.shape[2])
        output = jnp.zeros(output_shape)

        for i in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride):
            output = output.at[:, i // self.stride, :].set(jnp.sum(x[:, i:i + self.kernel.shape[0], :, jnp.newaxis] * self.kernel, axis=(1, 2)) + self.bias)

        return output

class Conv2D:
    """
    A 2D convolutional neural network layer.

    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): The size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]]): The stride of the convolution. Defaults to (1, 1).
        padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
        kernel_init (Optional[Callable]): A function to initialize the kernel. Defaults to jax.random.normal.
        bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = (1, 1), padding: str = 'VALID', kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None):
        key = random.PRNGKey(0)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(key, (kernel_size[0], kernel_size[1], input_channels, output_channels))
        self.bias = bias_init(key, (output_channels,))
        self.stride = stride
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Conv2D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        if self.padding == 'SAME':
            pad_height = (self.kernel.shape[0] - 1) // 2
            pad_width = (self.kernel.shape[1] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_height = pad_width = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_shape = (x.shape[0], (x.shape[1] - self.kernel.shape[0]) // self.stride[0] + 1, (x.shape[2] - self.kernel.shape[1]) // self.stride[1] + 1, self.kernel.shape[3])
        output = jnp.zeros(output_shape)

        for i in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride[0]):
            for j in range(0, x.shape[2] - self.kernel.shape[1] + 1, self.stride[1]):
                output = output.at[:, i // self.stride[0], j // self.stride[1], :].set(jnp.sum(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[1], :, jnp.newaxis] * self.kernel, axis=(1, 2, 3)) + self.bias)

        return output

class Conv3D:
    """
    A 3D convolutional neural network layer.

    Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        kernel_size (Union[int, Tuple[int, int, int]]): The size of the convolutional kernel.
        stride (Union[int, Tuple[int, int, int]]): The stride of the convolution. Defaults to (1, 1, 1).
        padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
        kernel_init (Optional[Callable]): A function to initialize the kernel. Defaults to jax.random.normal.
        bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]] = (1, 1, 1), padding: str = 'VALID', kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None):
        key = random.PRNGKey(0)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(key, (kernel_size[0], kernel_size[1], kernel_size[2], input_channels, output_channels))
        self.bias = bias_init(key, (output_channels,))
        self.stride = stride
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        if self.padding == 'SAME':
            pad_depth = (self.kernel.shape[0] - 1) // 2
            pad_height = (self.kernel.shape[1] - 1) // 2
            pad_width = (self.kernel.shape[2] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_depth, pad_depth), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_depth = pad_height = pad_width = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_shape = (x.shape[0], (x.shape[1] - self.kernel.shape[0]) // self.stride[0] + 1, (x.shape[2] - self.kernel.shape[1]) // self.stride[1] + 1, (x.shape[3] - self.kernel.shape[2]) // self.stride[2] + 1, self.kernel.shape[4])
        output = jnp.zeros(output_shape)

        for i in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride[0]):
            for j in range(0, x.shape[2] - self.kernel.shape[1] + 1, self.stride[1]):
                for k in range(0, x.shape[3] - self.kernel.shape[2] + 1, self.stride[2]):
                    output = output.at[:, i // self.stride[0], j // self.stride[1], k // self.stride[2], :].set(jnp.sum(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[1], k:k + self.kernel.shape[2], :, jnp.newaxis] * self.kernel, axis=(1, 2, 3, 4)) + self.bias)

        return output
    
class MaxPooling1D:
    """
    A 1D max pooling layer.

    Args:
        pool_size (int): The size of the pooling window. Defaults to 2.
        stride (Optional[int]): The stride of the pooling operation. Defaults to pool_size.
        padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
    """
    def __init__(self, pool_size: int = 2, stride: Optional[int] = None, padding: str = 'VALID'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the MaxPooling1D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        if self.padding == 'SAME':
            pad_size = (self.pool_size - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_size = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_length = (x.shape[1] - self.pool_size) // self.stride + 1
        output = jnp.zeros((x.shape[0], output_length, x.shape[2]))

        for i in range(0, x.shape[1] - self.pool_size + 1, self.stride):
            output = output.at[:, i // self.stride, :].set(jnp.max(x[:, i:i + self.pool_size, :], axis=1))

        return output

class MaxPooling2D:
    """
    A 2D max pooling layer.

    Args:
        pool_size (Tuple[int, int]): The size of the pooling window. Defaults to (2, 2).
        stride (Optional[Tuple[int, int]]): The stride of the pooling operation. Defaults to pool_size.
        padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
    """
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), stride: Optional[Tuple[int, int]] = None, padding: str = 'VALID'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the MaxPooling2D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        if self.padding == 'SAME':
            pad_height = (self.pool_size[0] - 1) // 2
            pad_width = (self.pool_size[1] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_height = pad_width = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_height = (x.shape[1] - self.pool_size[0]) // self.stride[0] + 1
        output_width = (x.shape[2] - self.pool_size[1]) // self.stride[1] + 1
        output = jnp.zeros((x.shape[0], output_height, output_width, x.shape[3]))

        for i in range(0, x.shape[1] - self.pool_size[0] + 1, self.stride[0]):
            for j in range(0, x.shape[2] - self.pool_size[1] + 1, self.stride[1]):
                output = output.at[:, i // self.stride[0], j // self.stride[1], :].set(jnp.max(x[:, i:i + self.pool_size[0], j:j + self.pool_size[1], :], axis=(1, 2)))

        return output

class MaxPooling3D:
    """
    A 3D max pooling layer.

    Args:
        pool_size (Tuple[int, int, int]): The size of the pooling window. Defaults to (2, 2, 2).
        stride (Optional[Tuple[int, int, int]]): The stride of the pooling operation. Defaults to pool_size.
        padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
    """
    def __init__(self, pool_size: Tuple[int, int, int] = (2, 2, 2), stride: Optional[Tuple[int, int, int]] = None, padding: str = 'VALID'):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the MaxPooling3D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor.
        """
        if self.padding == 'SAME':
            pad_depth = (self.pool_size[0] - 1) // 2
            pad_height = (self.pool_size[1] - 1) // 2
            pad_width = (self.pool_size[2] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_depth, pad_depth), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_depth = pad_height = pad_width = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_depth = (x.shape[1] - self.pool_size[0]) // self.stride[0] + 1
        output_height = (x.shape[2] - self.pool_size[1]) // self.stride[1] + 1
        output_width = (x.shape[3] - self.pool_size[2]) // self.stride[2] + 1
        output = jnp.zeros((x.shape[0], output_depth, output_height, output_width, x.shape[4]))

        for d in range(0, x.shape[1] - self.pool_size[0] + 1, self.stride[0]):
            for i in range(0, x.shape[2] - self.pool_size[1] + 1, self.stride[1]):
                for j in range(0, x.shape[3] - self.pool_size[2] + 1, self.stride[2]):
                    output = output.at[:, d // self.stride[0], i // self.stride[1], j // self.stride[2], :].set(
                        jnp.max(x[:, d:d + self.pool_size[0], i:i + self.pool_size[1], j:j + self.pool_size[2], :], axis=(1, 2, 3))
                    )

        return output

class Dropout:
    """
    A dropout layer for regularization.

    Args:
        rate (float): The dropout rate, between 0 and 1. Fraction of the input units to drop.
        seed (Optional[int]): A seed to ensure reproducibility.
    """
    def __init__(self, rate: float = 0.5, seed: Optional[int] = None):
        self.rate = rate
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed) if seed is not None else None

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply dropout to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor.
            training (bool): Whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
            jnp.ndarray: Output tensor with dropout applied.
        """
        if not training or self.rate == 0.0:
            return x

        if self.rng is None:
            self.rng = jax.random.PRNGKey(jax.random.randint(jax.random.PRNGKey(0), (), 0, 1e6))

        keep_prob = 1.0 - self.rate
        self.rng, dropout_rng = jax.random.split(self.rng)
        mask = jax.random.bernoulli(dropout_rng, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, 0)

    def set_rate(self, rate: float):
        """
        Set a new dropout rate.

        Args:
            rate (float): The new dropout rate, between 0 and 1.
        """
        self.rate = rate

    def set_seed(self, seed: int):
        """
        Set a new seed for reproducibility.

        Args:
            seed (int): The new seed.
        """
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

class Flatten:
    """
    A layer that flattens the input tensor.

    Args:
        start_dim (int): The first dimension to flatten. Defaults to 1.
        end_dim (int): The last dimension to flatten. Defaults to -1.
    """
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the Flatten layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Flattened output tensor.
        """
        return jnp.reshape(x, (x.shape[0], -1))

    def set_start_dim(self, start_dim: int):
        """
        Set a new start dimension for flattening.

        Args:
            start_dim (int): The new start dimension.
        """
        self.start_dim = start_dim

    def set_end_dim(self, end_dim: int):
        """
        Set a new end dimension for flattening.

        Args:
            end_dim (int): The new end dimension.
        """
        self.end_dim = end_dim

class BatchNormalization:
    """
    A batch normalization layer.

    Args:
        num_features (int): The number of features in the input.
        eps (float): A small value to avoid division by zero. Defaults to 1e-5.
        momentum (float): The momentum for the moving average. Defaults to 0.9.
        affine (bool): If True, this module has learnable affine parameters. Defaults to True.
        gamma_init (Optional[Callable]): A function to initialize the gamma parameter. Defaults to ones.
        beta_init (Optional[Callable]): A function to initialize the beta parameter. Defaults to zeros.
        moving_mean_init (Optional[Callable]): A function to initialize the moving mean. Defaults to zeros.
        moving_var_init (Optional[Callable]): A function to initialize the moving variance. Defaults to ones.
        beta_regularizer (Optional[Callable]): Regularizer function for the beta parameter. Defaults to None.
        gamma_regularizer (Optional[Callable]): Regularizer function for the gamma parameter. Defaults to None.
        beta_constraint (Optional[Callable]): Constraint function for the beta parameter. Defaults to None.
        gamma_constraint (Optional[Callable]): Constraint function for the gamma parameter. Defaults to None.
        synchronized (bool): If True, synchronize the batch statistics across multiple devices. Defaults to False.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9, affine: bool = True, gamma_init: Optional[Callable] = None, beta_init: Optional[Callable] = None, moving_mean_init: Optional[Callable] = None, moving_var_init: Optional[Callable] = None, beta_regularizer: Optional[Callable] = None, gamma_regularizer: Optional[Callable] = None, beta_constraint: Optional[Callable] = None, gamma_constraint: Optional[Callable] = None, synchronized: bool = False, **kwargs):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.gamma_init = gamma_init if gamma_init is not None else lambda key, shape: jnp.ones(shape)
        self.beta_init = beta_init if beta_init is not None else lambda key, shape: jnp.zeros(shape)
        self.moving_mean_init = moving_mean_init if moving_mean_init is not None else lambda key, shape: jnp.zeros(shape)
        self.moving_var_init = moving_var_init if moving_var_init is not None else lambda key, shape: jnp.ones(shape)
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.synchronized = synchronized

        key = random.PRNGKey(0)
        if self.affine:
            self.gamma = self.gamma_init(key, (num_features,))
            self.beta = self.beta_init(key, (num_features,))
        else:
            self.gamma = None
            self.beta = None
        self.running_mean = self.moving_mean_init(key, (num_features,))
        self.running_var = self.moving_var_init(key, (num_features,))

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the BatchNormalization layer.

        Args:
            x (jnp.ndarray): Input tensor.
            training (bool): Whether the layer is in training mode. Defaults to True.

        Returns:
            jnp.ndarray: Normalized output tensor.
        """
        if training:
            batch_mean = jnp.mean(x, axis=0)
            batch_var = jnp.var(x, axis=0)
            x_normalized = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            x_normalized = (x - self.running_mean) / jnp.sqrt(self.running_var + self.eps)
        
        if self.affine:
            return self.gamma * x_normalized + self.beta
        else:
            return x_normalized

    def set_gamma(self, gamma: jnp.ndarray):
        """
        Set a new gamma parameter.

        Args:
            gamma (jnp.ndarray): The new gamma parameter.
        """
        self.gamma = gamma

    def set_beta(self, beta: jnp.ndarray):
        """
        Set a new beta parameter.

        Args:
            beta (jnp.ndarray): The new beta parameter.
        """
        self.beta = beta

    def set_moving_mean(self, moving_mean: jnp.ndarray):
        """
        Set a new moving mean.

        Args:
            moving_mean (jnp.ndarray): The new moving mean.
        """
        self.running_mean = moving_mean

    def set_moving_var(self, moving_var: jnp.ndarray):
        """
        Set a new moving variance.

        Args:
            moving_var (jnp.ndarray): The new moving variance.
        """
        self.running_var = moving_var

    def set_eps(self, eps: float):
        """
        Set a new epsilon value.

        Args:
            eps (float): The new epsilon value.
        """
        self.eps = eps

    def set_momentum(self, momentum: float):
        """
        Set a new momentum value.

        Args:
            momentum (float): The new momentum value.
        """
        self.momentum = momentum

    def set_affine(self, affine: bool):
        """
        Set a new affine value.

        Args:
            affine (bool): The new affine value.
        """
        self.affine = affine
        key = random.PRNGKey(0)
        if self.affine:
            self.gamma = self.gamma_init(key, (self.num_features,))
            self.beta = self.beta_init(key, (self.num_features,))
        else:
            self.gamma = None
            self.beta = None
