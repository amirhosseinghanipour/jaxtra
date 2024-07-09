from typing import Callable, Optional, Tuple, Union
import jax.numpy as jnp
from jax import random

class Input:
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the InputLayer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Transformed input tensor.
        """
        return x

class Dense:
    """
    A fully connected (dense) neural network layer.

    This layer implements a dense (fully connected) neural network layer, which performs a linear transformation
    followed by an optional activation function. It also supports L1 and L2 regularization, activity regularization,
    and dropout for regularization during training.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        key (random.PRNGKey): A JAX random key for initializing weights.
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        weight_init (Optional[Callable]): Function to initialize weights. Defaults to jax.random.normal.
        bias_init (Optional[Callable]): Function to initialize biases. Defaults to jnp.zeros.
        use_bias (bool): Whether to include a bias term. Defaults to True.
        dtype (jnp.dtype): Data type of the weights and biases. Defaults to jnp.float32.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_dim: int, output_dim: int, key: random.PRNGKey, 
                    activation: Optional[Callable] = None, 
                    weight_init: Optional[Callable] = None, 
                    bias_init: Optional[Callable] = None, 
                    use_bias: bool = True, 
                    dtype: jnp.dtype = jnp.float32, 
                    l1_reg: float = 0.0,
                    l2_reg: float = 0.0,
                    activity_reg: float = 0.0,
                    dropout_rate: float = 0.0,
                    training: bool = True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.dtype = dtype
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        # Initialize weights
        if weight_init is None:
            weight_init = random.normal
        self.weights = weight_init(key, (input_dim, output_dim)).astype(dtype)

        # Initialize bias
        if use_bias:
            if bias_init is None:
                bias_init = jnp.zeros
            self.bias = bias_init((output_dim,)).astype(dtype)
        else:
            self.bias = None

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Dense layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): JAX random key for dropout. Required if dropout_rate > 0.0 and training is True.

        Returns:
            jnp.ndarray: Output tensor after applying the Dense layer transformations.
        """
        x = jnp.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, x.shape)
            x = jnp.where(mask, x / keep_prob, 0)
        return x

    def get_params(self) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Get the parameters of the Dense layer.

        Returns:
            Tuple[jnp.ndarray, Optional[jnp.ndarray]]: A tuple containing the weights and biases of the layer.
        """
        return self.weights, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, Optional[jnp.ndarray]]):
        """
        Set the parameters of the Dense layer.

        Args:
            params (Tuple[jnp.ndarray, Optional[jnp.ndarray]]): A tuple containing the weights and biases to set.
        """
        self.weights, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 regularization loss.

        Returns:
            jnp.ndarray: The L1 regularization loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.weights))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 regularization loss.

        Returns:
            jnp.ndarray: The L2 regularization loss.
        """
        return self.l2_reg * jnp.sum(self.weights ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity regularization loss.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The activity regularization loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total regularization loss.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The total regularization loss, including L1, L2, and activity regularization losses.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Dense layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Dense layer.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'dtype': self.dtype,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'weights': self.weights,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Dense layer.

        Args:
            data (dict): A dictionary containing the serialized Dense layer.

        Returns:
            Dense: The deserialized Dense layer.
        """
        layer = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            activation=data['activation'],
            use_bias=data['use_bias'],
            dtype=data['dtype'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
            activity_reg=data['activity_reg'],
            dropout_rate=data['dropout_rate']
        )
        layer.weights = data['weights']
        layer.bias = data['bias']
        return layer
    
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
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding: str = 'VALID', 
                    kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None, 
                    activation: Optional[Callable] = None, l1_reg: float = 0.0, l2_reg: float = 0.0, 
                    activity_reg: float = 0.0, dropout_rate: float = 0.0, training: bool = True):
        """
        Initialize the Conv1D layer.

        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (int): The stride of the convolution. Defaults to 1.
            padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
            kernel_init (Optional[Callable]): A function to initialize the kernel. Defaults to jax.random.normal.
            bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
            activation (Optional[Callable]): Activation function to apply. Defaults to None.
            l1_reg (float): L1 regularization factor. Defaults to 0.0.
            l2_reg (float): L2 regularization factor. Defaults to 0.0.
            activity_reg (float): Activity regularization factor. Defaults to 0.0.
            dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
            training (bool): Whether the layer is in training mode. Defaults to True.
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        self.key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(self.key, (kernel_size, input_channels, output_channels))
        self.bias = bias_init(self.key, (output_channels,))

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): Random key for dropout. Defaults to None.

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

        if self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0)
        return output

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the parameters of the Conv1D layer.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The kernel and bias of the layer.
        """
        return self.kernel, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Set the parameters of the Conv1D layer.

        Args:
            params (Tuple[jnp.ndarray, jnp.ndarray]): The kernel and bias to set.
        """
        self.kernel, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 loss for the Conv1D layer.

        Returns:
            jnp.ndarray: The L1 loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.kernel))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 loss for the Conv1D layer.

        Returns:
            jnp.ndarray: The L2 loss.
        """
        return self.l2_reg * jnp.sum(self.kernel ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity loss for the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The activity loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total loss for the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The total loss.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Conv1D layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Conv1D layer.
        """
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'kernel': self.kernel,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Conv1D layer.

        Args:
            data (dict): A dictionary containing the serialized Conv1D layer.

        Returns:
            Conv1D: The deserialized Conv1D layer.
        """
        layer = cls(
            input_channels=data['input_channels'],
            output_channels=data['output_channels'],
            kernel_size=data['kernel_size'],
            stride=data['stride'],
            padding=data['padding'],
            activation=data['activation'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
            activity_reg=data['activity_reg'],
            dropout_rate=data['dropout_rate']
        )
        layer.kernel = data['kernel']
        layer.bias = data['bias']
        return layer

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
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                stride: Union[int, Tuple[int, int]] = (1, 1), padding: str = 'VALID', 
                kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None, 
                activation: Optional[Callable] = None, l1_reg: float = 0.0, l2_reg: float = 0.0, 
                activity_reg: float = 0.0, dropout_rate: float = 0.0, training: bool = True):
        """
        Initialize the Conv2D layer.

        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): The size of the convolutional kernel.
            stride (Union[int, Tuple[int, int]]): The stride of the convolution. Defaults to (1, 1).
            padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
            kernel_init (Optional[Callable]): A function to initialize the kernel. Defaults to jax.random.normal.
            bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
            activation (Optional[Callable]): Activation function to apply. Defaults to None.
            l1_reg (float): L1 regularization factor. Defaults to 0.0.
            l2_reg (float): L2 regularization factor. Defaults to 0.0.
            activity_reg (float): Activity regularization factor. Defaults to 0.0.
            dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
            training (bool): Whether the layer is in training mode. Defaults to True.
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.activation = activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        self.key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(self.key, (self.kernel_size[0], self.kernel_size[1], input_channels, output_channels))
        self.bias = bias_init(self.key, (output_channels,))

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Conv2D layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): JAX random key for dropout. Required if dropout_rate > 0.0 and training is True.

        Returns:
            jnp.ndarray: Output tensor after applying the Conv2D layer transformations.
        """
        if self.padding == 'SAME':
            pad_height = (self.kernel.shape[0] - 1) // 2
            pad_width = (self.kernel.shape[1] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_height = pad_width = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_shape = (x.shape[0], (x.shape[1] - self.kernel.shape[0]) // self.stride[0] + 1, 
                        (x.shape[2] - self.kernel.shape[1]) // self.stride[1] + 1, self.kernel.shape[3])
        output = jnp.zeros(output_shape)

        for i in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride[0]):
            for j in range(0, x.shape[2] - self.kernel.shape[1] + 1, self.stride[1]):
                output = output.at[:, i // self.stride[0], j // self.stride[1], :].set(
                    jnp.sum(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[1], :, jnp.newaxis] * self.kernel, axis=(1, 2, 3)) + self.bias)

        if self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0)
        return output

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the parameters of the Conv2D layer.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the kernel and biases of the layer.
        """
        return self.kernel, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Set the parameters of the Conv2D layer.

        Args:
            params (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the kernel and biases to set.
        """
        self.kernel, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 regularization loss.

        Returns:
            jnp.ndarray: The L1 regularization loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.kernel))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 regularization loss.

        Returns:
            jnp.ndarray: The L2 regularization loss.
        """
        return self.l2_reg * jnp.sum(self.kernel ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity regularization loss.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The activity regularization loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total regularization loss.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The total regularization loss, including L1, L2, and activity regularization losses.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Conv2D layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Conv2D layer.
        """
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'kernel': self.kernel,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Conv2D layer.

        Args:
            data (dict): A dictionary containing the serialized Conv2D layer.

        Returns:
            Conv2D: The deserialized Conv2D layer.
        """
        layer = cls(
            input_channels=data['input_channels'],
            output_channels=data['output_channels'],
            kernel_size=data['kernel_size'],
            stride=data['stride'],
            padding=data['padding'],
            activation=data['activation'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
            activity_reg=data['activity_reg'],
            dropout_rate=data['dropout_rate']
        )
        layer.kernel = data['kernel']
        layer.bias = data['bias']
        return layer

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
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]] = (1, 1, 1), padding: str = 'VALID', 
                    kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None, 
                    activation: Optional[Callable] = None, l1_reg: float = 0.0, l2_reg: float = 0.0, 
                    activity_reg: float = 0.0, dropout_rate: float = 0.0, training: bool = True):
        """
        Initialize the Conv3D layer.

        Args:
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            kernel_size (Union[int, Tuple[int, int, int]]): The size of the convolutional kernel.
            stride (Union[int, Tuple[int, int, int]]): The stride of the convolution. Defaults to (1, 1, 1).
            padding (str): The padding method, either 'SAME' or 'VALID'. Defaults to 'VALID'.
            kernel_init (Optional[Callable]): A function to initialize the kernel. Defaults to jax.random.normal.
            bias_init (Optional[Callable]): A function to initialize the biases. Defaults to jax.random.normal.
            activation (Optional[Callable]): Activation function to apply. Defaults to None.
            l1_reg (float): L1 regularization factor. Defaults to 0.0.
            l2_reg (float): L2 regularization factor. Defaults to 0.0.
            activity_reg (float): Activity regularization factor. Defaults to 0.0.
            dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
            training (bool): Whether the layer is in training mode. Defaults to True.
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding
        self.activation = activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        self.key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(self.key, (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], input_channels, output_channels))
        self.bias = bias_init(self.key, (output_channels,))

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): Random key for dropout. Defaults to None.

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

        for d in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride[0]):
            for i in range(0, x.shape[2] - self.kernel.shape[1] + 1, self.stride[1]):
                for j in range(0, x.shape[3] - self.kernel.shape[2] + 1, self.stride[2]):
                    output = output.at[:, d // self.stride[0], i // self.stride[1], j // self.stride[2], :].set(
                        jnp.sum(x[:, d:d + self.kernel.shape[0], i:i + self.kernel.shape[1], j:j + self.kernel.shape[2], :, jnp.newaxis] * self.kernel, axis=(1, 2, 3, 4)) + self.bias)

        if self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0)
        return output

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the parameters of the Conv3D layer.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The kernel and bias of the layer.
        """
        return self.kernel, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Set the parameters of the Conv3D layer.

        Args:
            params (Tuple[jnp.ndarray, jnp.ndarray]): The kernel and bias to set.
        """
        self.kernel, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 loss for the Conv3D layer.

        Returns:
            jnp.ndarray: The L1 loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.kernel))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 loss for the Conv3D layer.

        Returns:
            jnp.ndarray: The L2 loss.
        """
        return self.l2_reg * jnp.sum(self.kernel ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity loss for the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The activity loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total loss for the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The total loss.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Conv3D layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Conv3D layer.
        """
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'kernel': self.kernel,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Conv3D layer.

        Args:
            data (dict): A dictionary containing the serialized Conv3D layer.

        Returns:
            Conv3D: The deserialized Conv3D layer.
        """
        layer = cls(
            input_channels=data['input_channels'],
            output_channels=data['output_channels'],
            kernel_size=data['kernel_size'],
            stride=data['stride'],
            padding=data['padding'],
            activation=data['activation'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
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
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding: str = 'VALID', 
                    kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None, 
                    activation: Optional[Callable] = None, l1_reg: float = 0.0, l2_reg: float = 0.0, 
                    activity_reg: float = 0.0, dropout_rate: float = 0.0, training: bool = True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        self.key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(self.key, (kernel_size, input_channels, output_channels))
        self.bias = bias_init(self.key, (output_channels,))

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): Random key for dropout. Defaults to None.

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

        if self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0)
        return output

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the parameters of the Conv1D layer.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The kernel and bias of the layer.
        """
        return self.kernel, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Set the parameters of the Conv1D layer.

        Args:
            params (Tuple[jnp.ndarray, jnp.ndarray]): The kernel and bias to set.
        """
        self.kernel, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 loss for the Conv1D layer.

        Returns:
            jnp.ndarray: The L1 loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.kernel))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 loss for the Conv1D layer.

        Returns:
            jnp.ndarray: The L2 loss.
        """
        return self.l2_reg * jnp.sum(self.kernel ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity loss for the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The activity loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total loss for the Conv1D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The total loss.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Conv1D layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Conv1D layer.
        """
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'kernel': self.kernel,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Conv1D layer.

        Args:
            data (dict): A dictionary containing the serialized Conv1D layer.

        Returns:
            Conv1D: The deserialized Conv1D layer.
        """
        layer = cls(
            input_channels=data['input_channels'],
            output_channels=data['output_channels'],
            kernel_size=data['kernel_size'],
            stride=data['stride'],
            padding=data['padding'],
            activation=data['activation'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
            activity_reg=data['activity_reg'],
            dropout_rate=data['dropout_rate']
        )
        layer.kernel = data['kernel']
        layer.bias = data['bias']
        return layer

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
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                stride: Union[int, Tuple[int, int]] = (1, 1), padding: str = 'VALID', 
                kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None, 
                activation: Optional[Callable] = None, l1_reg: float = 0.0, l2_reg: float = 0.0, 
                activity_reg: float = 0.0, dropout_rate: float = 0.0, training: bool = True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.activation = activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        self.key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(self.key, (self.kernel_size[0], self.kernel_size[1], input_channels, output_channels))
        self.bias = bias_init(self.key, (output_channels,))

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Conv2D layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): JAX random key for dropout. Required if dropout_rate > 0.0 and training is True.

        Returns:
            jnp.ndarray: Output tensor after applying the Conv2D layer transformations.
        """
        if self.padding == 'SAME':
            pad_height = (self.kernel.shape[0] - 1) // 2
            pad_width = (self.kernel.shape[1] - 1) // 2
            x = jnp.pad(x, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        elif self.padding == 'VALID':
            pad_height = pad_width = 0
        else:
            raise ValueError("Padding must be 'SAME' or 'VALID'")

        output_shape = (x.shape[0], (x.shape[1] - self.kernel.shape[0]) // self.stride[0] + 1, 
                        (x.shape[2] - self.kernel.shape[1]) // self.stride[1] + 1, self.kernel.shape[3])
        output = jnp.zeros(output_shape)

        for i in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride[0]):
            for j in range(0, x.shape[2] - self.kernel.shape[1] + 1, self.stride[1]):
                output = output.at[:, i // self.stride[0], j // self.stride[1], :].set(
                    jnp.sum(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[1], :, jnp.newaxis] * self.kernel, axis=(1, 2, 3)) + self.bias)

        if self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0)
        return output

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the parameters of the Conv2D layer.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the kernel and biases of the layer.
        """
        return self.kernel, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Set the parameters of the Conv2D layer.

        Args:
            params (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing the kernel and biases to set.
        """
        self.kernel, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 regularization loss.

        Returns:
            jnp.ndarray: The L1 regularization loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.kernel))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 regularization loss.

        Returns:
            jnp.ndarray: The L2 regularization loss.
        """
        return self.l2_reg * jnp.sum(self.kernel ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity regularization loss.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The activity regularization loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total regularization loss.

        Args:
            x (jnp.ndarray): The input tensor.

        Returns:
            jnp.ndarray: The total regularization loss, including L1, L2, and activity regularization losses.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Conv2D layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Conv2D layer.
        """
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'kernel': self.kernel,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Conv2D layer.

        Args:
            data (dict): A dictionary containing the serialized Conv2D layer.

        Returns:
            Conv2D: The deserialized Conv2D layer.
        """
        layer = cls(
            input_channels=data['input_channels'],
            output_channels=data['output_channels'],
            kernel_size=data['kernel_size'],
            stride=data['stride'],
            padding=data['padding'],
            activation=data['activation'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
            activity_reg=data['activity_reg'],
            dropout_rate=data['dropout_rate']
        )
        layer.kernel = data['kernel']
        layer.bias = data['bias']
        return layer

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
        activation (Optional[Callable]): Activation function to apply. Defaults to None.
        l1_reg (float): L1 regularization factor. Defaults to 0.0.
        l2_reg (float): L2 regularization factor. Defaults to 0.0.
        activity_reg (float): Activity regularization factor. Defaults to 0.0.
        dropout_rate (float): Dropout rate for regularization during training. Defaults to 0.0.
        training (bool): Whether the layer is in training mode. Defaults to True.
    """
    def __init__(self, input_channels: int, output_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]] = (1, 1, 1), padding: str = 'VALID', 
                    kernel_init: Optional[Callable] = None, bias_init: Optional[Callable] = None, 
                    activation: Optional[Callable] = None, l1_reg: float = 0.0, l2_reg: float = 0.0, 
                    activity_reg: float = 0.0, dropout_rate: float = 0.0, training: bool = True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding
        self.activation = activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.activity_reg = activity_reg
        self.dropout_rate = dropout_rate
        self.training = training

        self.key = random.PRNGKey(0)
        if kernel_init is None:
            kernel_init = random.normal
        if bias_init is None:
            bias_init = random.normal

        self.kernel = kernel_init(self.key, (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], input_channels, output_channels))
        self.bias = bias_init(self.key, (output_channels,))

    def __call__(self, x: jnp.ndarray, key: Optional[random.PRNGKey] = None) -> jnp.ndarray:
        """
        Forward pass through the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.
            key (Optional[random.PRNGKey]): Random key for dropout. Defaults to None.

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

        for d in range(0, x.shape[1] - self.kernel.shape[0] + 1, self.stride[0]):
            for i in range(0, x.shape[2] - self.kernel.shape[1] + 1, self.stride[1]):
                for j in range(0, x.shape[3] - self.kernel.shape[2] + 1, self.stride[2]):
                    output = output.at[:, d // self.stride[0], i // self.stride[1], j // self.stride[2], :].set(
                        jnp.sum(x[:, d:d + self.kernel.shape[0], i:i + self.kernel.shape[1], j:j + self.kernel.shape[2], :, jnp.newaxis] * self.kernel, axis=(1, 2, 3, 4)) + self.bias)

        if self.activation is not None:
            output = self.activation(output)
        if self.dropout_rate > 0.0 and self.training:
            if key is None:
                raise ValueError("Random key must be provided for dropout during training.")
            keep_prob = 1.0 - self.dropout_rate
            mask = random.bernoulli(key, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0)
        return output

    def get_params(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the parameters of the Conv3D layer.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The kernel and bias of the layer.
        """
        return self.kernel, self.bias

    def set_params(self, params: Tuple[jnp.ndarray, jnp.ndarray]):
        """
        Set the parameters of the Conv3D layer.

        Args:
            params (Tuple[jnp.ndarray, jnp.ndarray]): The kernel and bias to set.
        """
        self.kernel, self.bias = params

    def l1_loss(self) -> jnp.ndarray:
        """
        Compute the L1 loss for the Conv3D layer.

        Returns:
            jnp.ndarray: The L1 loss.
        """
        return self.l1_reg * jnp.sum(jnp.abs(self.kernel))

    def l2_loss(self) -> jnp.ndarray:
        """
        Compute the L2 loss for the Conv3D layer.

        Returns:
            jnp.ndarray: The L2 loss.
        """
        return self.l2_reg * jnp.sum(self.kernel ** 2)

    def activity_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the activity loss for the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The activity loss.
        """
        return self.activity_reg * jnp.sum(x)

    def total_loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the total loss for the Conv3D layer.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: The total loss.
        """
        return self.l1_loss() + self.l2_loss() + self.activity_loss(x)

    def serialize(self) -> dict:
        """
        Serialize the Conv3D layer to a dictionary.

        Returns:
            dict: A dictionary containing the serialized Conv3D layer.
        """
        return {
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'activity_reg': self.activity_reg,
            'dropout_rate': self.dropout_rate,
            'kernel': self.kernel,
            'bias': self.bias
        }

    @classmethod
    def deserialize(cls, data: dict):
        """
        Deserialize a dictionary to a Conv3D layer.

        Args:
            data (dict): A dictionary containing the serialized Conv3D layer.

        Returns:
            Conv3D: The deserialized Conv3D layer.
        """
        layer = cls(
            input_channels=data['input_channels'],
            output_channels=data['output_channels'],
            kernel_size=data['kernel_size'],
            stride=data['stride'],
            padding=data['padding'],
            activation=data['activation'],
            l1_reg=data['l1_reg'],
            l2_reg=data['l2_reg'],
            activity_reg=data['activity_reg'],
            dropout_rate=data['dropout_rate']
        )
        layer.kernel = data['kernel']
        layer.bias = data['bias']
        return layer
    
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

    def serialize(self) -> dict:
        return {
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            pool_size=data['pool_size'],
            stride=data['stride'],
            padding=data['padding']
        )

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

    def serialize(self) -> dict:
        return {
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            pool_size=data['pool_size'],
            stride=data['stride'],
            padding=data['padding']
        )

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
                    output = output.at[:, d // self.stride[0], i // self.stride[1], j // self.stride[2], :].set(jnp.max(x[:, d:d + self.pool_size[0], i:i + self.pool_size[1], j:j + self.pool_size[2], :], axis=(1, 2, 3)))

        return output

    def serialize(self) -> dict:
        return {
            'pool_size': self.pool_size,
            'stride': self.stride,
            'padding': self.padding
        }

    @classmethod
    def deserialize(cls, data: dict):
        return cls(
            pool_size=data['pool_size'],
            stride=data['stride'],
            padding=data['padding']
        )

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
