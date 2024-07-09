import jax.numpy as jnp
from typing import Callable

def relu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Output array with ReLU applied element-wise.
    """
    return jnp.maximum(0, x)

def leaky_relu(x: jnp.ndarray, alpha: float = 0.01) -> jnp.ndarray:
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Args:
        x (jnp.ndarray): Input array.
        alpha (float): Slope of the activation function for x < 0. Default is 0.01.

    Returns:
        jnp.ndarray: Output array with Leaky ReLU applied element-wise.
    """
    return jnp.where(x > 0, x, alpha * x)

def elu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """
    Exponential Linear Unit (ELU) activation function.

    Args:
        x (jnp.ndarray): Input array.
        alpha (float): Scaling factor for the negative factor. Default is 1.0.

    Returns:
        jnp.ndarray: Output array with ELU applied element-wise.
    """
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))

def selu(x: jnp.ndarray) -> jnp.ndarray:
    """
    Scaled Exponential Linear Unit (SELU) activation function.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Output array with SELU applied element-wise.
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))

def swish(x: jnp.ndarray) -> jnp.ndarray:
    """
    Swish activation function.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Output array with Swish applied element-wise.
    """
    return x * sigmoid(x)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """
    Sigmoid activation function.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Output array with Sigmoid applied element-wise.
    """
    return 1 / (1 + jnp.exp(-x))

def tanh(x: jnp.ndarray) -> jnp.ndarray:
    """
    Hyperbolic Tangent (Tanh) activation function.

    Args:
        x (jnp.ndarray): Input array.

    Returns:
        jnp.ndarray: Output array with Tanh applied element-wise.
    """
    return jnp.tanh(x)

def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Softmax activation function.

    Args:
        x (jnp.ndarray): Input array.
        axis (int): Axis along which the softmax is computed. Default is -1.

    Returns:
        jnp.ndarray: Output array with Softmax applied along the specified axis.
    """
    e_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e_x / jnp.sum(e_x, axis=axis, keepdims=True)