import jax
import jax.numpy as jnp
from typing import Tuple, Union, Callable, Optional, Iterator, Any
import numpy as np
import tensorflow as tf
import torch

class DataLoader:
    def __init__(
        self,
        x: Union[np.ndarray, jnp.ndarray, tf.Tensor, torch.Tensor, Any],
        y: Union[np.ndarray, jnp.ndarray, tf.Tensor, torch.Tensor, Any],
        size: float = 0.2,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        transform_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        prefetch: bool = True,
    ):
        """
        Load data into JAX-compatible format for JAXTRA pipelines.

        Args:
            x: Input data (NumPy, JAX, TensorFlow, or PyTorch arrays).
            y: Target labels (same types as x).
            size (float): Proportion of data for the test split (0 to 1). Default is 0.2.
            batch_size (int): Size of batches for training. Default is 32.
            seed (int): Random seed for reproducibility. Default is 42.
            shuffle (bool): Whether to shuffle data before splitting/batching. Default is True.
            transform_fn (Optional[Callable]): JAX-compatible function to transform data on-the-fly.
            prefetch (bool): Whether to prefetch batches asynchronously. Default is True.
        """
        self.x = self._to_jax_array(x)
        self.y = self._to_jax_array(y)
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.transform_fn = transform_fn or (lambda x: x)   
        self.prefetch = prefetch

        assert self.x.shape[0] == self.y.shape[0], "Mismatch in number of samples between x and y"

        self.num_samples = self.x.shape[0]
        self.split_index = int(self.num_samples * (1 - size))

        self._split_and_shuffle = jax.jit(self._split_and_shuffle)

    def _to_jax_array(self, data: Any) -> jnp.ndarray:
        """Convert various data types to JAX arrays."""
        if isinstance(data, np.ndarray):
            return jnp.array(data)
        elif isinstance(data, jnp.ndarray):
            return data
        elif isinstance(data, tf.Tensor):
            return jnp.array(data.numpy())
        elif isinstance(data, torch.Tensor):
            return jnp.array(data.detach().cpu().numpy())
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _split_and_shuffle(self, x: jnp.ndarray, y: jnp.ndarray, key):
        """JIT-compiled function to shuffle and split data."""
        indices = jax.random.permutation(key, x.shape[0]) if self.shuffle else jnp.arange(x.shape[0])
        x_shuffled, y_shuffled = x[indices], y[indices]
        x_train, x_test = x_shuffled[:self.split_index], x_shuffled[self.split_index:]
        y_train, y_test = y_shuffled[:self.split_index], y_shuffled[self.split_index:]
        return (x_train, y_train), (x_test, y_test)

    def __call__(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Load, shuffle, and split data into training and test sets.

        Returns:
            Tuple containing (x_train, y_train), (x_test, y_test).
        """
        key = jax.random.PRNGKey(self.seed)
        return self._split_and_shuffle(self.x, self.y, key)

    def batch_generator(self, x: jnp.ndarray, y: jnp.ndarray, batch_size: Optional[int] = None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Generate batches with optional prefetching and transformations.

        Args:
            x (jnp.ndarray): Input data.
            y (jnp.ndarray): Target labels.
            batch_size (Optional[int]): Override default batch size if provided.

        Yields:
            Tuple[jnp.ndarray, jnp.ndarray]: Batches of transformed data and labels.
        """
        batch_size = batch_size or self.batch_size
        num_samples = x.shape[0]
        indices = jnp.arange(num_samples)
        key = jax.random.PRNGKey(self.seed)

        if self.shuffle:
            indices = jax.random.permutation(key, num_samples)

        @jax.jit
        def _get_batch(start_idx: int, end_idx: int):
            batch_indices = indices[start_idx:end_idx]
            x_batch = self.transform_fn(x[batch_indices])
            y_batch = y[batch_indices]
            return x_batch, y_batch

        # Use jax.experimental.host_callback for true async instead
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            yield _get_batch(start_idx, end_idx)

    def parallel_batch_generator(self, num_devices: int) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Generate batches in parallel across multiple devices using jax.pmap.

        Args:
            num_devices (int): Number of devices to split batches across.

        Yields:
            Batches sharded across devices.
        """
        (x_train, y_train), _ = self()
        num_samples = x_train.shape[0]
        batches_per_device = num_samples // (self.batch_size * num_devices)
        total_batch_size = self.batch_size * num_devices

        pad_size = (total_batch_size - num_samples % total_batch_size) % total_batch_size
        if pad_size > 0:
            x_train = jnp.pad(x_train, [(0, pad_size), (0, 0)], mode="constant")
            y_train = jnp.pad(y_train, [(0, pad_size)], mode="constant")

        x_train = x_train.reshape((batches_per_device, num_devices, self.batch_size, -1))
        y_train = y_train.reshape((batches_per_device, num_devices, self.batch_size))

        @jax.pmap
        def process_batch(x_batch, y_batch):
            return self.transform_fn(x_batch), y_batch

        for i in range(batches_per_device):
            yield process_batch(x_train[i], y_train[i])
