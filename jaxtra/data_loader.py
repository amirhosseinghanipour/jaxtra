import jax
import jax.numpy as jnp
from typing import Tuple, Union, Callable, Optional, Iterator, Any
import numpy as np
import tensorflow as tf
import torch
from jax import random
import queue
import threading
import traceback

class DataLoader:
    """
    A JAX-compatible DataLoader.

    This class provides an efficient way to iterate over datasets in batches, with support for shuffling,
    transformation, prefetching, and splitting into training and test sets. It is designed to handle various
    input data types (NumPy, JAX, TensorFlow, PyTorch) and ensures compatibility with JAX's functional paradigm.

    Attributes:
        x (jnp.ndarray): Input data converted to a JAX array.
        y (jnp.ndarray): Target labels converted to a JAX array.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the dataset before batching.
        seed (int): Random seed for reproducibility during shuffling.
        transform_fn (Callable): Function to apply transformations to the input data on-the-fly.
        prefetch (bool): Whether to use asynchronous prefetching for batches.
        prefetch_buffer_size (int): Number of batches to prefetch asynchronously.
        drop_last (bool): Whether to drop the last incomplete batch if the dataset size is not divisible by batch_size.
        num_samples (int): Total number of samples in the dataset.
        key (jax.random.PRNGKey): JAX PRNG key for randomness.

    Methods:
        __init__: Initializes the DataLoader with the provided dataset and configuration.
        _to_jax_array: Converts input data to a JAX array.
        __len__: Returns the number of batches per epoch.
        __iter__: Provides an iterator over the batches of data.
        split: Splits the dataset into training and test DataLoaders.
    """

    def __init__(
        self,
        x: Union[np.ndarray, jnp.ndarray, tf.Tensor, torch.Tensor, Any],
        y: Union[np.ndarray, jnp.ndarray, tf.Tensor, torch.Tensor, Any],
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        transform_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        prefetch: bool = False,
        prefetch_buffer_size: int = 2,
        drop_last: bool = False,
    ):
        """
        Initialize the DataLoader.

        Args:
            x: Input data (NumPy, JAX, TensorFlow, or PyTorch arrays).
            y: Target labels (same types as x).
            batch_size (int): Size of batches. Default is 32.
            shuffle (bool): Whether to shuffle data before batching. Default is True.
            seed (Optional[int]): Random seed for shuffling. Default is None (random seed).
            transform_fn (Optional[Callable]): JAX-compatible function to transform data on-the-fly.
            prefetch (bool): Whether to prefetch batches asynchronously. Default is False.
            prefetch_buffer_size (int): Number of batches to prefetch. Default is 2.
            drop_last (bool): Whether to drop the last incomplete batch. Default is False.

        Raises:
            ValueError: If `batch_size` is non-positive, `prefetch_buffer_size` is invalid,
                        `x` and `y` have mismatched sample counts, or `transform_fn` is not callable.
            TypeError: If input data cannot be converted to JAX arrays or `transform_fn` is invalid.
        """
        try:
            self.x = self._to_jax_array(x)
            self.y = self._to_jax_array(y)
        except Exception as e:
            raise TypeError(f"Failed to convert input data to JAX arrays: {str(e)}") from e

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        self.batch_size = batch_size

        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle must be a boolean, got {type(shuffle)}")
        self.shuffle = shuffle

        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"seed must be an integer or None, got {type(seed)}")
        self.seed = seed if seed is not None else np.random.randint(0, 2**32)

        if transform_fn is not None:
            if not callable(transform_fn):
                raise ValueError(f"transform_fn must be callable, got {type(transform_fn)}")
            try:
                test_sample = jnp.ones((1,) + self.x.shape[1:], dtype=self.x.dtype)
                _ = transform_fn(test_sample)
            except Exception as e:
                raise TypeError(f"transform_fn is not JAX-compatible: {str(e)}") from e
        self.transform_fn = transform_fn or (lambda x: x)

        if not isinstance(prefetch, bool):
            raise ValueError(f"prefetch must be a boolean, got {type(prefetch)}")
        self.prefetch = prefetch

        if not isinstance(prefetch_buffer_size, int) or prefetch_buffer_size <= 0:
            raise ValueError(f"prefetch_buffer_size must be a positive integer, got {prefetch_buffer_size}")
        self.prefetch_buffer_size = max(1, prefetch_buffer_size)

        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last must be a boolean, got {type(drop_last)}")
        self.drop_last = drop_last

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"Mismatch in number of samples: x has {self.x.shape[0]}, y has {self.y.shape[0]}")
        self.num_samples = self.x.shape[0]
        
        if self.num_samples == 0:
            raise ValueError("Dataset is empty; x and y must contain at least one sample")
        
        self.key = random.PRNGKey(self.seed)

    def _to_jax_array(self, data: Any) -> jnp.ndarray:
        """
        Convert various data types to JAX arrays.

        Args:
            data: Input data (NumPy, JAX, TensorFlow, or PyTorch arrays).

        Returns:
            jnp.ndarray: Data converted to a JAX array.

        Raises:
            ValueError: If the input data type is unsupported or conversion fails.
        """
        if data is None:
            raise ValueError("Input data cannot be None")
        try:
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
        except Exception as e:
            raise ValueError(f"Failed to convert data to JAX array: {str(e)}") from e

    def __len__(self) -> int:
        """
        Return the number of batches per epoch.

        If `drop_last` is True, the last incomplete batch is dropped, and the count reflects only full batches.

        Returns:
            int: Number of batches per epoch.
        """
        if self.num_samples == 0:
            return 0
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Provide an iterator over batches of data.

        Yields:
            Tuple[jnp.ndarray, jnp.ndarray]: Batches of transformed data and labels.

        Raises:
            RuntimeError: If prefetching fails due to threading issues or batch generation errors.
        """
        if self.num_samples == 0:
            return iter([])

        self.key, subkey = random.split(self.key)
        indices = random.permutation(subkey, self.num_samples) if self.shuffle else jnp.arange(self.num_samples)

        @jax.jit
        def _get_batch(indices_batch: jnp.ndarray):
            """
            Extract a batch of data using dynamic indices.

            Args:
                indices_batch (jnp.ndarray): Indices of the samples to include in the batch.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: Batch of transformed data and corresponding labels.
            """
            x_batch = self.transform_fn(jnp.take(self.x, indices_batch, axis=0))
            y_batch = jnp.take(self.y, indices_batch, axis=0)
            return x_batch, y_batch

        if self.prefetch:
            q = queue.Queue(maxsize=self.prefetch_buffer_size)
            exception = None

            def fill_queue():
                """
                Fill the prefetch queue with batches of data.

                This function runs in a background thread and stops when all batches are added to the queue.
                """
                nonlocal exception
                try:
                    for start in range(0, self.num_samples, self.batch_size):
                        end = min(start + self.batch_size, self.num_samples)
                        if self.drop_last and (end - start) < self.batch_size:
                            break
                        batch_indices = indices[start:end]
                        batch = _get_batch(batch_indices)
                        q.put(batch)
                    q.put(None)
                except Exception as e:
                    exception = e
                    q.put(None)

            thread = threading.Thread(target=fill_queue)
            thread.start()

            while True:
                batch = q.get()
                if batch is None:
                    thread.join()
                    if exception is not None:
                        raise RuntimeError(f"Prefetching failed: {str(exception)}\n{traceback.format_exc()}") from exception
                    break
                yield batch
        else:
            for start in range(0, self.num_samples, self.batch_size):
                end = min(start + self.batch_size, self.num_samples)
                if self.drop_last and (end - start) < self.batch_size:
                    break
                batch_indices = indices[start:end]
                try:
                    yield _get_batch(batch_indices)
                except Exception as e:
                    raise RuntimeError(f"Batch generation failed at indices {start}:{end}: {str(e)}") from e

    def split(self, test_size: float = 0.2) -> Tuple['DataLoader', 'DataLoader']:
        """
        Split the dataset into training and test DataLoaders.

        Args:
            test_size (float): Proportion of data for the test set (0 to 1). Default is 0.2.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and test DataLoaders.

        Raises:
            ValueError: If `test_size` is not between 0 and 1 or results in an empty split.
        """
        if not isinstance(test_size, (int, float)) or not 0 <= test_size <= 1:
            raise ValueError(f"test_size must be a number between 0 and 1, got {test_size}")

        split_idx = int(self.num_samples * (1 - test_size))
        if split_idx == 0 or split_idx == self.num_samples:
            raise ValueError(f"test_size {test_size} results in an empty training or test set with {self.num_samples} samples")

        self.key, subkey = random.split(self.key)
        indices = random.permutation(subkey, self.num_samples) if self.shuffle else jnp.arange(self.num_samples)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        try:
            train_x = jnp.take(self.x, train_indices, axis=0)
            train_y = jnp.take(self.y, train_indices, axis=0)
            test_x = jnp.take(self.x, test_indices, axis=0)
            test_y = jnp.take(self.y, test_indices, axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to split dataset: {str(e)}") from e

        try:
            train_loader = DataLoader(
                train_x, train_y, batch_size=self.batch_size, shuffle=self.shuffle,
                seed=self.seed, transform_fn=self.transform_fn, prefetch=self.prefetch,
                prefetch_buffer_size=self.prefetch_buffer_size, drop_last=self.drop_last
            )
            test_loader = DataLoader(
                test_x, test_y, batch_size=self.batch_size, shuffle=self.shuffle,
                seed=self.seed, transform_fn=self.transform_fn, prefetch=self.prefetch,
                prefetch_buffer_size=self.prefetch_buffer_size, drop_last=self.drop_last
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create split DataLoaders: {str(e)}") from e

        return train_loader, test_loader