import jax.numpy as jnp
from typing import Tuple, Union
import numpy as np
import jax

class DataLoader:
    def __init__(self, x: Union[np.ndarray, jnp.ndarray], y: Union[np.ndarray, jnp.ndarray], size: float = 0.2, seed: int = 42, shuffle: bool = True):
        """
        DataLoader for reading, processing, and splitting data into training and test sets.

        Args:
            x (Union[np.ndarray, jnp.ndarray]): Input data.
            y (Union[np.ndarray, jnp.ndarray]): Target labels.
            size (float): Proportion of the dataset to include in the test split. Default is 0.2.
            seed (int): Random seed for reproducibility. Default is 42.
            shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        """
        self.x = jnp.array(x) if isinstance(x, np.ndarray) else x
        self.y = jnp.array(y) if isinstance(y, np.ndarray) else y
        self.size = size
        self.seed = seed
        self.shuffle = shuffle

    def __call__(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Load data, convert to JAX format, and split into training and test sets.

        Returns:
            Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]: 
            A tuple containing two tuples:
                - (x_train, y_train): Training data and labels.
                - (x_test, y_test): Test data and labels.
        """
        data, labels = self.x, self.y
        if self.shuffle:
            key = jax.random.PRNGKey(self.seed)
            indices = jax.random.permutation(key, data.shape[0])
            data = data[indices]
            labels = labels[indices]
            
        split_index = int(data.shape[0] * (1 - self.size))
        x_train, x_test = data[:split_index], data[split_index:]
        y_train, y_test = labels[:split_index], labels[split_index:]

        return (x_train, y_train), (x_test, y_test)

    def batch_generator(self, x: jnp.ndarray, y: jnp.ndarray, batch_size: int = 32):
        """
        Generate batches of data.

        Args:
            x (jnp.ndarray): Input data.
            y (jnp.ndarray): Target labels.
            batch_size (int): Size of each batch. Default is 32.

        Yields:
            Tuple[jnp.ndarray, jnp.ndarray]: Batches of data and labels.
        """
        num_samples = x.shape[0]
        indices = jnp.arange(num_samples)
        if self.shuffle:
            key = jax.random.PRNGKey(self.seed)
            indices = jax.random.permutation(key, num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield x[batch_indices], y[batch_indices]