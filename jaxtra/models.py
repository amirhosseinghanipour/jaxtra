import jax
import jax.numpy as jnp
from typing import Callable, Optional, Union, Tuple
import json
import numpy as np
from jax import random

class Sequential:
    """
    A sequential model that stacks layers linearly, compatible with JAX and DataLoader.

    Args:
        layers (list[Callable]): List of layers (e.g., Dense, Conv2D) to be added to the model.
        device (str): Device to run the model on ('cpu', 'gpu', or 'tpu'). Default is 'cpu'.
    """

    def __init__(self, layers: list[Callable], device: str = "cpu"):
        """
        Initialize the Sequential model.

        Raises:
            ValueError: If layers list is empty or device is invalid.
            TypeError: If layers are not callable or lack required methods.
        """
        if not layers or not isinstance(layers, list):
            raise ValueError("layers must be a non-empty list")
        for layer in layers:
            if not callable(layer):
                raise TypeError(f"All layers must be callable, got {type(layer)} for {layer}")
            # Check for required methods (except for layers like Flatten that don’t have params)
            if not hasattr(layer, 'serialize') or not hasattr(layer, 'deserialize'):
                raise TypeError(f"Layer {layer.__class__.__name__} must implement serialize and deserialize")
        self.layers = layers

        try:
            self.device = jax.devices(device)[0]
        except Exception as e:
            raise ValueError(f"Invalid device '{device}': {str(e)}") from e

        self.training = True
        self.optimizer = None
        self.loss = None
        self.params = None
        self.key = random.PRNGKey(0)  # For dropout during training

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the model.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after passing through all layers.

        Raises:
            ValueError: If input is not a JAX array or model is not compiled.
        """
        if not isinstance(x, jnp.ndarray):
            raise ValueError(f"Input must be a JAX array, got {type(x)}")
        if self.params is None:
            raise ValueError("Model must be compiled before calling")
        x = jax.device_put(x, self.device)
        return self.forward(x)

    def compile(self, optimizer: Callable, loss: Callable, sample_input: Optional[jnp.ndarray] = None):
        """
        Compile the model with an optimizer and loss function.

        Args:
            optimizer (Callable): Optimizer function (e.g., from jax.example_libraries.optimizers).
            loss (Callable): Loss function taking (targets, predictions) as arguments.
            sample_input (Optional[jnp.ndarray]): Sample input to initialize parameters.

        Raises:
            TypeError: If optimizer or loss is not callable.
            ValueError: If sample_input is needed but not provided or invalid.
        """
        if not callable(optimizer):
            raise TypeError(f"optimizer must be callable, got {type(optimizer)}")
        if not callable(loss):
            raise TypeError(f"loss must be callable, got {type(loss)}")

        self.optimizer = optimizer
        self.loss = loss

        if self.params is None:
            if sample_input is None:
                raise ValueError("sample_input must be provided to initialize parameters on first compile")
            if not isinstance(sample_input, jnp.ndarray):
                raise ValueError(f"sample_input must be a JAX array, got {type(sample_input)}")
            sample_input = jax.device_put(sample_input, self.device)
            self.params = self.get_params()
            _ = self.forward(sample_input)  # Ensure params are initialized

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the model.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after passing through all layers.
        """
        self.key, subkey = random.split(self.key)
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = self.training
            x = layer(x, key=subkey if self.training and hasattr(layer, 'dropout_rate') and layer.dropout_rate > 0 else None)
        return x

    def fit(self, data_loader: 'DataLoader', epochs: int = 1, callbacks: Optional[list[Callable]] = None):
        """
        Train the model using a DataLoader for a fixed number of epochs.

        Args:
            data_loader (DataLoader): DataLoader instance providing batched data.
            epochs (int): Number of epochs to train the model. Defaults to 1.
            callbacks (Optional[list[Callable]]): List of callback functions to be called during training.

        Raises:
            ValueError: If data_loader is invalid, epochs is non-positive, or model is not compiled.
            TypeError: If callbacks are not callable.
        """
        from data_loader import DataLoader  # Import here to avoid circular import

        if not isinstance(data_loader, DataLoader):
            raise ValueError(f"data_loader must be a DataLoader instance, got {type(data_loader)}")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}")
        if self.optimizer is None or self.loss is None:
            raise ValueError("Model must be compiled with an optimizer and loss before training")
        if callbacks is not None:
            if not isinstance(callbacks, list):
                raise ValueError(f"callbacks must be a list, got {type(callbacks)}")
            for cb in callbacks:
                if not callable(cb):
                    raise TypeError(f"All callbacks must be callable, got {type(cb)}")

        for epoch in range(epochs):
            def loss_fn(params, x, y):
                self.set_params(params)
                preds = self.forward(x)
                # Add regularization losses
                reg_loss = sum(layer.total_loss(preds) for layer in self.layers if hasattr(layer, 'total_loss'))
                return self.loss(y, preds) + reg_loss

            total_loss = 0.0
            num_batches = 0
            self.training = True
            for x_batch, y_batch in data_loader:
                x_batch = jax.device_put(x_batch, self.device)
                y_batch = jax.device_put(y_batch, self.device)
                params = self.get_params()
                try:
                    self.key, subkey = random.split(self.key)
                    grads = jax.grad(loss_fn)(params, x_batch, y_batch)
                    updated_params = self.optimizer.update(params, grads)
                    self.set_params(updated_params)
                    total_loss += loss_fn(updated_params, x_batch, y_batch).item()
                    num_batches += 1
                except Exception as e:
                    raise RuntimeError(f"Training failed at epoch {epoch + 1}, batch {num_batches + 1}: {str(e)}") from e

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

            if callbacks:
                for callback in callbacks:
                    try:
                        callback(self)
                    except Exception as e:
                        raise RuntimeError(f"Callback failed at epoch {epoch + 1}: {str(e)}") from e

    def evaluate(self, data_loader: 'DataLoader') -> float:
        """
        Evaluate the model using a DataLoader.

        Args:
            data_loader (DataLoader): DataLoader instance providing batched data.

        Returns:
            float: Average loss value across all batches.

        Raises:
            ValueError: If data_loader is invalid or model is not compiled.
        """
        from data_loader import DataLoader  # Import here to avoid circular import

        if not isinstance(data_loader, DataLoader):
            raise ValueError(f"data_loader must be a DataLoader instance, got {type(data_loader)}")
        if self.loss is None:
            raise ValueError("Model must be compiled with a loss function before evaluation")

        self.training = False
        total_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in data_loader:
            x_batch = jax.device_put(x_batch, self.device)
            y_batch = jax.device_put(y_batch, self.device)
            try:
                preds = self.forward(x_batch)
                reg_loss = sum(layer.total_loss(preds) for layer in self.layers if hasattr(layer, 'total_loss'))
                total_loss += (self.loss(y_batch, preds) + reg_loss).item()
                num_batches += 1
            except Exception as e:
                raise RuntimeError(f"Evaluation failed at batch {num_batches + 1}: {str(e)}") from e
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        print("Model Summary:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {layer.__class__.__name__}")

    def save(self, filepath: str):
        """
        Save the model parameters and architecture to a file.

        Args:
            filepath (str): Path to save the model file (will append '.jxt' if not present).

        Raises:
            RuntimeError: If serialization fails.
        """
        if not isinstance(filepath, str) or not filepath:
            raise ValueError("filepath must be a non-empty string")

        if not filepath.endswith('.jxt'):
            filepath += '.jxt'

        try:
            model_dict = {
                'layers': [layer.serialize() for layer in self.layers],
                'device': str(self.device)
            }
            params = self.get_params()
            with open(filepath, 'wb') as f:
                np.savez(f, architecture=json.dumps(model_dict), params=np.array(params, dtype=object))
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {filepath}: {str(e)}") from e

    @classmethod
    def load(cls, filepath: str) -> 'Sequential':
        """
        Load a model from a file.

        Args:
            filepath (str): Path to the model file (will append '.jxt' if not present).

        Returns:
            Sequential: Loaded model instance.

        Raises:
            RuntimeError: If loading fails due to file issues or incompatible data.
        """
        if not isinstance(filepath, str) or not filepath:
            raise ValueError("filepath must be a non-empty string")

        if not filepath.endswith('.jxt'):
            filepath += '.jxt'

        try:
            with open(filepath, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                model_dict = json.loads(data['architecture'])
                params = data['params'].tolist()

            from layers import Dense, Conv1D, Conv2D, Conv3D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Dropout, Flatten, BatchNormalization
            layer_classes = {
                'Dense': Dense, 'Conv1D': Conv1D, 'Conv2D': Conv2D, 'Conv3D': Conv3D,
                'MaxPooling1D': MaxPooling1D, 'MaxPooling2D': MaxPooling2D, 'MaxPooling3D': MaxPooling3D,
                'Dropout': Dropout, 'Flatten': Flatten, 'BatchNormalization': BatchNormalization
            }

            layers = []
            for layer_data in model_dict['layers']:
                layer_type = layer_data.get('type', layer_data.get('__class__'))  # Flexible key
                if layer_type not in layer_classes:
                    raise ValueError(f"Unsupported layer type: {layer_type}")
                layers.append(layer_classes[layer_type].deserialize(layer_data))

            model = cls(layers, device=model_dict['device'])
            model.set_params(params)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filepath}: {str(e)}") from e

    def get_params(self) -> list:
        """
        Get the parameters of all layers in the model.

        Returns:
            list: List of parameters for each layer.

        Raises:
            RuntimeError: If parameter retrieval fails.
        """
        try:
            return [layer.get_params() if hasattr(layer, 'get_params') else None for layer in self.layers]
        except Exception as e:
            raise RuntimeError(f"Failed to get model parameters: {str(e)}") from e

    def set_params(self, params: list):
        """
        Set the parameters of all layers in the model.

        Args:
            params (list): List of parameters for each layer.

        Raises:
            ValueError: If params length doesn’t match layers or is invalid.
            RuntimeError: If setting parameters fails.
        """
        if not isinstance(params, list) or len(params) != len(self.layers):
            raise ValueError(f"params must be a list matching the number of layers ({len(self.layers)}), got {len(params) if isinstance(params, list) else type(params)}")
        try:
            for layer, layer_params in zip(self.layers, params):
                if hasattr(layer, 'set_params') and layer_params is not None:
                    layer.set_params(layer_params)
            self.params = params
        except Exception as e:
            raise RuntimeError(f"Failed to set model parameters: {str(e)}") from e