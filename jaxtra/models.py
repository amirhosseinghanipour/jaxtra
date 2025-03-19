import jax
import jax.numpy as jnp
from typing import Callable, Optional
import json
import numpy as np
from layers import Conv2D, Dense

class Sequential:
    """
    A sequential model that stacks layers linearly.

    Args:
        layers (list[Callable]): list of layers to be added to the model.
    """
    def __init__(self, layers: list[Callable], device: str = "cpu"):
        self.layers = layers
        self.device = jax.devices(device)[0]
        self.training = True

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the model.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = jax.device_put(x, self.device)
            for layer in self.layers:
                x = layer(x)
        return x
    
    def compile(self, x: jnp.ndarray, y: jnp.ndarray, optimizer: Callable, loss: Callable):
        """
        Compile the model with an optimizer and loss function.

        Args:
            optimizer (Callable): Optimizer function.
            loss (Callable): Loss function.
            sample_input (jnp.ndarray): Sample input tensor for initializing the model.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.forward(x)
        
        initial_loss = self.loss(y, self.forward(x))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the model.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, x: jnp.ndarray, y: jnp.ndarray, epochs: int = 1, callbacks: Optional[list[Callable]] = None):
        """
        Train the model for a fixed number of epochs.

        Args:
            x (jnp.ndarray): Input data.
            y (jnp.ndarray): Target data.
            epochs (int): Number of epochs to train the model. Defaults to 1.
            callbacks (Optional[list[Callable]]): list of callback functions to be called during training. Defaults to None.
        """
        for epoch in range(epochs):
            def loss_fn(params, x, y):
                self.set_params(params)
                preds = self.forward(x)
                return self.loss(y, preds)

            params = self.get_params()
            grads = jax.grad(loss_fn)(params, x, y)
            updated_params = self.optimizer.update(params, grads)
            self.set_params(updated_params)
            print(f"Epoch {epoch + 1}/{epochs} completed")

            if callbacks:
                for callback in callbacks:
                    callback(self)

    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Evaluate the model on the given data.

        Args:
            x (jnp.ndarray): Input data.
            y (jnp.ndarray): Target data.

        Returns:
            float: Loss value.
        """
        self.training = False
        preds = self.forward(x)
        loss_value = self.loss(y, preds)
        self.training = True
        return loss_value

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        print("Model Summary:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {layer.__class__.__name__}")

    def save(self, filepath: str):
        """
        Save the model parameters to a file.

        Args:
            filepath (str): Path to the file where the parameters will be saved.
        """
        if not filepath.endswith('.jxt'):
            filepath += '.jxt'
        
        # Serialize architecture and parameters
        model_dict = {
            'layers': [layer.serialize() for layer in self.layers],
            'device': str(self.device)
        }
        params = self.get_params()
        
        # Save to file
        with open(filepath, 'wb') as f:
            # Save architecture as JSON and parameters as NumPy arrays
            np.savez(f, architecture=json.dumps(model_dict), params=params)

    @classmethod
    def load(cls, filepath: str):
        """
        Load the model parameters from a file.

        Args:
            filepath (str): Path to the file from which the parameters will be loaded.
        """
        if not filepath.endswith('.jxt'):
            filepath += '.jxt'
        
        with open(filepath, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            model_dict = json.loads(data['architecture'])
            params = data['params']

        # Reconstruct layers
        layers = []
        for layer_data in model_dict['layers']:
            layer_type = layer_data.pop('type', None)  # Assume layers include type info
            if layer_type == 'Dense':
                layers.append(Dense.deserialize(layer_data))
            elif layer_type == 'Conv2D':
                layers.append(Conv2D.deserialize(layer_data))
            # Add other layer types as needed

        # Create model and set parameters
        model = cls(layers, device=model_dict['device'])
        model.set_params(params)
        return model

    def get_params(self):
        """
        Get the parameters of all layers in the model.

        Returns:
            list: list of parameters for each layer.
        """
        params = []
        for layer in self.layers:
            params.append(layer.get_params())
        return params

    def set_params(self, params):
        """
        Set the parameters of all layers in the model.

        Args:
            params (list): list of parameters for each layer.
        """
        for layer, layer_params in zip(self.layers, params):
            layer.set_params(layer_params)