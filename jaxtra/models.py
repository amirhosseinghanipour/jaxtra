import jax
import jax.numpy as jnp
from typing import List, Callable, Optional

class Sequential:
    """
    A sequential model that stacks layers linearly.

    Args:
        layers (List[Callable]): List of layers to be added to the model.
    """
    def __init__(self, layers: List[Callable]):
        self.layers = layers
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

    def fit(self, x: jnp.ndarray, y: jnp.ndarray, epochs: int = 1, callbacks: Optional[List[Callable]] = None):
        """
        Train the model for a fixed number of epochs.

        Args:
            x (jnp.ndarray): Input data.
            y (jnp.ndarray): Target data.
            epochs (int): Number of epochs to train the model. Defaults to 1.
            callbacks (Optional[List[Callable]]): List of callback functions to be called during training. Defaults to None.
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
        params = self.get_params()
        with open(filepath, 'wb') as f:
            jnp.save(f, params)

    def load(self, filepath: str):
        """
        Load the model parameters from a file.

        Args:
            filepath (str): Path to the file from which the parameters will be loaded.
        """
        with open(filepath, 'rb') as f:
            params = jnp.load(f, allow_pickle=True)
        self.set_params(params)

    def get_params(self):
        """
        Get the parameters of all layers in the model.

        Returns:
            List: List of parameters for each layer.
        """
        params = []
        for layer in self.layers:
            params.append(layer.get_params())
        return params

    def set_params(self, params):
        """
        Set the parameters of all layers in the model.

        Args:
            params (List): List of parameters for each layer.
        """
        for layer, layer_params in zip(self.layers, params):
            layer.set_params(layer_params)