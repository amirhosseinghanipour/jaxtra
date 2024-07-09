# JAXTRA
JAXTRA is a high-level deep learning library built on top of JAX. JAXTRA provides a high-level API for building and training neural networks, and a low-level API for building custom layers and activations. It provides a variety of neural network layers, activation functions, and utilities to help you build and train deep learning models with ease. JAXTRA is designed to be easy to use and flexible, and is suitable for both research and production use cases. 

## Features 
- High-level API for building and training neural networks
- Low-level API for building custom layers and activations
- Variety of neural network layers, activation functions, and utilities
- Easy to use and flexible
- Suitable for both research and production use cases

## Installation
JAXTRA is available on PyPI and can be installed using pip:
```bash
pip install jaxtra
```

## Usage
Here is a simple example of how to use JAXTRA to create and train a neural network:

```python
from jaxtra import Dense, InputLayer
import jax

#  Initialize random key
key = jax.random.PRNGKey(0)

# Define the model
model = InputLayer(input_shape=(32,))
model = Dense(units=64, activation='relu')
model = Dense(units=10, activation='softmax')

# Define the loss function
loss = jax.scipy.special.softmax_cross_entropy

# Define the optimizer
optimizer = jax.optimizers.Adam(learning_rate=0.01)

# Compile the model
model.compile(optimizer=optimizer, loss=loss)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## Acknowledgements

- [JAX](https://github.com/google/jax) - JAX is used as the backend for automatic differentiation and GPU/TPU acceleration.

## Contact

If you have any questions or feedback, feel free to open an issue on GitHub or contact the project maintainer at [amirhosseinghanipour@fgn.ui.ac.ir](mailto:amirhosseinghanipour@fgn.ui.ac.ir).

## License
JAXTRA is licensed under the MIT License - see the LICENSE file for details.