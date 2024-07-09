from setuptools import setup, find_packages
import os

VERSION = '0.0.2'
DESCRIPTION = 'JAXTRA'
LONG_DESCRIPTION = 'A package for building and training neural networks with JAX.'

# Setting up
setup(
    name="jaxtra",
    version=VERSION,
    author="Amirhossein Ghanipour",
    author_email="<amirhosseinghanipour@fgn.ui.ac.ir>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['jax'],
    keywords=['jax', 'deep learning', 'neural networks', 'artificial intelligence'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux"
    ],
    python_requires=">=3.11.9",
)