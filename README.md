# Text-To-Image
## Generative Adversarial Network (GAN) using CIFAR-10 Dataset
### Overview
This project demonstrates the implementation of a Generative Adversarial Network (GAN) using the CIFAR-10 dataset. The GAN consists of a Generator and a Discriminator network, trained in an adversarial manner to generate realistic RGB images.

### Table of Contents
### Prerequisites
Installation
Usage
Training the GAN
Generating Images
### Model Architecture
Generator
Discriminator
Hyperparameters
Dataset
Results
Acknowledgements
### Prerequisites
Python 3.7 or higher
PyTorch
torchvision
matplotlib
nltk

### Usage
Training the GAN

### Generating Images
After training the GAN, you can generate images based on a text prompt:

### Model Architecture
### Generator
The Generator takes a noise vector as input and transforms it through several layers of transposed convolution to produce an RGB image. Each layer is followed by batch normalization and a ReLU activation function, except the last layer, which uses a Tanh activation function.

### Discriminator
The Discriminator takes an image as input and processes it through several layers of convolution, followed by batch normalization and LeakyReLU activation functions. The final layer outputs a probability using a Sigmoid activation function, indicating whether the input image is real or fake.

### Hyperparameters
latent_dim: 100 (Dimensionality of the noise vector)
base_feature_size: 64 (Base number of feature maps)
num_channels: 3 (RGB images)
learning_rate: 0.0002
beta1: 0.5 (Beta1 parameter for Adam optimizer)
num_epochs: 25
batch_size: 128
### Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. For this project, the images are resized to 128x128 pixels.

### Results
During training, the Discriminator and Generator losses are printed periodically. After training, you can generate and visualize images based on text prompts.

### Acknowledgements
This implementation is based on the original GAN paper by Ian Goodfellow et al. (2014).
The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research.
Feel free to contribute to this project by opening issues or submitting pull requests. Happy coding!
