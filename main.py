# coding: utf-8

import random
import numpy
from PIL import Image
from simple_two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

# Network parameters
input_size  = 784
hidden_size = 100
output_size = 10

# Weight file path
weightFilePath = 'mnist_weights_784_100_10.hdf5'

# Initialize
network = TwoLayerNet(input_size, hidden_size, output_size)
network.getWeightsFromFile(weightFilePath)
network.showWeightSize()

# Load image to predict
(x_train, t_train), _ = load_mnist(flatten=True, normalize=False)
idx = random.randint(0, x_train.shape[0]-1)

# Predict image
result = network.predict(x_train[idx])
print(result)

# Show image data
img = x_train[idx].reshape(28, 28) 
pil_img = Image.fromarray(numpy.uint8(img))
pil_img.show()




