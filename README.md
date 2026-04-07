## Neural network from scratch (no Tensorflow or Pytorch)

### Notes:
- Images are from the MNIST database
- Each image is 784 pixels (28 x 28)
- Each (grayscale) pixel is just a value 0-255 (0 = black, 255 = white)
- We can represent m images (each 784 pixels) as a matrix where each row is 784 columns long and represents the pixel values of 1 image
- Take the transpose of this matrix (so now each column represents the pixel values of 1 image)

### Structure:
- 2 layers total
- Node = neuron = holds a value (in this case a grayscale value)
- This value is called the neuron's "activation"
- (0th) Input layer: 784 nodes (each pixel maps to a node)
- (1st) Hidden layer: 10 nodes
- (2nd) Output layer: 10 nodes (each corresponding to a number 0-9)

### Training:
- Forward propagation: Take an image, run it through the network, compute output
- $A^{[0]} =$ Input layer matrix