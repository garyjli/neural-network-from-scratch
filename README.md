## Neural network from scratch (no Tensorflow or Pytorch)

### General Notes:
- Images are from the MNIST database
- Each image is 784 pixels (28 x 28)
- Each (grayscale) pixel is just a value between 0-255 inclusive (0 = black, 255 = white)
- We can represent $m$ images (each 784 pixels) as a matrix:
  - Each row is 784 columns long
  - Each row represents the pixel values of 1 image
- Take the transpose of this matrix:
  - Now each column represents the pixel values of 1 image

### Terminology:
- Node = neuron = holds a numerical value (in this case a grayscale value)
- This value is called the neuron's "activation"

### Structure:
- 2 layers total
- (0th) Input layer: 784 nodes (each pixel maps to a node)
- (1st) Hidden layer: 10 nodes
- (2nd) Output layer: 10 nodes (each corresponding to a number 0-9)

### Math & Training:
- Let $A^{[0]}$ = Input layer matrix ($784 x m$)
- Let $Z^{[1]}$ = Hidden layer matrix ($10 x m$)
- Let $A^{[1]}$ = Output layer matrix
- Forward propagation: Take image, run through network, compute output