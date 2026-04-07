## Neural network from scratch (no Tensorflow or Pytorch)

### Notes:
- Images are from the MNIST database
- Each image is 784 pixels (28 x 28)
- Each (grayscale) pixel is just a value 0-255 (0 = black, 255 = white)
- We can represent m images (each 784 pixels) as a matrix where each row is 784 columns long and represents the pixel values of 1 image
- Take the transpose of this matrix (so now each column represents the pixel values of 1 image)