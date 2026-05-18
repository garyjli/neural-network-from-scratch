# Building a neural network using only numpy

## General Notes:
- Images are from the MNIST database
- Each image is 784 pixels (28 x 28)
- Each (grayscale) pixel of an image is just a value between 0-255 inclusive (0 = black, 255 = white)
- We can represent $m$ images (each 784 pixels) as a single matrix:
  - $m$ rows, 784 columns
  - So each row represents all 784 pixel values of one image
- Take the transpose of this matrix:
  - Now each column represents the pixel values of one image
- Node/neuron: Holds a numerical value (in this case a grayscale value)
  - This value is called the neuron's "activation"

## Our NN Structure:
- 2 layers total
- (0th) Input layer: 784 nodes (each pixel maps to a node)
- (1st) Hidden layer: 10 nodes
- (2nd) Output layer: 10 nodes (corresponds to numbers 0-9)
- Each neuron in a successive layer is "connected" to all neurons in the previous layer
  - In other words, every neuron in layer $L$ receives input from every neuron in layer $L - 1$
  - Thus each neuron has one weight per input neuron and one bias
  - Each connection has a "weight" associated with it
- Example: The activation of a particular node in the 1st layer is calculated by applying a (nonlinear) activation function to a linear combination of weights $\times$ activation values (plus a bias value) from the 0th layer
  - Example: $g(w_{1}a_{1} + w_{2}a_{2} + \cdots + w_{n}a_{n} + b)$

## Math:
- Let $A^{[0]}$ = matrix of all activations of our 0th layer (784 x $m$)
  - 784 rows $\rightarrow$ one per pixel
  - $m$ columns $\rightarrow$ one per input image
- Let $Z^{[1]}$ = matrix of all "pre-activations" of our 1st layer (10 x $m$)
- Specifically: $Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$
  - $W^{[1]}$ = 10 x 784 matrix
    - 10 rows $\rightarrow$ each row consists of 784 weights of all the connections between the 0th layer and a particular neuron in the 1st layer
    - These weights are shared across all different input images
  - $b^{[1]}$ = 10 x 1 vector
    - 10 rows $\rightarrow$ each entry corresponds to the bias of a particular neuron in the 1st layer
    - This bias is shared across all different input images
- Let $A^{[1]}$ = matrix of all activations of our 1st layer (10 x $m$)
  - 10 rows $\rightarrow$ one activation per neuron
  - $m$ columns $\rightarrow$ one per input image
  - This matrix is achieved by applying an activation function (e.g. tanh, sigmoid, ReLU) to $Z^{[1]}$
    - $A^{[1]} = g(Z^{[1]})$
- Let $Z^{[2]}$ = matrix of all "pre-activations" (logits) of our 2nd layer (10 x $m$)
- Specifically: $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
  - $W^{[2]}$ = 10 x 10 matrix
    - 10 rows $\rightarrow$ each row consists of 10 weights of all the connections between the 1st layer and a particular neuron in the 2nd layer
  - $b^{[2]}$ = 10 x 1 vector
    - 10 rows $\rightarrow$ each entry corresponds to the bias of a particular neuron in the 2nd layer
- Let $A^{[2]}$ = matrix of probability distributions (10 x $m$)
  - 10 rows $\rightarrow$ one probability per each of our 10 classes (digits 0-9)
  - $m$ columns $\rightarrow$ one distribution per input image
  - This matrix is achieved by applying a softmax function to $Z^{[2]}$
    - $A^{[2]}$ = softmax $(Z^{[2]})$
- Softmax function:
  - For a vector $z = (z_1, z_2, \ldots, z_n)$:
    - softmax $(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$
  - In words, softmax takes the exponential of each neuron's logit and divides it by the sum of exponentials of all logits
  - Produces a value between 0 and 1 for each neuron, where these values form a probability distribution that sums to 1

## Training:
- Forward propagation: Take image, run through network, compute output
- Back propagation: Start with a prediction, find out how much prediction deviated by the actual label (error), observe how each weight contributed to this error, adjust accordingly