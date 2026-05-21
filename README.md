# Building a neural network from scratch
In this repo, I implement a neural network from scratch using numpy to classify handwritten digits from the MNIST dataset.


## General Notes:
- Each image consists of 784 pixels (28 x 28)
  - Each pixel represents a value $\in [0, 255]$ (grayscale value)
- Thus we can represent $m$ images as a single matrix $A$ with:
  - $m$ rows
  - 784 columns
  - So each row represents all 784 pixel values of one image
- For the math behind our NN, we'll want to use the transpose of $A$, $A^T$, which has:
  - $m$ columns
  - 784 rows
  - Now each column represents one image (one training sample)
- Neuron: A node which holds a numerical value
  - This numerical value is called the neuron's "activation value"


## Our NN Structure:
- 0th layer (input): 784 nodes (each pixel maps to a node)
- 1st layer (hidden): 10 nodes
- 2nd layer (output): 10 nodes (corresponds to numbers 0-9)
- Each neuron in a successive layer is "connected" to all neurons in the previous layer
  - In other words, each neuron in layer $L$ receives input from all neurons in layer $L - 1$
  - Each connection has a weight associated with it
  - Thus each neuron has:
    - One weight per input neuron
    - One bias value
- For our neural network, the activation value of a single node $j$ in our 1st layer is calculated by:
  - Taking activation values of all nodes in the 0th layer
  - Taking weights of all connections from those nodes to $j$
  - Multiplying each activation value by its corresponding weight
  - Adding these products together along with node $j$'s bias value
  - Applying a non-linear activation function (in this case, ReLU) to the entire result
  - This is:
    - ${a_j}^{[1]} = \mathrm{ReLU}({w_{j1}}^{[1]}{a_1}^{[0]} + {w_{j2}}^{[1]}{a_2}^{[0]} + \cdots + {w_{jn}}^{[1]}{a_n}^{[0]} + {b_j}^{[1]})$
  - If we treat this as a matrix operation for all nodes in the 1st layer, then:
    - $A^{[1]} = \mathrm{ReLU}(W^{[1]}A^{[0]} + b^{[1]})$
- Similarly, for the nodes going from our 1st layer to our 2nd layer:
  - We will apply the same operation as before, but we'll use a softmax function instead:
    - $A^{[2]} = \mathrm{softmax}(W^{[2]}A^{[1]} + b^{[2]})$
- More on how to compute this later


## Training Overview:
- 3 parts
  - Forward propagation:
    - Take an image, run it through the network, and see what the model outputs
    - Compute loss
  - Backpropagation:
    - Compute gradients $dW$ and $db$, which tell us how each weight and bias should change to reduce loss
  - Update parameters:
    - Use gradients and a learning rate to update our weights and biases


## Math:
- Let $A^{[0]}$ be a matrix (784 x $m$) of all activation values from our 0th layer
  - 784 rows $\rightarrow$ one per pixel
  - $m$ columns $\rightarrow$ one per input image
- We now compute: $Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$
  - $Z^{[1]}$ is a matrix (10 x $m$) of all "pre-activation" values of our 1st layer
    - This matrix is 10 x $m$ since we have 10 neurons in our 1st layer
  - $W^{[1]}$ is a matrix (10 x 784) of all the weights of the connections from our 0th layer to our 1st layer
    - 10 rows $\rightarrow$ each row consists of 784 weights of all the connections between neurons in the 0th layer and a particular neuron in the 1st layer
    - Note: These weights are shared across all different input images
  - $b^{[1]}$ is a vector (10 x 1) of the bias values of each neuron in our 1st layer
    - 10 rows $\rightarrow$ each row element is the bias of a neuron in the 1st layer
    - Note: This bias is shared across all different input images
- Let $A^{[1]} = \mathrm{ReLU}(Z^{[1]})$
  - $A^{[1]}$ is a matrix (10 x $m$) of all activation values from our 1st layer
    - 10 rows $\rightarrow$ one activation per neuron
    - $m$ columns $\rightarrow$ one per input image
- We now compute: $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
  - $Z^{[2]}$ is a matrix (10 x $m$) of all "pre-activation" values of our 2nd layer
    - This matrix is 10 x $m$ since we have 10 neurons in our 2nd layer
  - $W^{[2]}$ is a matrix (10 x 10) of all the weights of the connections from our 1st layer to our 2nd layer
    - 10 rows $\rightarrow$ each row consists of 10 weights of all the connections between neurons in the 1st layer and a particular neuron in the 2nd layer
  - $b^{[2]}$ is a vector (10 x 1) of the bias values of each neuron in our 2nd layer
    - 10 rows $\rightarrow$ each row element is the bias of a neuron in the 2nd layer
- Let $A^{[2]} = \mathrm{softmax}(Z^{[2]})$
  - $A^{[2]}$ is a matrix (10 x $m$) of our probability distributions
    - 10 rows $\rightarrow$ one probability for each of our 10 classes (digits 0-9)
    - $m$ columns $\rightarrow$ one distribution per input image
- Softmax function:
  - For a vector $z = (z_1, z_2, \ldots, z_n)$:
    - $\mathrm{softmax}(z)_i$ = $\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$
  - In words, softmax takes the exponential of each neuron's logit (pre-activation value) and divides it by the sum of exponentials of all logits
  - Produces a value between 0 and 1 for each neuron, where these values form a probability distribution that sums to 1