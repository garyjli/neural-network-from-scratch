# Building a neural network from scratch
In this project, I build a neural network with numpy that aims to classify digits, using images from the MNIST database.


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
- (0th) Input layer: 784 nodes (each pixel maps to a node)
- (1st) Hidden layer: 10 nodes
- (2nd) Output layer: 10 nodes (corresponds to numbers 0-9)
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
- Similarly, for the nodes going from our 1st layer to our 2nd layer, we'll apply the same rule, but we'll use a softmax function instead:
  - $A^{[2]} = \mathrm{softmax}(W^{[2]}A^{[1]} + b^{[2]})$


## Training Overview:
- 3 parts
  - Forward propagation:
    - Take an image, run it through the network, and see what the model outputs


## Math:
- Define $A^{[0]}$ to be a matrix of all activations of our input layer (784 x $m$)
  - 784 rows $\rightarrow$ one per pixel
  - $m$ columns $\rightarrow$ one per input image
- We now want to compute: $Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$
  - $Z^{[1]}$ = matrix of all "pre-activations" of our 1st layer (10 x $m$)
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
    - softmax $(z)\_i$ = $\frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$
  - In words, softmax takes the exponential of each neuron's logit and divides it by the sum of exponentials of all logits
  - Produces a value between 0 and 1 for each neuron, where these values form a probability distribution that sums to 1