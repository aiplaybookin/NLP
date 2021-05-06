#### Session 1 
***
## Example of Feed forward Deep Neural Network 

Quest 		: Create a feed forward DNN with 44 parameters

Solution 	: Full code can be found in END2_Session1.ipynb colab file above

Input dimension  = 2

Output = 1

add image here


***
## Understanding "a neural network neuron"

A neural network neuron is similar to our brain neuron but not same ( brain neuron can store as well as compute but neural network neuron can just store temporary data )
A neural network neuron has input connections ( similar to brain neuron has dentrides ) and output connections ( similar to axon ).


It is basic building block for neural network. One or more neuron stack up to form a hidden layer. 
A neural network can have one or more hidden layers.

#### A neural network with two inputs and a single neuron looks like -

![plot](./images_readme/neuralnet_aneuron.jpg)

Here x1 and x2 are two inputs and while training the neural network we try to optimise w1 & w2 weights associated with incoming connections (plus a bias b, similar to intercept in ML).

Output 

y = tanh ( w1 * x1 + w2 * x2 + b )

tanh is the activation function. Other examples of activation functions are sigmoid, ReLU,  Leaky ReLU etc.

##### Why activation function?
In the absence of activation function output y could result in any value between -inf to +inf and when we have multiple hidden layers this may explode. 
So we use activation functions as bounding the output upstream say in [0,1]

https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/

***
## Understanding "Learning rate"

While training any neural network we use a configurable hyperparameter "Learning rate". 
This learning rate is the step size determining the extend of change in weights in hidden layers in each iteration ( backpropagation ) during training.

Very high values of learning rate fails the network to converge to minima thus yielding larger errors on each iteration.
![plot](./images_readme/learning_rate.gif)

***
## Understanding "weights initialization"


***
## Understanding "Loss" in neural network



***
## Understanding "chain rule" in gradient flow
