#### Session 2
***
## 🏋️‍♀️ Playing Backpropagation in Excel 🤽‍♀️🏟

** Objective **

To understand backpropagation and chain rule in Neural Networks

Let's follow the below NN architecture -

![plot](./images/NN_Architecture.JPG)

Here we have ** 2 inputs (i1 & i2), one hidden layer (in green) and one output layer ( in maroon) **.


All connections are labelled and assigned with intial weights as shown in figure above.
> Assume there are no bias terms, just for the sake of understanding backprop.

**Light and dark color circles (green/ maroon) are neurons without activation and with activation repectively.** 
Here for simplicity we are using sigmoid function as activation function.

Now from basics of neural network we can write below equations,

For hidden layer ( incoming connections w1, w2, w3 & w4 and inputs i1 & i2 ) -

![plot](./images/hiddenlayer1_eq.JPG) ![plot](./images/inputH1.JPG)

For output layer ( incoming connections w5, w6, w7 & w8 ) - 

![plot](./images/outputlayer_eq.JPG)

Now total *error* is error from T1 and T2, lets call E1 & E2, so -

![plot](./images/totalerror_eq.JPG)



* Additional notes : Activation Functions *
 
It also performs a nonlinear transformation on the input to get better results on a complex neural network.
Activation function also helps to normalize the output of any input in the range between 1 to -1. Thus reduce the computation time because the neural network sometimes trained on millions of data points.

The sigmoid function causes a problem mainly termed as vanishing gradient problem which occurs because we convert large input in between the range of 0 to 1 and therefore their derivatives become much smaller which does not give satisfactory output.

