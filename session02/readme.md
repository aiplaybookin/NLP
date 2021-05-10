#### Session 2
***
## 🏋️‍♀️ Playing Backpropagation in Excel 🤽‍♀️🏟

_Objective_

To understand backpropagation and chain rule in Neural Networks

Let's follow the below NN architecture -

![plot](./images/NN_Architecture.JPG)

Here we have *2 inputs (i1 & i2), one hidden layer (in green) and one output layer ( in maroon)*.


All connections are labelled and assigned with intial weights as shown in figure above.
> Assume there are no bias terms, just for the sake of understanding backprop.

*Light and dark color (green/ maroon) are parameters without activation and with activation repectively.* 
Here for simplicity we are using sigmoid function as activation function.

Now from basics of neural network we can write below equations,

For hidden layer ( incoming connections w1, w2, w3 & w4 ) -

![plot](./images/hiddenlayer1_eq.JPG)

For output layer ( incoming connections w5, w6, w7 & w8 ) - 

![plot](./images/outputlayer_eq.JPG)

Now total *error* is error from T1 and T2, lets call E1 & E2, so -

![plot](./images/totalerror_eq.JPG)
