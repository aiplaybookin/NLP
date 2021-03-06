#### Session 2
***
## 🏋️‍♀️ Backpropagation in Excel 🤽‍♀️🏟

**Objective**

To understand backpropagation and chain rule in Neural Networks

Let's follow the below NN architecture -

![plot](./images/NN_Architecture.JPG)

Here we have **2 inputs (i1 & i2), one hidden layer (in green) and one output layer ( in maroon)**.


All connections are labelled and assigned with intial weights as shown in figure above.

Assume,
> There are no bias terms, just for the sake of understanding backprop.

> Learning rate η= 0.5 

Assume below are **true expected outputs t1 & t2. i1 & i2 are inpt values.**
> Just for understanding purpose we will use same inputs and same outputs 

| Output t1       | Output t2     | i1     | i2     |
| :-------------- | :-----------: | -----: |------: |
|  0.01           | 0.99          | 0.05   | 0.1    | 


**Light and dark color circles (green/ maroon) are neurons without activation and with activation repectively.** 
Here for simplicity we are using sigmoid function as activation function.

Now from basics of neural network we can write below equations,

For hidden layer ( incoming connections w1, w2, w3 & w4 and inputs i1 & i2 ) -

![plot](./images/inputH1.JPG) ![plot](./images/hiddenlayer1_eq.JPG) 

For output layer ( incoming connections w5, w6, w7 & w8 ) - 

![plot](./images/inputH2.JPG) ![plot](./images/outputlayer_eq.JPG)

As we have two outputs ( assuming T1 and T2 as actual truth outputs), there will be errors in both, 
lets call E1 & E2. Hence -

![plot](./images/inputH3.JPG) ![plot](./images/totalerror_eq.JPG)

With above basic equations in hand we need to update the all weights in each iteration in such a way that
**will minimize the error E_total** via backpropagation.

We know -

![plot](./images/updateWeights.png)

In next subsequent steps we will see how to compute **δE_total wrt δwx ( here x = 1,2,3....8; denoting w1, w2, ...w8)**

**δE_total/δw5**
> δE_total/δw5 = δ(E1 +E2)/δw5

*Because wrt w5 there is no impact from E2 ( constant), so removing E2 and later applying chain ruleand expanding.*

> δ(E1 +E2)/δw5 = δ(E1)/δw5 = (δE1/δa_o1) * (δa_o1/δo1) * (δo1/δw5)

*Computing separately*

>> δE1/δa_o1 = δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = a_o1 - t1

*We substituted E1 = ½ * ( t1 - a_o1)² & then did partial derivative*

>> δa_o1/δo1 = δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1 * (1 - a_o1 )

*Substituted a_o1 = σ(o1) and took partial derivative of sigmoid : derivative of sigmoid is sigmoid * (1-sigmoid)*

>> δo1/δw5 = a_h1

*Substituted o1 = w5 * a_h1 + w6 * a_h2 and took derivative wrt δw5*

Finally,

**δE_total/δw5 = (a_o1 - t1 ) * a_o1 * (1 - a_o1 ) * a_h1**

By seeing above equation, we know that δE_total/δw5 depends on a_o1, t1, and a_h1. 
In similar way we can also write for w6 (as w6 is taking input from a_h2 )

**δE_total/δw6**

> δE_total/δw6 = (a_o1 - t1 ) * a_o1 * (1 - a_o1 ) * **a_h2**

Similarly for w7, 

**δE_total/δw7**

> δE_total/δw7 = (a_o2 - t2 ) * a_o2 * (1 - a_o2 ) * a_h1

and for w8,

**δE_total/δw8**

> δE_total/δw8 = (a_o2 - t2 ) * a_o2 * (1 - a_o2 ) * a_h2


Relationship of E_total wrt w1:

**δE_total/δw1**

> δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1

> δE_total/δw1 = δ(E_total)/δa_h1 * δa_h1/δh1 * δh1/δw1

*Substituting a_h1 = σ(h1) = 1/(1+exp(-h1)) and then taking derivative wrt δh1*

**δE_total/δw1 = δ(E_total)/δa_h1 * a_h1 * (1- a_h1) * i1**

Similarly,

**δE_total/δw2 = δ(E_total)/δa_h1 * a_h1 * (1- a_h1) * i2**

**δE_total/δw3 = δ(E_total)/δa_h2 * a_h2 * (1- a_h2) * i1**

**δE_total/δw4 = δ(E_total)/δa_h2 * a_h2 * (1- a_h2) * i2**

Expanding δE_total/δa_h1 to simplify further -

>>> δE_total/δa_h1 = δ(E1+E2)/δa_h1 

>>>> δE1/δa_h1 = δE1/δa_o1 * δa_o1/δo1 * δo1/δa_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5

>>>> δE2/δa_h1 = δE1/δa_o2 * δa_o2/δo2 * δo2/δa_h1 = (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7

Similarly for w3 & w4, expanding δE_total/δa_h2

>>> δE_total/δa_h2 = δ(E1+E2)/δa_h2 

>>>> δE1/δa_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6

>>>> δE2/δa_h2 = (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8



*Finally by substituting above δ(E_total)/δa_h1 for δE_total/δw1*

**δE_total/δw1 =( (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7 )  * a_h1 * (1- a_h1) * i1**



### This way we have simplified the δE_total wrt δwx and with each iteration, we can update the weights (w1, w2, ....w8) untill the change in error is flat.

***
## Output from Excel Example

![plot](./images/output1.JPG) 

![plot](./images/output2.JPG) 

![plot](./images/output3.JPG) 

### Running for learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]

![plot](./images/learningrate.JPG) 