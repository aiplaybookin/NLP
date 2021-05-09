#### Session 2
***
## 🏋️‍♀️ Playing Backpropagation in Excel 🤽‍♀️🏟

_Objective_

To understand backpropagation and chain rule in Neural Networks

Let's follow the below architecture -

![plot](./images/NN_Architecture.JPG)

i1, i2 = inputs

h1 = w1*i1 + w2*i2
h2 = w3*i1 + w4*i2

a_h1 = σ(h1) = 1/(1+exp(-h1))
a_h2 = σ(h2)

o1 = w5*a_h1 + w6*a_h2
o2 = w7*a_h1 + w8*a_h2

a_o1 = σ(o1)
a_o2 = σ(o2)

E1 = ½ * ( t1 - a_o1)²
E2 = ½ * ( t2 - a_o2)²
E_Total = E1 + E2
