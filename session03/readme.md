#### Session 3 
***
## Custom NN 
### 2 Inputs :
##### - 1 image i/p from MNIST
##### - 1 random number (0-9)

### 2 Outputs :
##### - the "number" that was represented by the MNIST image ( Classification )
##### - the "sum" of MNIST number with the random number that was generated and sent as the input to the network

## Proposed Solution 
***

![plot](./images/network.JPG)

MNIST images shape is a 4D tensor of shape(samples, channels, height, width). As MNIST images are grayscale image number of the channel will be 1.

C = 1

H=28

W=28

Batch = 100 ( we took here )


**First i/p tensor** [100,1,28,28] from image per batch is fed to CONV Layer 1 


Flatten Output from CONV Layer 2
> 12x4x4 = 192

**Second input** is one-hot encoding for digits 0-9 
we randomly generated , ....

>   digits = torch.randint(0, 10, (batch_size,)) # batch_size rand ints

>   digits_one = F.one_hot(digits, num_classes=10) # one hot encoding

Adding 10 
> 192 + 10 = 202

You can see in network below -

![plot](./images/params.JPG)


#### Combine the two inputs

As you can see we have combined **random digit** input to **first FC Layer**. We do so by **concatenating 
the flattened output tensor from CONV Layer 2**

> t = torch.cat((t, d), dim=1)

#### Understanding CNN i/p (square) & o/p 

![plot](./images/ip_op.JPG)

#### Loss function
Losses in output layer is evaluated using **Cross Entropy** ( for both outputs )

> Output 1 : 10 unique classes (0-9)

> Output 2 : 19 unique classes (0-19, as min 0+0 = 0 & max 9+9=18 )

For muli-class ( say unique classes = 10 for 0-9 image classes )

**Loss is calculated using separately for each class (one class or rest all classes) label per observation and sum the result.**

![plot](./images/lossfunction.JPG)

*where,*

M : number of classes (0,1,2,3....9)

log : natural log

y : binary indicator (0 or 1) if class label c is the correct classification for observation o

p : predicted probability observation o is of class c


#### Accuracy

We must check Accuracy ( correctly predicted / total observation ) for both outputs 

- MNIST
- ADDER


