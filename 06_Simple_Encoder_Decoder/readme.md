#### Session 6
***
## Simple Encoder & Decoder 

### Task

1. Encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. VERY IMPORTANT TO MAKE THIS SINGLE VECTOR


2. This single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell.

3. And send this final vector to a Linear Layer and make the final prediction. 

This is how it should look:

	embedding

	word from a sentence +last hidden vector -> encoder -> single vector

	single vector + last hidden vector -> decoder -> single vector

	single vector -> FC layer -> Prediction


### Solution

Peek into Data set 
![plot](./images/dataset_head.JPG)

3 Class - ( 0 : Negative, 1 : Positive, 2 : Neutral )
![plot](./images/dataset_target.JPG)

#### Tokenization
![plot](./images/tokenize_eg.JPG)

High level Architecture
![plot](./images/highlevel_arch.JPG)