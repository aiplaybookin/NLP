#### Session 4
***
## RNN & LSTM

Recurrent Neural Network
 - speech recognition ( very difficult ; need additional work to determine discrete words )
 - language translation
 - stock predictions
 - image annotation ( interesting )


RNN models are limited to **short term memory.** Because of backprop - > weight learning is poor 
in initial layers as at any layer learning happens for its own layer and for 1 layer before ( because presence of 
hidden input ) thus resulting in vaninsing gradient as we move backwards.

##### Modified RNN
 - LSTM ( Long Short memory )
 - GRU ( Gated Recurring Units )
 
