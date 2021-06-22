#### Session 7
***
## Simple Sequence to Sequence Modeling

### Task

Train Model on following datasets

1. http://www.cs.cmu.edu/~ark/QA-data/

A corpus of Wikipedia articles, manually-generated factoid questions from them, and 
manually-generated answers to these questions, for use in academic research.

2. https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

A set of Quora questions to determine whether pairs of question texts actually correspond 
to semantically equivalent queries. More than 400,000 lines of potential questions duplicate 
question pairs.

[Dataset source](https://kili-technology.com/blog/chatbot-training-datasets/)

### Data pre-processing

For any text dataset 

1. Tokenize the text : Convert sentence into list of tokens. **Tokens are not words** e.g. '!', "'s"

spaCy is mostly used package to convert sentence to tokens.

Below function takes in a sentence and gives out list of tokens.
```
def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

2. Set Field from torchtext.legacy.data - basic proccessing for each record in text dataset. 
	
	a. Which tokenizer to use - here we using spacy,
	
	b. Calulating len of list - to be used for padding and sorting
	
	c. Convert all elements of list into lowercase 
	
	d. Prefix / Post fix  - to keep track of start ( initial token ) and end of sequence.

```	
SRC = Field(tokenize = tokenize_en,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            include_lengths=True)
			
```

3. Convert from list to torchtext dataset

```
example = [Example.fromlist([str(df.question1[i]), str(df.question2[i])], fields) for i in range(df.shape[0])]

quoraDataset = Dataset(example, fields)

```

4. Split into Train & Test

```
(train, test) = quoraDataset.split(split_ratio=[0.70, 0.30], random_state=random.seed(SEED))
```

5. Build the vocabulary - **Vocab should be build using train dataset** only. If we use test or
validation dataset, there will be possibility of data leaks.

min_freq used to retrict tokens occuring more than twice else map to <unk>.

```
SRC.build_vocab(train, min_freq = 2)
TRG.build_vocab(train, min_freq = 2)
```

6. Create an iterator 

```
BATCH_SIZE = 64

train_iterator, test_iterator = BucketIterator.splits(
    (train, test), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: len(x.question1),
    sort_within_batch=True, 
    device = device)
```
	
7. Save the input text vocabulary to used for predictions from model

```
import os, pickle
with open(F"./gdrive/MyDrive/NLP/tokenizer.pkl", 'wb') as tokens:
  pickle.dump( SRC.vocab.stoi, tokens)
```  

### Understanding - Encoder

1. Takes in indexed SRC tokens

2. Does Embedding ( all senetences in batch at once )

3. Pass to LSTM 

4. Returns last Hidden & Cell state outputs (This makes the context vector)

Note : Dropout is applied between layers of multi-layer RNN 

### Understanding - Decoder

1. Takes in single indexed token at a time 

2. Unsqueeze to create a additional dimension (for batch)

3. Create embeddings

4. Pass to LSTM ( use hidden and cell state from ENCODER output to initialise here )

5. Pass the output from previous step 4 to linear FC layer


### Understanding - Seq2Seq Class

We create a additional class Seq2Seq which will help combine encoder and decoder, act as a wrapper.

NOTE : Make sure number of layers, hidden and cell states dimensions are same for both encoder and decoder.
Otherwise handle it with additional tricks ;)

1. Takes input sentences

2. Pass to encoder -> spits out last hidden & cell

3. First input decoder for each sentence is passed as <sos> token

4. Loop till length of target senetence minus 1 ( number of target tokens ) is reached.
Minus 1 because last token is always <eos> in target list of tokens and we do not have any outcome if this
is input to decoder.

	a. Pass input, previous hidden and cell state to decoder

	b. Take output, hidden and cell state from decoder and stack the output in list
	
	c. Use ```Teacher ratio (0 to 1): Random choice of cheating i.e. pass actual target truth or
	the previous output from decoder.```
	
	d. go to step a with chosen input (output from decoder or actual truth ), hidden and cell state. Repeat.
	


NOTE : HID_DIM and N_LAYERS are same for ENcoder and Decoder - easier to pass info.

```
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
```

### Understanding - Loss

We have to remove paddings before we calculate loss, we do so by getting all padded index and pass it 
in CrossEntropyLoss specifying to ignore. 

```
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
```

NOTE: Loss is calcuated using avg of token by token comparison.

[More on Optimizers](https://ruder.io/optimizing-gradient-descent/)

### Understanding - Train

1. Iterate and get a batch 
	
	a. Get SRC and TRG tokens for batch 
	
	b. Make the gradient zero
	
	c. Pass the SRC and TRG tokens to seq2seq model and get output
	
	d. Flatten the output
	
	e. Remove the first item from flattened output ( remember first input tojen was <sos> so it's 
	coresponding output would be 0 and we ignore so that it doesn't add to losses.
	
	f. Compute loss using backward prop
	
	g. ```Clip``` prevents gradient exploding
	
	```
	torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
	```
	
	h. Add the loss
	
2. Return avg loss 

### Understanding - Evaluate

Similar to train except 

1. Use ```model.eval()``` : This prevents from using dropouts 

2. Use ```torch.no_grad()``` : We don't want to compute gradients for backprop

3. Do not use ```Teacher param``` : Turn off using 0

```
output = model(src, trg, 0)
```


### Training Logs

### Final Model Performance Metrics

### Few Predictions and Comparison

### Points to Ponder upon

- Same vocab for SRC & TRG provided both are in same language.



