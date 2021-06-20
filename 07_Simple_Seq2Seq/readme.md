#### Session 7
***
## Simple Sequence to Sequence Modeling

### Task

Starter code  [Colab](https://colab.research.google.com/drive/1wxfX9cmtuo1mz5uYQyQyVUhQyB5Cih_A?usp=sharing "Google Colab")

Train Model on following datasets

1. http://www.cs.cmu.edu/~ark/QA-data/

2. https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs


Source datasets (https://kili-technology.com/blog/chatbot-training-datasets/)

### Understanding Data & pre-processing

For any text dataset 

1. Tokenize the text : Convert sentence into list of tokens. **Tokens are not words** e.g. '!', "'s"

spaCy is mostly used package to convert sentence to tokens.

Below function takes in a sentence and gives out list of tokens.

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

2. Set Field from torchtext.legacy.data - basic proccessing for each record in text dataset. 
	
	a. Which tokenizer to use - here we using spacy,
	
	b. Calulating len of list - to be used for padding and sorting
	
	c. Convert all elements of list into lowercase 
	
	d. Prefix / Post fix  - to keep track of start ( initial token ) and end of sequence.
	
SRC = Field(tokenize = tokenize_en,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True,
            include_lengths=True)
			

3. Convert from list to torchtext dataset

example = [Example.fromlist([str(df.question1[i]), str(df.question2[i])], fields) for i in range(df.shape[0])]

quoraDataset = Dataset(example, fields)

4. Split into Train & Test

(train, test) = quoraDataset.split(split_ratio=[0.70, 0.30], random_state=random.seed(SEED))

5. Build the vocabulary

SRC.build_vocab(train, min_freq = 2)
TRG.build_vocab(train, min_freq = 2)

6. Create an iterator 

BATCH_SIZE = 64

train_iterator, test_iterator = BucketIterator.splits(
    (train, test), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x: len(x.question1),
    sort_within_batch=True, 
    device = device)
	
7. Save the input text vocabulary to used for predictions from model

import os, pickle
with open(F"./gdrive/MyDrive/NLP/tokenizer.pkl", 'wb') as tokens:
  pickle.dump( SRC.vocab.stoi, tokens)
  

### Understanding - Encoder

### Understanding - Decoder

### Understanding - Seq2Seq


### Understanding - Optimizer and Loss


### Understanding - Train

### Understanding - Evaluate

### Understanding - Training

### Final Model Performance Metrics

### Few Predictions and Comparison


