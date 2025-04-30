# Building a Transformer from Scratch: A Step-by-Step Guide
## Input Embeddings
* Input embeddings can convert string data into vectors (in this case with 512 dimensions)
* First, convert the sentence into input IDs, which are numbers representing the position of each word in the vocabulary. Each number is then mapped to an embedding.
```
import torch
import torch.nn as nn (nn  is the building blocks for making a neural network)
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```
* In the code above, the **d_model** parameter determines the dimensionality of the embeddings (size of each embedding vector)
* **vocab_size** determines size of the vocabulary (the total number of unique tokens).
* **self.d_model** attribute stores the dimensionality of the embeddings.
* **self.vocab_size** stores the vocabulary size
* **self.embedding** initializes an embedding layer using nn.Embedding, which maps each token in the vocabulary to a d_model-dimensional vector.
* The **forward** method defines the forward pass of a module, which is the computation that the module performs on input data.
* The **x** parameter is a tensor containing token indices (token IDs from a tokenized input sequence)
* **self.embedding(x)** looks up the embedding vectors for the input token indices x using the embedding layer. This results in a tensor of shape (batch_size, sequence_length, d_model)
* Multiplying by **math.sqrt(self.d_model)** scales the embedding vectors by the square root of d_model. This scaling is a common practice in Transformer models, as it helps stabilize the gradients during training. The idea is that the variance of the embeddings is preserved when they are added to other vectors (like positional encodings)
## Positional Encoding
* Positional encoding conveys positional information of each word within a sentence to the model. Another vector, of the same size as the embedding, is added, which contains special values that inform the model about the position of each word in the sentence.
* Two formulas from the original Transformers paper can be used to create this vector. The first formula is applied to even positions, and the second is applied to odd positions.
```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)
        
        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1) # (seq, 1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq, d_model)
        return self.dropout(x)
```
* The parameters for the above block are **d_model**, **seq** (the size of the vocabulary/total number of unique tokens) and **Dropout** (rate of regularization, reduce overfitting)
* **self.d_model** stores the dimension of the input embeddings
* **self.seq** stores maximum length of the input sequences
* **self.dropout** initializes a dropout layer with the given rate
### Positional Encoding Matrix
* **pe** is a zero tensor of shape (seq, d_model), which holds positional encodings
* **position** is a tensor containing the position indices from 0 to seq-1, reshaped to (seq, 1)
* **div_term** is a tensor used for scaling the positions, calculated as 10,000 raised to the power of (2i/d_model) where i is the index of the embedding dimension
### Applying Formulas from the Original Paper
* **pe[:, 0::2]** assigns the sine of the scaled positions to the even indices of the positional encoding matrix
* **pe[:, 1::2]** assigns the cosine of the scaled positions to the odd indices of the positional encoding matrix
### Batch Dimension
* **pe = pe.unsqueeze(0)** adds an extra dimension at the beginning to accomodate batch processing, resulting in shape (1, seq, d_model)
### Register Buffer
* **self.register_buffer('pe', pe)** is not considered as a model parameter, but will be part of the model state and moved to the appropraite device when the model is moved.
### Forward Pass (Add Positional Encoding)
* **x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)** adds the positional encodings to the input embeddings x. The requires_grad_(False) ensures positional encodings aren't updated during backpropagation.
### Apply Dropout
* **return self.dropout(x)** applies dropout to the combined embeddings and positional encodings to prevent overfitting.
* Source: https://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a
