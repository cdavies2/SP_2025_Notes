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

## Add Norm (Layer Normalization)
```
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha (multiplicative) is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias (additive) is a learnable parameter

    def forward(self, x):
        # x: (batch, seq, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```
* **self.eps** stores the epsilon value to prevent numerical instability during division
* **self.alpha** is a learnable scale parameter initialized to ones, of shape (features,). It scales normalized output
* **self.bias** is a learnable shift parameter initialized to zeroes, of shape (features,). It shifts normalized output
### Compute Mean
* **mean = x.mean(dim=-1, keepdim=True)** calculate the mean across the last dimension (hidden_size) for each data point in the batch and sequence. The result is a tensor of shape (batch, seq, 1) keeping the dimension for broadcasting.
### Compute Standard Deviation
* **std=x.std(dim=-1, keepdim=True)** calculates the standard deviation across the last dimension (hidden_size) for each data point in the batch and sequence. The result is a tensor of shape (batch, seq, 1) keeping the dimension for broadcasting.
### Normalize
* **(x-mean)/(std + self.eps)** normalizes the input x by subtracting the mean and dividing by the standard deviation plus a small epsilon to avoid division by zero
### Scale and Shift
* **self.alpha * (x - mean)/(std + self.eps) + self.bias** scales the normalized output by self.alpha and shifts it by self.bias. Both self.alpha and self.bias are learnable parameters that allow the model to scale and shift the normalized values apporpriately.

## Feed Forward Layer
* Two matrices are used here. The first matrix has dimensions from 512 to 2048 and the second has dimensions from 2048 to 512.
```
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq, d_model) --> (batch, seq, d_ff) --> (batch, seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```
* **self.linear_1** is a linear layer (fully connected layer) that maps from d_model to d_ff. This is the first linear transformation applied to the input.
* **self.linear_2** is a linear layer that maps from d_ff back to d_model. This is the second linear transformation applied to the intermediate representation.
### Linear Transformation and Activation
* **self.linear_1(x)** applies the first linear transformation, mapping the input from d_model to d_ff
* **torch.relu(self.linear_1(x))** applies the ReLU activation function to introduce non-linearity. The ReLU function sets all negative values to zero and keeps positive values unchanged.
### Second Linear Transformation
* **self.linear_2(...)** applies the second linear transformation, mapping the intermediate representation from d_ff back to d_model
* **return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))** returns the final output tensor of shape (batch, seq, d_model).

## Multi-Head Attention
```
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq, seq) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq, seq) --> (batch, h, seq, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)  
        return self.w_o(x)
```
* **self.d_model** stores the dimension of the embedding vectors
* **self.h** stores the number of attention heads
* **self.d_k** is the dimension of the vectors processed by each head, calculated as d_model // h. This ensures the embedding dimension is evenly split across the heads
* **self.w_q, self.w_k, self.w_v, self.w_o** are linear layers that project the input vectors to queries, keys, values, and outputs, respectively
### Attention Calculation (attention static method) Parameters
* **query, key, value** are the input vectors for the attention mechanism
* **mask** is an optional mask tensor to prevent attention to certain positions.
### Steps
* **Compute Attention Scores**: scores are computed using the dot product of the query and key tensors, scaled by the square root of d_k to maintain stable gradients. This is done as **(query @ key.transpose(-2, -1)/math.sqrt(d_k)
* **Apply Mask**: if a mask is provided, positions where the mask is zero are set to a very low value (effectively -inf) to ensure they don't affect the attention calculation
* **Compute Attention Output**: the attention output is obtained by multiplying the normalized attention scores with the value tensor
* **Return**: the method returns the attention output and the attention scores for possible visualization
* **Reshape and Split Heads**: projected tensors are reshaped and transposed to split the embedding dimension into multiple heads. The shape changes from (batch, seq, d_model) to (batch, seq, h, d_k) and then transposed to (batch, h, seq, d_k)
* **Calculate Attention**: the reshaped queries, keys, and values are passed to the attention method to compute the attention output and scores
* **Combine Heads**: the attention outputs from all heads are concaternated and reshaped back to the original embedding dimension. The shape changes from (batch, h, seq, d_k) to (batch, seq, h, d_k) and finally to (batch, seq, d_model).
* **Final Linear Transformation**: the combined output is projected back to the original embedding dimension using the w_o linear layer.

## Residual Connection
```
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
```
* **sublayer** is a callable sub-layer (multi-head attention or feedforward network) to be applied to the normalized input
### Steps
* **Apply Sublayer**: the normalized input is then passed to the sub-layer (EX: multi-head attention for feedforward network) specified by sublayer. This call to sublayer (self.norm(x)) applies the sub-layer's transformation to the normalized input.
* **Add Residual Connection**: the original input tensor x is added to the output of the dropout layer. This addition operation implements the residual connection, which allows the network to learn identity mappings and mitigates the vanishing gradient problemin deep networks. The resulting tensor is the output of the residual connection block

## The Encoder
```
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
```
* **self.self_attention_block** stores the multi-head self-attention block
* **self.feed_forward_block** stores the feed-forward network block
* **self.residual_connections** is a list of two ResidualConnection instances, one for the self-attention sub-layer and one for the feed-forward sub-layer
### Feed Forward Steps
* **Self-Attention with Residual Connection**: input tensor x is first normalized and passed through the self-attention block (self.self_attention_block) with a residual connection around it. This is done using the first ResidualConnection in the list: _self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))_. The lambda function is used to pass the normalized input to the self-attention block
* **Feed-Forward with Residual Connection**: the output from the self-attention sub-layer is then normalized and passed through the feed-forward block _(self.feed_forward_block)_, with another residual connection around it. This is done using the second ResidualConnection in the list: _self.residual_connections[1](x, self.feed_forward_block).
* The method returns the output tensor after passing through both the self-attention and feed-forward sub-layers with residual connections and normalization.
```
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```
### Feed Forward Steps
* **Pass Through Encoder Layers**: the input tensor x is sequentially passed through each layer in the self.layers list. Each layer processes the input tensor and updates it. This is done using the loop _for layer in self.layers: x=layer(x, mask)_
* **Final Layer Normalization**: the output tensor from the last encoder layer is normalized using self.norm(x)
* The method returns the normalized output tensor after processing through the stack of encoder layers.

## The Decoder
```
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
```
* **self_attention_block** is an instance of the MultiHeadAttentionBlock class for self-attention within the target sequence.
* **cross_attention_block** is an instance of the MultiHeadAttentionBlock class for attention over the encoder's output (cross-attention)
* **feed_forward_block** is an instance of a feed-forward network (EX: FeedForwardBlock)
* **self.self_attention_block** stores the self-attention block
* **self.cross_attention_block** stores the cross-attention block
* **self.feed_forward_block** stores the feed-forward network block
* **self.residual_connections**: list of three ResidualConnection instances, one for each sub-layer (self-attention, cross-attention, and feed-forward)
### Feed Forward Parameters
* **src_mask** is a mask tensor for the source sequence to prevent attention to certain positions
* **tgt_mask** is a mask tensor for the target sequence to prevent attention to future positions (during training) and certain positions
### Steps
* **Cross-Attention with Residual Connection**: the output from the self-attention sub-layer is normalized and passed through the cross-attention block with another residual connection around it. This is done using the second ResidualConnection in the list: _self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))_. The cross-attention mechanism allows each position in the target sequence to attend to all positions in the source sequence (encoder output), providing the necessary context for generating the target sequence
* The method returns the output tensor after processing through the self-attention, cross-attention, and feed-forward sub-layers with residual connections and normalization.
```
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
```
### Forward Pass
* The forward method processes the input tensor x through each DecoderBlock in the self.layers list
* Each DecoderBlock layer performs self-attention, cross-attention, and feed-forward operations, with residual connections and layer normalization applied to each sub-layer
* The *src_mask* ensures that the attention mechanism only attends to valid positions in the course sequence.
* The *tgt_mask* prevents the decoder from attending to future positions in the target sequence during training, maintaining the autoregressive nature of the model.
* The final output tensor is normalized using the LayerNormalization instance (self.norm)

## Linear Layer
* The ProjectionLayer is responsible for mapping the output of the decoder to the vocabulary space, which allows the model to generate or predict the next word in the sequence.
```
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq, d_model) --> (batch, seq, vocab_size)
        return self.proj(x)
```
* **self.proj** is an instance of nn.Linear, a fully connected linear layer that performs the projection from the model's output dimension (d_model) to the vocabulary size (vocab_size)
* The forward method takes the input tensor x from the decoder, which has a shape of (batch, seq, d_model)
* The input tensor is passed through the linear layer *self.proj*
* The linear layer computes the dot product between the input tensor and its weight matrix, and adds the bias (if applicable), transforming the input tensor from the model's feature space to the vocabulary space.
* The output tensor now has a shape of *(batch, seq, vocab_size)*, where each position in the sequence is associated with a vector of logits for each word in the vocabulary.

## Transformer
```
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq, vocab_size)
        return self.projection_layer(x)
```
* **Initialization**: the __ init __ method initializes the Transformer class with instances of the encoder, decoder, embedding layers, positional encoding layers, and the projection layer. These components are needed to transform input sequences through the encoding and decoding processes.
* **Encoding**: the encode method handles the source input sequence. It first converts the input tokens into embeddings and then adds positional encodings to retain the order of the tokens. The combined embeddings are processed by the encoder to produce the encoded representation of the source sequence
* **Decoding**: the decode method handles the target input sequence. It converts the target tokens into embeddings, adds positional encodings, and processes the combined embeddings along with the encoder's output through the decoder. The decoder output is the decoded representation, which is used for generating the final predictions.
* **Projection**: the project method maps the decoder's output to the vocabulary space. This step helps translate continuous decoder output into discrete token predictions.
* A `build_transformer` method is in the article
* Source: https://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a
