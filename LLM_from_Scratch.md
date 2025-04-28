# Let's Build GPT: from scratch, in code, spelled out.
* This sample model will use Tiny Shakespeare as its dataset (a concatenation of all of the works of Shakespeare).

* Source: https://www.youtube.com/watch?v=kCc8FmEb1nY

# What are transformers in Generative AI?
* Transformers allow large language models to track relationships between chunks of data, derive meaning, and generate/predict new content mimicking the original dataset.
* Neural networks are foundational to Generative AI, taking in data via layers of interconnected nodes (neurons), which then process and decipher patterns in it, using that information to make predictions or decisions.
* Three popular techniques for implementing Generative AI include...
  * Generative Adversarial Networks (GANs)
  * Variational Autoencoders (VAEs)
  * Transformers
## What are Generative Adversarial Networks? (GAN)
* These models have two main components....
  * A generator tries to produce data
  * A discriminator evaluates data (improving the generator's ability to produce convincing data).
* GANs have many limitations; they can be difficult to train and often generate the same output repeatly (while something like ChatGPT produces different responses to the same prompt quite often).
## What are Variational Autoencoders (VAEs)
* Variational Autoencoders (VAEs) are a generative model used for unsupervised machine learning and have three main components....
  * Encoder: acts as a scanner, gathering input
  * Decoder: reconstructs input
  * Loss Function: ensures the result mirrors the original inputs while also containing unique differences.
* VAEs have many limitations; the loss function can be overly complex, and striking the right balance between making generated content look real (reconstruction) and ensuring it's structured correctly (regularization) can be challenging.
## How Transformers are different from GANs and VAEs
* Transformer models perform all parts of a sequence simultaneously (making them more efficient and GPU-friendly, and they understand sentence context.
## How does the Transformer architecture work?
* A transformer is an architecture of neural networks that takes in a text sequence as input and produces another text sequence as output.
* Input is a series of tokens (chunks of text with meaning) from the provided text.
* Once input is received, the sequence is converted to numerical vectors, called embeddings, that capture the context of each token. Embeddings allow models to process textual data mathematically and understand the intricate details/relationship of language (EX: "Good" and "great" are classified similarly as they both have positive sentiment and are often used as adjectives).
* Positional embeddings help a model understand the position of a token within a sequence (EX: "hot dog" means something different from "dog hot"
* The encoder processes and prepares input data by understanding its structure and nuances. It contains two mechanisms...
 * _Self-attention_: relates every word in the input sequence to every other word (giving them a score to indicating how much attention should be paid to it)
 * _Feed-forward_: fine-tuner, takes scores from the self-attention process and further refines the understanding of each word, ensuring nuances are captured and the learning process is optimized.
* The decoder uses previous outputs (the output embeddings from the previous time step of the decoder and processed input from the encoder), taking into account original data and what has been produced so far to create appropriate output.
* The output is a new sequence of tokens.
* The Generative Pre-Trained Transformer (GPT) is a model built using the Transformer architecture. Transformers are the foundation of tools like ChatGPT.
* Source: https://www.pluralsight.com/resources/blog/ai-and-data/what-are-transformers-generative-ai

# Attention is All You Need
* Recurrent models typically factor computation along symbol positions of input and output sequences, generating parallel hidden states.
* Attention mechanisms allow modeling of dependencies without regard to their distance in the input or output sequences, but such mechanisms were often used in conjunction with a recurrent network.
* The Transformer model architecture eschews recurrence and relies entirely on the attention mechanism, resulting in more parallelization and higher translation quality.
## Background
* In models like the Extended Neural GPU and ByteNet, the number of operations needed to relate signals from two arbitrary input/output position grows in the distance between positions, making it difficult to learn dependencies between far distances.
* In the Transformer, this is reduced to a constant number of operations.
* Self-attention (or intra-attention) is an attention mechanism relating different positions of a single sequence in order to compute a representation of said sequence. It is useful for summarization and reading comprehension.
* End-to-end memory nutworks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and perform well on question answering tasks.
* The Transformer is the first transduction model relying on self-attention to compute representations of its input/output without a sequence-alligence RNN or convolution.
## Model Architecture
* In a neural network, the encoder maps an input sequence of symbol representations to a sequence of continuous representations, and the decoder generates an output sequence of symbols, one element at a time.
* These models are auto-regressive, consuming previously generated symbols as additional input when generating the next.
* Transformers have fully connected self-attention layers for the encoder and decoder.
### Encoder and Decoder Stacks
* _Encoder_: the encoder is a stack of N=6 identical layers. Each layer has two sublayers, a multi-head self-attention mechanism, and a position-wise fully connected feed-forward network. There is a residual connection around each sub-layer followed by layer normalization (LayerNorm(x+Sublayer(x)). All sub-layers and embedding layers have outputs of dimenison dmodel = 512.
* _Decoder_: also composed of a stack of N=6 identical layers, along with a third sub-layer to erform multi-head attention over the output of the encoder. The self-attention sub-layer is modified to prevent positions from attending to subsequent positions. This, along with the fact that output embeddings are offset by one position, ensures that predictions for position i depend only on outputs from positions less than i.
 ### Attention
 * An attention function involves mapping a query and set of key value pairs to an output, where the query, keys, values, and output are all vectors.
 * _Scaled Dot-Product Attention_: input consists of queries and keys of dimension dk, and values of dimension dv. The dot products of the query with all keys is computed, each is divided by the square root of dk, and a softmax function is used to obtain the weights on the values. The matrix of outputs is computed as...
  * Attention(Q, K, V) = softmax(QK^T/ √dk ) * V
 * _Multi-Head Attention_: instead of performing a single attention function with dmodel-dimensional keys, values and queries, these objects are linearly projected h times with different, learned linear projections to dk, dk, and dv dimensions respectively. On each of these projects, the attention function is performed in parallel, yielding dv-dimensional output values. These are concatenated and projected, resulting in the final values.
  * MultiHead(Q, K, V) = Concat(head1,....,headh)W^O
          where headi = Attention(QWi^Q, KWi^K, VWi^V)
 ### Applications of Attention in our Model
 1. 
* Source: https://arxiv.org/pdf/1706.03762

# Softmax Activation
* The softmax function is often used in the final layer of a neural network model. It converts raw output scores (called logits) into probabilities by taking the exponential of each output and normalizing said values by dividing the sum of all the exponents.
* A softmax ensures output values are between 0 and 1 and add up to 1, making them interpretable as probabilities.
* The mathemical expression for softmax is e to the power of the input to the softmax function for class i divided by the sum of the exponentials of all the raw class scores in the output layer.
           k
 * e^zi / ∑ e^zj
           j=1
* A softmax function converts a vector to a probability distribution by...
  1. _Input_: the function takes a vector z of real numbers, representing the outputs from the final layer of the neural network
  2. _Exponentiation_: each element in z is exponentiated using the mathematical constant e (approximately 2.718), ensuring all values are positive
  3. _Normalization_: exponentiated values are divided by the sum of all exponentiated values, guaranteeing output values sum to 1.
* Some properties of the softmax function include....
 * _Output range_: guarantees output values are between 0 and 1
 * _Sum of probabilities_: sum of all outputs equals 1
 * _Interpretability_: raw output is transformed into probabilities, making predictions easier to understand and analyze.
## Applications of softmax activation
* Softmax is often used for image recognition, Natural Language Processing, and other neural network tasks. Softmax can be implemented in Python using the code below:
```
from math import exp

def softmax(input_vector):
    # Calculate the exponent of each element in the input vector
    exponents = [exp(i) for i in input_vector]

    # Correct: divide the exponent of each value by the sum of the exponents
    # and round off to 3 decimal places
    sum_of_exponents = sum(exponents)
    probabilities = [round(exp(i) / sum_of_exponents, 3) for i in exponents]

    return probabilities

print(softmax([3.2, 1.3, 0.2, 0.8]))
```
* Unlike functions like sigmoid or ReLU (Rectified Linear Unit), which are used in hidden layers for binary classification or non-linear transformations, softmax is uniquely suited for the output layer of multi-class scenarios. Sigmoid doesn't ensure outputs sum to one, and ReLU doesn't provide probabilities.
## Advantages of Using Softmax
* _Probability distribution_: softmax provides a well-defined probability distribution for each class, enabling us to assess a network's confidence in its predictions
* _Interpretability_: probabilities are easier to understand than raw output
* _Numerical stability_: the softmax function has strong numerical stability, making it efficient for training neural networks.
* Source: https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/

# How to Build an LLM from Scratch: A Step-by-Step Guide

## Determine the Use Case for Your LLM
* The use case for your LLM has a major influence on the model's size (complexity can determine number of parameters), training data requirements (more parameters require more training data), and computational resources.
* Key reason for creating your own LLM include....
 * Domain-Specificity: training the LLM with industry-specific data
 * Greater Data Security: incorporating sensitive information without concerns about data storage or usage in open-source or proprietary models
 * Ownership and Control: retaining control over confidential data allows continuous improvement of your LLM as knowledge grows and needs to evolve.
## Create Your Model Architecture
* A neural network is the core engine of your model that determines its capabilities and performance.
* The Transformers architecture is highly recommended for LLMs because it is able to capture underlying patterns and relationships in data, handle long-range dependencies in text, and process input of various lengths efficiently.
* Frameworks like PyTorch from Meta and TensorFlow from Google provide components for creating a model with Transformers architecture.

## Creating the Transformer's Components
* The _Embedding Layer_ converts input into vector representations for efficient processing. This has three main steps...
  1. Tokenization: breaking input into tokens, often sub-word tokens of approximately four characters
  2. Integer Assignment: assign each token an integer ID and save it in a vocabulary dictionary
  3. Vector Conversion: converting each integer into a multi-dimensional vector, each token feature is represented by a vector dimension.
* Transformers have two embedding layers, one in the encoder for input embeddings and one in the decoder for output embeddings.
* A transformer generates _positional encodings_ and adds them to each embedding to track token positions within a sequence, allowing parallel token processing
* _Self-attention mechanism_ is the most crucial part of a transformer, as it compares embeddings to determine their similarity and semantic relevance. It generates a weighted input representation, capturing relationships between tokens to calculate the most likely output.
  * At each self-attention layer, input is projected across several smaller dimensional spaces called heads, each focusing on different aspects of the input in parallel. The original self-attention mechanism has eight heads, but the number can vary based on objective and available resources.
  *  An encoder has one multi-head attention layer, and a decoder has two
* _Feed-Forward Network_ captures higher level features of input to determine complex underlying relationships, and it has three sub-layers
  1. First Linear Layer: projects input onto a higher-dimensional space (EX: 512 to 2048)
  2. Non-Linear Activation Function: introduces non-linearity to help learn more realistic relationships. A common activation function is Rectified Linear Unit (ReLU)
  3. Second Linear Layer: transforms the higher-dimensional representation back to its original dimensionality, compressing additional information while retaining relevant aspects.
* _Normalization layers_ ensure input embeddings fall into a reasonable range. Transformers normalize output for each token at every layer, preserving relationships between token aspects, and not interfering with the self-attention mechanism.
* _Residual Connections_ feed output of one layer directly into the input of another, improving data flow and preventing information loss.
## Assembling the Encoder and Decoder
* _Encoder_ takes the input sequence and converts it into a weighted embedding that the decoder can use to generate input. It is constructed as follows....
  1. _Embedding Layer_: converts input tokens into vector representations
  2. _Positional Encoder_: adds positional information to the embeddings to maintain the order of tokens
  3. _Residual Connection_: feeds into a normalization layer
  4. _Self-Attention Mechanism_: compares each embedding against others to determine their similarity and relevance
  5. _Normalization Layer_: ensures stable training by normalizing the output of the self-attention mechanism
  6. _Residual Connection_: feeds into another normalization layer
  7. _Feed-Forward Network_: captures higher-level features of the input sequence
  8. _Normalization Layer_: ensures output remains within a reasonable range.
* _Decoder_ takes the weighted embedding produced by the encoder to generate output (tokens with highest probability based on input). Some major differences between decoder and encoder architecture include...
  1. _Two Self-Attention Layers_: decoder has an additional self-attention layer
  2. _Two Types of Self-Attention_: masked multi-head attention uses casual masking to prevent comparisons against future token, and encoder-decoder multi-head attention has each output token calculate attention scores against all input tokens better to establish the relationship between input and output. This also employs casual masking to avoid influence from future output tokens.
* Decoder structure is as follows...
  1. _Embedding Layer_: converts the output tokens into vector representations
  2. _Positional Encoder_: adds positional information to the embeddings
  3. _Residual Connection_: feeds into a normalization layer
  4. _Masked Self-Attention Mechanism_: ensures model doesn't see future tokens
  5. _Normalization Layer_: stabilizes output of the masked self-attention mechanism
  6. _Residual Connection_: feeds into another normalization layer
  7. _Encoder-Decoder Self-Attention Mechanism_: establishes relationships between input and output tokens
  8. _Normalization Layer_: ensures stable training by normalizing the output
  9. _Residual Connection_: feeds into another normalization layer
  10. _Feed-Forward Network_: captures higher-level features
  11. _Normalization Layer_: maintains stablitity in the output.

## Combining the Encoder and Decoder to Complete the Transformer
* 
* Source: https://blog.spheron.network/how-to-build-an-llm-from-scratch-a-step-by-step-guide
