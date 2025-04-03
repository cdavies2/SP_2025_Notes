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
 * _Multi-Head Attention_:
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
* Source: https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/
