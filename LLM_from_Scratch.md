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
* 
* Source: https://arxiv.org/pdf/1706.03762
