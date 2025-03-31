# Creating a Large Language Model from Scratch: A Beginner's Guide

* Source: https://www.pluralsight.com/resources/blog/ai-and-data/how-build-large-language-model

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
 
* Source: https://www.pluralsight.com/resources/blog/ai-and-data/what-are-transformers-generative-ai
