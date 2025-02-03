# Transformers
* Transformers provides APIs and tools to download and train state-of-the-art pretrained models, which can reduce the cost and time required to train a model from scratch.
* Transformers supports models used for natural language processing (language modeling, code and text generation), computer vision (image classification), audio (speech recognition), and multimodal (information extraction from scanned documents, video classification).
* The model can be trained in three lines of code in one framework and loaded for inference in another.
* https://huggingface.co/docs/transformers/en/index
## Quick Tour
* First, you must make sure you have all necessary libraries installed

  ```!pip install transformers datasets evaluate accelerate```
* You also need to install your preferred machine learning framework

  ```pip install torch```
* The `pipeline()` is the easiest and fastest way to use a pretrained model for inference
* Some pipeline() modalities include...
  * Text classification: assigning a label to a given sequence of text, `pipeline(task="sentiment-analysis")`
  * Text generation: generate text given a prompt, `pipeline(task="text-generation")`
  * Automatic speech recognition: transcribe speech into text, `pipeline(task="automatic-speech-recognition")`
  * Visual question answering: answer a question about an image, given said image and a question, `pipeline(task="vqa")`
  * Document question answering: answer a question about a given document, `pipeline(task="document-question-answering")`
  * Image captioning: generate a caption for a given image, `pipeline(task="image-to-text")`
* Using Jupyter Notebooks for transformers models is highly recommended because they allow for cell-by-cell execution, which can be helpful to better split logical components from one another and to have faster debugging cycles as intermediate results can be stored. Notebooks can also be easily shared with other collaborators.
* https://huggingface.co/docs/transformers/en/quicktour

## The Pipeline Abstraction
* transformers.pipeline() is the utility factory method to build a pipeline
* Pipelines are made of...
  * A Tokenizer instance in charge of mapping raw textual input to token
  * A Model instance
  * Some (optional) post processing for enhancing model's output
*  Parameters include...
  * task(str): defines which pipeline will be returned,
    * EX(task: "text-generation")
  * model(str or PreTrainedModel): the model that will be used by the piepline to make predictions. It can be None, an identifier, or an actual pretrained model
    *  EX: (model="mistralai/Mistral-7B-v0.1)
  *  max_new_tokens: maximum tokens generated
    * EX: max_new_tokens=128
  * 
    
   

## LLM Prompting Guide
* Large language models like Falcon and Llama are pretrained transformer models initially trained to predict the next token given some input text.
* Designing prompts to ensure optimal output is often called "prompt engineering", an iterative process that requires a fair amount of experimentation. Natural languages are more expressive and flexible than programming languages, but their prompts are much more sensitive to change.
* The majority of modern LLMs are decoder-only transformers (EX: Llama2, Falcon, GPT2).
* Encoder-decoder-style models are typically used for generative tasks where output heavily relies on input (translation and summarization) while decoder-only models are used for all other types of generative tasks
* Decoder-only models use the `text-generation` pipeline, as seen below.
```
from transformers import pipeline
import torch

torch.manual_seed(0)
generator = pipeline('text-generation', model = 'openai-community/gpt2')
prompt = "Hello, I'm a language model"

generator(prompt, max_length = 30)
```
* https://huggingface.co/docs/transformers/en/tasks/prompting
