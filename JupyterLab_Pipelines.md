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
  * model_kwargs-"torch_dtype": torch.bfloat16: bfloat16 is the default data type
* https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html

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

## Chat Templates
* In a chat context, rather than continuing a single string of text, the model instead continues a conversation that consists of one or more messages, each of which includes a role like "user" or "assistant" along with message text.
* Much like tokenization, different models expect different input formats for chat, so chat templates (part of the tokenizer) specify how to convert conversations, represented as lists of messages, into a single tokenizable string in the expected format.
* The example below uses the mistralai/Mistral-7B-Instruct-v0.1 model
```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)
```
* The tokenizer adds the control tokens [INST] and [/INST] to indicate start and end of user messages (but not assistant messages) and condenses the chat into a single string.

 ### Is there an Automated Pipeline for Chat?
 * Text generation pipelines support chat inputs, making it easy to use chat models. This has been functionally merged into the TextGenerationPipeline, and works as below
```
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response
```
* The pipeline takes care of all the details of tokenization and calling `apply_chat_template` for you. Just initialize the pipeline and pass it the list of messages

### What are "Generation Prompts"?
* The `add_generation_prompt` argument in the `apply_chat_template` method tells the template to add tokens that indicate the start of a bot response. For instance....
```
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```
* By adding the tokens that indicate the start of a bot response, we ensure that when the model generates text, it will write a bot response instead of say, continuing the user's message.

### What does "continue_final_message" do?
* When passing a list of messages to `apply_chat_template` or `TextGenerationPipeline` you can choose to format the chat so the model will continue the final message in the chat instead of starting a new one.
* This is done by removing any end-of-sequence tokens that indicate the end of the final message, so the model will simply extend the final mesage when it begins to generate text. An example is below...
```
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```
* The model will generate text that continues the JSON string, rather than starting a new message.
* Because `add_generation_prompt` adds the tokens that start a new message, and `continue_final_message` removes any end-of-message tokens from the final message, you cannot use them together.
* The default behavior of `TextGenerationPipeline` is to set `add_generation_prompt=True` so that it starts a new message. However, if the final message in the input chat has the "assistant" role, it will assume that this message is a prefill and switch to `continue_final_message=True` instead, because most models do not support multiple consecutive assistant messages. This can be overridden by explicitly passing the `continue_final_message` argument when calling the pipeline.
