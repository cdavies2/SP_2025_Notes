# Hugging Face LoRA
* LoRA of large language models is a popular, lightweight training technique that inserts a smaller number of new weights into the model, and only training on those.
* LoRA makes training faster and more memory-efficient.
* Before running the training script, you need to install the library from source:
  ```
  git clone https://github.com/huggingface/diffusers
  cd diffusers
  pip install .
  ```
* Then install the dependencies in the example folder
  ```
  cd examples/text_to_image
  pip install -r requirements.txt
  ```
* Initialize a Hugging Face Accelerate environment:
  ```accelerate config```
* To set up a default Accelerate environment without choosing configurations...
  ```accelerate config default```
## Script Parameters
* All parameters and their descriptions are found in the parse_args() function. Default values are provided for most parameters, but you can set your own too. For instance, to increase the number of epochs to train:
  ```
  accelerate launch train_text_to_image_lora.py \
  --num_train_epochs=150 \
  ```
* Some LoRA relevant parameters of Text-to-image training include...
  * --rank: the inner dimension of low-rank matrices to train; a higher rank means more trainable parameters
  * --learning_rate: the default learning rate is 1e-4, but with LoRA, you can use a higher learning rate
## Training Script
* Dataset preprocessing and training are found in main(), and if you need to adapt the training script, make changes there.

# Using Accelerate
* Allows users to scale PyTorch code for training and inference on distributed setups with hardware like GPUs and TPUs. It also makes running inference with larger models more accessible.
* Accelerate automatically selects the appropriate configuration values for a given distributed training framework through a unified configuration file generated from the accelerate config command.
  ``` accelerate config```
* The above command creates and saves a default_config.yaml file in Accelerates cache folder, which stores the configuration for your training environment, helping Accelerate correcly launch the training script.
* Once the environment is configured, you can test your setup with ```accelerate test```. You can add --config file to this command to specify the location of the configuration file. Once the environment is set up, launch the training script with accelerate launch
```accelerate launch path_to_script.py --args_for_the_script ```
### Adapt Training Code
* The Accelerator class adapts PyTorch code to run on different distributed setups
* To enable the script to run on multiple GPUs or TPUs...
  1. Import and instantiate the Accelerator class at the beginning of your training script, which initializes what's needed for distributed training and automatically detects the training environment based on how code was launched.
    ```
    from accelerate import Accelerator
    accelerator = Accelerator()
    ```
  2. Remove calls like .cuda() on your model and input data. You can also deactivate automatic device placement by passing device_placement=False when initializing the Accelerator.
  3. Pass all relevant PyTorch object training (optimizer, model, dataloader(s), learning rate scheduler) to the prepare() method as soon as they're created. This method wraps the model in a container optimized for your distributed setup, uses Accelerates version of the optimizer and scheduler, and creates a sharded version of your dataloader for distribution across GPUs or TPUs.
  ```
  model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
  )
  ```
  4. Replace loss.backward() with backward() to use the correct backward() method for your training setup.
  ```accelerator.backward(loss)```
### Distributed Evaluation
* To perform distributed evaluation, pass the validation dataloader to the prepare() method...
  ```validation_dataloader = accelerator.prepare(validation_dataloader)```
* Each device in a distributed setup only receives part of the evaluation data, meaning predictions (of the same size) should be grouped together. If sizes are different, use the pad_across_processes to pad the tensors to the largest size.
  ```
  for inputs, targets in validation_dataloader:
      predictions = model(inputs)
      # Gather all predictions and targets
      all_predictions, all_targets = accelerator.gather_for_metrics((predictions,
    targets))
      # Example of use with a *Datasets.Metric*
      metric.add_batch(all_predictions, all_targets)
  ```
### Empty Weights Initialization
* init_empty_weights() initializes models of any size by creating a model skeleton and moving and placing parameters each time they're created to PyTorch's meta device, ensuring only a small part of the model is loaded into memory at a time.
* For instance, loading an empty Mixtral-8x7B model takes significantly less memory than fully loading the models and weights on the CPU.
```
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```
### Load and Dispatch Weights
* The load_checkpoint_and_dispatch() function loads full or sharded checkpoints into the empty model, and automatically distribute weights across all available devices.
* The device_map parameter determines where to place model layers, and specifying "auto" places them on the GPU first, then CPU, and finally hard drive as memory-mapped tensors if there's still not enough memory. The no_split_module_classes parameter indicates which modules shouldn't be split across devices (especially those with a residual connection).
```
from accelerate import load_checkpoint_and_dispatch

model_checkpoint = "your-local-model-folder"
model = load_checkpoint_and_dispatch(
    model, checkpoint=model_checkpoint, device_map="auto", no_split_module_classes=['Block']
)
```
* Source: https://huggingface.co/docs/accelerate/quicktour

# Text-to-image
