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
  * `--rank`: the inner dimension of low-rank matrices to train; a higher rank means more trainable parameters
  * `--learning_rate`: the default learning rate is 1e-4, but with LoRA, you can use a higher learning rate
## Training Script
* Dataset preprocessing and training are found in main(), and if you need to adapt the training script, make changes there.
### Using UNet
* Diffusers uses `~peft.LoraConfig` from the PEFT library to set up the parameters of the LoRA adapter such as the rank, alpha, and which modules to insert the LoRA weights into. The adapter is added to the UNet, and only the LoRA layers are filtered for optimization in `lora_layers`.
```
unet_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

unet.add_adapter(unet_lora_config)
lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
```
* The optimizer is initialized with the `lora_layers` because these are the only weights that'll be optimized.
```
optimizer = optimizer_cls(
    lora_layers,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```
### Using Text Encoder
* Diffusers also supports finetuning the text encoder with LoRA from the PEFT library when necessary, such as finetuning Stable Diffusion XL (SDXL). The `~peft.LoraConfig` is used to configure the parameters of the LoRA adapter which are then added to the text encoder, and only the LoRA layers are filtered for training.
```
text_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    init_lora_weights="gaussian",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
)

text_encoder_one.add_adapter(text_lora_config)
text_encoder_two.add_adapter(text_lora_config)
text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
```
* The optimizer is initialized with `lora_layers` because they're the only weights that'll be optimized
```
optimizer = optimizer_cls(
    lora_layers,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
```
* Aside from setting up LoRA layers, the training script is the same as train_text_to_image.py

## Launch the Script
* Let's train on the Naruto BLIP captions datset to generate Naruto characters. Set the environment variables `MODEL_NAME` and `DATASET_NAME` to the model and dataset respectively. You also specify where to save the model in OUTPUT_DIR, and the name of the model to save on the Hub with `HUB_MODEL_ID`. The script creates and saves the saved model checkpoints and trained LoRA weights (`pytorch_lora_weights.safetensors`) to your repository.
* A full training run takes ~5 hours on a 2080 Ti GPU with 11GB of VRAM.
```
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/naruto"
export HUB_MODEL_ID="naruto-lora"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A naruto with blue eyes." \
  --seed=1337
```
* Source: https://huggingface.co/docs/diffusers/en/training/lora
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
* gradient_checkpointing and mixed_precision allow you to train a model on a single 24GB GPU. You can reduce memory by enabling memory-efficient attention with xFormers. JAX/Flax training is also supported for efficient training on TPUs and GPUs, but doesn't support gradient checkpointing, gradient accumulation, or xFormers.
* Before running the train_text_to_image.py training script, first install the library from source...
  ```
  git clone https://github.com/huggingface/diffusers
  cd diffusers
  pip install .
  ```
* Next, navigate to the example folder containing the training script and install required dependencies for the script you're using
  ```
  cd examples/text_to_image
  pip install -r requirements.txt
  ```
 * Next, initalize an Accelerate environment with `accelerate config`
## Script Parameters
* The training script provides many parameters to help you customize the training run, all found in the parse_args() function.
* Some basic and important parameters include:
  * `--pretrained_model_name_or_path`: the name of the model on the Hub or a local path to the pretrained model
  * `--dataset_name`: the name of the dataset on the Hub or a local path to the dataset to train on
  * `--image_column`: the name of the image column in the dataset to train on
  * `--output_dir`: where to save the trained model
  * `--push_to_hub`: whether to push the trained model to the Hub
  * `--checkpointing_steps`: frequency of saving a checkpoint as the model trains; this is useful so if training is interrupted, you can continue training from that checkpoint by adding `--resume_from_checkpoint` to your training command
## Training Script
* The dataset preprocessing code and training loop are found in the main() function, and this is where the training script is modified
* The `train_text_to_image` script starts by loading a scheduler and tokenizer. You can choose to use a different scheduler here...
  ```
  noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
  tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
  )
  ```
* The script then loads the UNet model:
  ```
  load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
  model.register_to_config(**load_model.config)

  model.load_state_dict(load_model.state_dict())
  ```
* Next, the text and image columns of the dataset are preprocessed. The tokenize_captions function handles tokenizing the inputs, and the train_transforms function specifies the type of transforms to apply to the image. Both of these are built into `preprocess_train`
  ```
  def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples
  ```
* The training loop handles everything else, encoding images into latent space, adding noise to latents, computing text embeddings to condition on, updating model parameters, and saving and pushing the model to the Hub.
## Launch the Script
* As an example, train on the Naruto BLIP captions dataset to generate Naruto characters. The environment variables `MODEL_NAME` and `dataset_name` will be set to the model and dataset. If you're training on multiple GPUs, add the `--multi_gpu` parameter to the `accelerate launch` command.
* To train on a local dataset, set the `TRAIN_DIR` and `OUTPUT_DIR` environment variables to the path of the dataset and where to save the model to.
```
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model" \
  --push_to_hub
```
* After training is done, use the newly trained model for inference:
```
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("path/to/saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

image = pipeline(prompt="yoda").images[0]
image.save("yoda-naruto.png")
```
* Source: https://huggingface.co/docs/diffusers/en/training/text2image#script-parameters

# Load Adapters
* 
