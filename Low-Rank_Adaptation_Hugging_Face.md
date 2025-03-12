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
* Some adapters generate an entirely new model, while other only modify a smaller set of embeddings or weights. This means that each adapter also has a different loading process.
## Dreambooth
* DreamBooth finetunes an entire diffusion model on just several images of a subject to generate images of said subject in new styles and settings. This method uses a special word in the prompt that the model learns to associate with the subject image. Of all the training methods, DreamBooth produces the largest file size (usually a few GBs) because it is a full checkpoint model.
* In this example, the word `herge_style` in the prompt triggers the checkpoint
```
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("sd-dreambooth-library/herge-style", torch_dtype=torch.float16).to("cuda")
prompt = "A cute herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```
## LoRA
* LoRA is fast and generates smaller file sizes (a few hundred MBs). It can train a model to learn new styles from a few images. It works by inserting new weights into the diffusion model and then only those weights are trained.
* LoRA is a very general training technique that can be used with other training methods. For instance, it is common to train a model with DreamBooth and LoRA, or to load and merge mutliple LoRAs to create new and unique images.
* LoRAs also need to be used with another model:
```
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
```
* Then use the `load_lora_weights()` method to load new weights and specify their filename from the repository
```
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors")
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]
image
```
* The load_lora_weights() method loads LoRA weights into both the UNet and text encoder. It is the preferred way for loading LoRAs because it can handle cases where
  * LoRA weights don't have separate identifiers for the UNet and text encoder
  * LoRA weights have separate identifiers for the UNet and text encoder
* To directly load (and save) a LoRA adapter at the model level, use `~PeftAdapterMixin.load_lora_adapter`, which builds and prepares the necessary model configuration for the adapter. Like `load_lora_weights()`, `PeftAdapterMixin.load_lora_adapter` can load LoRAs for both the UNet and text encoder. For instance, if loading a LoRA for the UNet, `PeftAdapterMixin.load_lora_adapter` ignores the keys for the text encoder.
* The `weight_name` parameter specifies the weight file and `prefix` filters for the appropriate state dicts ("unet" in this case) to load.
```
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_lora_adapter("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", prefix="unet")

# use cnmt in the prompt to trigger the LoRA
prompt = "A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```
* Save an adapter with `~PeftAdapterMixin.save_lora_adapter`. To unload LoRA weights, use the unload_lora_weights() method to discard the LoRA weights and restore the model to its original weights. `pipeline.unload_lora_weights()`

### Adjust LoRA Weight Scale
* For load_lora_weights() and load_attn_procs(), you can pass the `cross_attention_kwargs={"scale": 0.5}` parameter to adjust how much of the LoRA weights to use. 0 is base weights, 1 is fully finetuned LoRA.
* For more granular control on the amount of LoRA weights used per layer, use `set_adapters()` and pass a dictionary specifying how much to scale weights in each layer by.
```
pipe = ... # create pipeline
pipe.load_lora_weights(..., adapter_name="my_adapter")
scales = {
    "text_encoder": 0.5,
    "text_encoder_2": 0.5,  # only usable if pipe has a 2nd text encoder
    "unet": {
        "down": 0.9,  # all transformers in the down-part will use scale 0.9
        # "mid"  # in this example "mid" is not given, therefore all transformers in the mid part will use the default scale 1.0
        "up": {
            "block_0": 0.6,  # all 3 transformers in the 0th block in the up-part will use scale 0.6
            "block_1": [0.4, 0.8, 1.0],  # the 3 transformers in the 1st block in the up-part will use scales 0.4, 0.8 and 1.0 respectively
        }
    }
}
pipe.set_adapters("my_adapter", scales)
```
## Kohya and TheLastBen
* These other LoRA trainers create different LoRA checkpoints than those trained by Diffusers, but they can still be loaded in the same way.
* EX: this is how the TheLastBen/William_Eggleston_Style_SDXL checkpoint is loaded
```
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("TheLastBen/William_Eggleston_Style_SDXL", weight_name="wegg.safetensors")

# use by william eggleston in the prompt to trigger the LoRA
prompt = "a house by william eggleston, sunrays, beautiful, sunlight, sunrays, beautiful"
image = pipeline(prompt=prompt).images[0]
image
```
* Source: https://huggingface.co/docs/diffusers/en/using-diffusers/loading_adapters#LoRA

# Create a Dataset
* Creating a dataset with Hugging Face Datasets confers all advantages of the library to your dataset: fast loading and processing, streaming enormous datasets, and memory mapping.
* Datasets has a low-code approach, reducing time taken to start training a model. In some cases, you just need to drag and drop your data files into a dataset repository on the Hub.

## File-Based Builders
* Hugging Face Datasets supports many common formats like csv, json/jsonl, parquet, and txt. Below is an example where multiple CSV files are passed as a list.
```
from datasets import load_dataset
dataset = load_dataset("csv", data_files="my_file.csv")
```
## Folder-Based Builders
* `ImageFolder` and `AudioFolder` are both used for quickly creating an image or speech and audio dataset with several thousand examples. They work well for rapidly prototyping computer vision and speech models before scaling to a larger dataset.
  * `ImageFolder` uses the Image feature to decode an image file. Many image extension formats are supported, such as jpg and png, but other formats are also supported. You can check the complete list of supported image extensions.
  * `AudioFolder` uses the Audio feature to decode an audio file. Audio extensions such as wav and mp3 are supported.
* Dataset splits are generated from the repository structure, and label names are automatically inferred from the directory name.
* EX: folder/train/grass/bulbasaur.png (train is used to generate split, grass is used to infer the label name, and bulbasaur is decoded by image)
* An image dataset is created by specifying imagefolder in load_dataset()
```
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="/path/to/pokemon")
```
* Any additional information about your dataset, such as text captions or transcriptions, can be included with a `metadata.csv` file in the folder containing your dataset. Said file needs a `file_name` column that links an image or audio file to its corresponding metadata.

## From Python Dictionaries
* You can also create a dataset from data in Python dictionaries. `from_` methods can accomplish this in two ways...
  * The from_generator() method is the most memory-efficient way to create a dataset from a generator due to a generator's iterative behavior (especially useful with larger datasets).
  ```
  from datasets import Dataset
  def gen():
      yield {"pokemon": "bulbasaur", "type": "grass"}
      yield {"pokemon": "squirtle", "type": "water"}
  ds = Dataset.from_generator(gen)
  ds[0]
  ```
  * A generator-based IterableDataset needs to be iterated over with a for loop
  ```
  from datasets import IterableDataset
  ds = IterableDataset.from_generator(gen)
  for example in ds:
      print(example)
  ```
  * The from_dict() method also lets you create a dataset from a dictionary
  ```
  from datasets import Dataset
  ds = Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]})
  ds[0]
  # returns {"pokemon": "bulbasaur", "type": "grass"}
  ```
* Source: https://huggingface.co/docs/datasets/en/create_dataset

# Load a Dataset from the Hub
* Before downloading a dataset, it's often helpful to quickly get general information, which is stored in DatasetInfo. It includes information like the dataset description, features, and size.
* load_dataset_builder() loads a dataset builder and inspects a dataset's attributes without committing to downloading it:
```
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")

ds_builder.info.description
# Returns: Movie Review Dataset. This is a dataset of containing 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. This data was first used in Bo Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.'', Proceedings of the ACL, 2005
ds_builder.info.features
# Returns: {'label': ClassLabel(names=['neg', 'pos'], id=None),
 'text': Value(dtype='string', id=None)}
```
* If you're happy with a dataset, load it with load_dataset()
```
from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
```
## Splits
* A split is a specific subset of a dataset like train and test. List a dataset's split names with the get_dataset_split_names() function
```
from datasets import get_dataset_split_names

get_dataset_split_names("cornell-movie-review-data/rotten_tomatoes")
```
* You can also load a specific split with the split parameter. Loading a dataset split returns a Dataset object.
```
from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
dataset
```
## Configurations
* Some datasets contain several sub-datasets. These are known as configurations or subsets, and you must explicitly select one when loading a dataset.
* Use the get_dataset_config_names() function to retrieve a list of all possible configurations available to your dataset.
```
from datasets import get_dataset_config_names

configs = get_dataset_config_names("PolyAI/minds14")
print(configs)
```
* Then load the configuration you want:
```
from datasets import load_dataset

mindsFR = load_dataset("PolyAI/minds14", "fr-FR", split="train")
```
## Remote Code
* To use a dataset with a loading script (Python code is used to generate the dataset), set `trust_remote_code=True`
```
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

c4 = load_dataset("c4", "en", split="train", trust_remote_code=True)
get_dataset_config_names("c4", trust_remote_code=True)
get_dataset_split_names("c4", "en", trust_remote_code=True)
```
* Source: https://huggingface.co/docs/datasets/en/load_hub

# Load Text Data
* By default, Hugging Face Datasets samples a text file line by line to build the dataset.
```
from datasets import load_dataset
dataset = load_dataset("text", data_files={"train": ["my_text_1.txt", "my_text_2.txt"], "test": "my_test_file.txt"})

dataset = load_dataset("text", data_dir="path/to/text/dataset")
```
* To sample a text file by paragraph or even an entire document, use the `sample_by` parameter.
```
dataset = load_dataset("text", data_files={"train": "my_train_file.txt", "test": "my_test_file.txt"}, sample_by="paragraph")

dataset = load_dataset("text", data_files={"train": "my_train_file.txt", "test": "my_test_file.txt"}, sample_by="document")
```
* You can also use grep patterns to load specific files
```
from datasets import load_dataset
c4_subset = load_dataset("allenai/c4", data_files="en/c4-train.0000*-of-01024.json.gz")
```
* Source: https://huggingface.co/docs/datasets/en/nlp_load

# PEFT LoRA
* LoRA makes fine-tuning more efficient by representing weight updates with two smaller matrices (called update matrices) through low-rank decomposition. Said matrices can be trained to adapt to new data while keeping the overall number of changes low. The original weight matrix remains frozen and receives no further adjustments, and the results combine the original and adapted weights.
* Advantages of LoRA include....
  * LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.
  * The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.
  * LoRA is orthogonal to many other parameter-efficient methods and can be combined with many of them.
  * Performance of models fine-tuned using LoRA is comparable to the performance of fully fine-tuned models
  * LoRA does not add any inference latency because adapter weights can be merged with the base model.
* In Transformer models, LoRA is usually added to attention blocks only. The resulting number of trainable parameters in a LoRA model depends on the size of the low-rank update matrices, which is determined mainly by the rank r and the shape of the original weight matrix.

## Merge LoRA Weights Into the Base Model
* To eliminate latency, use the merge_and_unload() function to merge adapter weights with the base model, allowing you to use the new merged model as a standalone model.
* During training, smaller weight matrices are separate, but once training is complete, said weights can be merged into an identical weight matrix.
## Utils for LoRA
* merge_adapter() merges LoRA layers into the base model while retaining the PeftModel, making umerging and loading different adapters easier.
* unmerge_adapter() unmerges LoRA layers from the base model while retaining the PeftModel (assists in merging, deleting, loading different adapters)
* unload() gets back the base model without merging active LoRA modules. This helps when you want to get back to the pretrained base model in some applications when you want to reset the model to its original state.
* delete_adapter() deletes an existing adapter
* add_weighted_adapter() combines multiple LoRAs into a new adapter based on the user provided weighing scheme
## Common LoRA Parameters in PEFT
* To fine-tune a model using LoRA, you must....
  1. Instantiate a base model
  2. Create a configuration (LoraConfig) where you define LoRA-specific parameters
  3. Wrap the base model with get_peft_model() to get a trainable PeftModel
  4. Train the PeftModel as you normally would train the base model
* LoraConfig lets you control how LoRA is applied to the base model through these parameters...
  * r: the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters
  * target_modules: the modules (for example, attention blocks) to apply the LoRA update matrices
  * lora_alpha: LoRA scaling factor
  * bias: specifies if the bias parameters should be trained. Can be 'none', 'all', or 'lora_only'
  * use_rslora: when set to True, uses Rank-Stablized LoRA, which sets the adapter scaling factor to lora_alpha/math.sqrt(r), since it was proven to work better. Otherwise, it will use the original default value of lora_alpha/r.
  * modules_to_save: list of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model's custom head that is randomly initialized for the fine-tuning task.
  * layers_to_transform: list of layers to be transformed by LoRA. If not specified, all layers in target_modules are transformed.
  * layers_pattern: pattern to match layer names in target_modules, if layers_to_transform is specified. By default PeftModel looks at common layer pattern (layers, h, blocks, etc), use it for exotic and custom models.
  * rank_pattern: the mapping from layer names or regexp expression to ranks which are different from the default rank specified by r
  * alpha_pattern: the mapping from layer names or regexp expression to alphas which are different from the default alpha specified by lora_alpha
## Initialization Options
* The initialization of LoRA weights is controlled by the parameter init_lora_weights of the LoraConfig. PEFT initializes LoRA weights by using Kaiming-uniform for weight A and initializing weight B as zeros, resulting in an identity transform.
* It's also possible to pass init_lora_weights="gaussian", which initializes weight A with a Gaussian distribution (weight B is still zeros) corresponding to the way that the diffusers library initializes LoRA weights.
* LoftQ performs initialization in a way that ensures quantization error is minimized. To do this, DON'T quantize the base model, instead use the code below...
  ```
  from peft import LoftQConfig, LoraConfig, get_peft_model
  
  base_model = AutoModelForCausalLM.from_pretrained(...)  # don't quantize here
  loftq_config = LoftQConfig(loftq_bits=4, ...)           # set 4bit quantization
  lora_config = LoraConfig(..., init_lora_weights="loftq", loftq_config=loftq_config)
  peft_model = get_peft_model(base_model, lora_config)
  ``` 
* Source: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
  
## LoRA Examples: Semantic Segmentation
* LoRA adds low-rank "update matrices" to specific blocks of a model, and during fine-tuning, only these matrices are trained, while the original model parameters are left unchanged.
### Install Dependencies
`!pip install transformers accelerate evaluate datasets peft -q `
### Authenticate to Share Your Model
* Use your hugging face token
```
from huggingface_hub import notebook_login

notebook_login()
```
### Load a Dataset
* To ensure a reasonable time frame for the example, limit the number of instances from a training set to 150.
```
from datasets import load_dataset

ds = load_dataset("scene_parse_150", split="train[:150]")
```
* Then split the dataset into train and test sets
```
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
```
### Load a Base Model
* Before loading a base model, define a helper function to check the total number of parameters a model has and how many of them are trainable.
```
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
```
* Choose a base model checkpoint (for instance, the SegFormer B0 variant). In addition to the checkpoint, pass label2id and id2label dictionaries to let the AutoModelForSemanticSegmentation class know we're interested in a custom base model where the decoder head should be randomly initialized using classes from the custom dataset.
```
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
print_trainable_parameters(model)
```
### Wrap Base Model as a PeftModel for LoRA Training
* To leverage LoRA, you must wrap the base model as a PeftModel, which has two steps
  1. Define LoRA configuration with LoraConfig
  2. Wrapping the original model with get_peft_model() using the config
```
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
```
* To enable LoRA technique, we must define the target modules within LoraConfig so that PeftModel can update the necessary matrices. We want to target the query and value matrices in the attention blocks of the base model. "query" and "value" should be specified in the target_modules argument of LoraConfig
* After the base model is wrapped with PeftModel along with the config, we get a new model where only the LoRA parameters are trainable ("update matrices") while the pre-trained parameters are kept frozen. To ensure classifier parameters are also trained, specify modules_to_save. This also ensures that these modules are serialized alongside the LoRA trainable parameters when using utilities like save_pretrained() and push_to_hub()
* When we wrap our base model with PeftModel and pass the configuration, a new model is obtained where only LoRA parameters are trainable, and pre-trained/randomly initialized classifier parameters are kept frozen. To ensure classifier parameters are trained, specify modules_to_save. This ensures that modules are serialized alongside LoRA trainable parameters when using utilities like save_pretrained() and push_to_hub()
* Other parameters include:
  * r: the dimension used by the LoRA update matrices
  * alpha: Scaling factor
  * bias: specifies if the bias parameters should be trained. None denotes none of the bias parameters will be trained.
* print_trainable_parameters lets us explore number of trainable parameters (should be lower in lora_model than the original). You can also manually verify what modules are trainable in the lora_model.
```
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
```
### Save the Model and Run Inference
* Use the save_pretrained() method of the lora_model to save the LoRA-only parameters locally. Alternatively, use the push_to_hub() method to upload these parameters directly to the Hugging Face Hub
```
model_id = "segformer-scene-parse-150-lora"
lora_model.save_pretrained(model_id)
```
* You can then prepare an inference_model and run inference
```
from peft import PeftConfig

config = PeftConfig.from_pretrained(model_id)
model = AutoModelForSemanticSegmentation.from_pretrained(
    checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

inference_model = PeftModel.from_pretrained(model, model_id)
```
* Source: https://huggingface.co/docs/peft/main/en/task_guides/semantic_segmentation_lora

## Fine-Tune a Pretrained Model
* Before fine-tuning a pretrained model, download a dataset and prepare it for training.
* First, load the dataset
```
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]
```
* A tokenizer is needed to process text and include a padding and truncation strategy to handle any variable sequence lengths. The Datasets map method can apply a preprocessing function over an entire dataset.
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
```
* A smaller subset of the full dataset can be created to fine-tune on, thus reducing processing time.
```
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```
### Train with PyTorch Trainer
* The Trainer class is optimized for training Transformers models, making it easier to start training without manually writing a training loop.
* Start by loading the model, specifying the number of expected labels.
* By default, weights are loaded in full precision regardless of data type, and setting `torch_dtype="auto"` loads weights in the data type defined in a model's config.json file to automatically load the most memory-optimal data type.
```
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
```
### Training Hyperparameters
* Create a TrainingArguments class which contains all the hyperparameters you can tune as well as flags for activating different training options. Specify where to save training checkpoints
```
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
```
### Evaluate
* Trainer needs to be passed a function from the Evaluate library (the accuracy function) to compute and report metrics.
```
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
```
* Call compute on metric to calculate prediction accuracy. Before passing predictions to compute, convert logits to predictions (models return logits)
```
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```
* If you'd like to monitor evaluation metrics during fine-tuning, specify the `eval_strategy` parameter in the training arguments to report the evaluation metric at the end of each epoch.
```
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```
### Trainer
* Create a Trainer object with your model, training arguments, training and test datasets, and evaluation function, and fine-tune by calling train()
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```
### Train in Native PyTorch
* If you don't want to use Trainer's loop, you can write your own in PyTorch. You might need to run the code below at the beginning to free some memory
```
from accelerate.utils.memory import clear_device_cache
del model
del trainer
clear_device_cache()
```
* Next, manually postprocess the dataset: Remove the text column (not accepted as input), rename the label column to labels, set the format to return Tensors, and create a smaller subset
```
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```
* Then create a DataLoader for your training and test datasets so you can iterate over batches of data, and load your model with the number of expected labels
```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```
* Create an optimizer and learning rate scheduler to fine-tune the model, then create the default learning rate scheduler from Trainer
```
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
```
* Specify device GPU if you have one (this can make training take minutes as opposed to hours on a CPU)
```
import torch
from accelerate.test_utils.testing import get_backend

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model.to(device)
```
* To keep track of training progress, use the tqdm library to add a progress bar over the number of training steps
```
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
* Instead of calculating and reporting the metric at the end of each epoch, accumulate all the batches with `add_batch` and calculate the metric at the very end
```
import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```
* Source: https://huggingface.co/docs/transformers/en/training
