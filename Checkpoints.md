# Saving and Loading Models
* There are three core saving and loading functions to be familiar with...
  1. _torch.save_: saves a serialized object to disk. This function uses Python's pickle entity for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved with this function
  2. _torch.load_: uses pickle's unpickling facilities to deserialize pickled object files to memory. This also facilitates the device to load the data
  3. _torch.nn.Module.load_state_dict_: loads a model's parameter dictionary using a deserialized state_dict.

## What is a state_dict?
* In PyTorch, learnable parameters (EX: weights and biases) of a torch.nn.Module model are contained in the model's parameters. A state_dict is simply a Python dictionary object that maps layers to their parameter tensors.
* Optimizer objects (torch.optim) also have a state_dict, which contains information about the optimizer's state as well as hyperparameters used.
* state_dict objects are Python dictionaries, so they can be easily saved, updated, altered, and restored, adding great modularity to Pytorch models and optimizers.
* Source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
### Example from Simple Model
```
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```
* For the above, the model's state_dict gives torch.size for each parameter (EX: conv1.weight = torch.Size([6, 3, 5, 5]), fc1.bias = torch.Size([120]), and the optimizer's state_dict provided param_groups (EX: 'lr': 0.001, 'dampening': 0, 'nesterov': False, 'params': [4675713712, 4675713784...]

## Saving and Loading Model for Inference
* Save state_dict (Recommended)
` torch.save(model.state_dict(), PATH) `
* Load state_dict (Recommended)
```
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
```
* When saving a model for inference, it's only required to save the trained model's learned parameters. Saving the model's state_dict with torch.save() will give the most flexibility for restoring the model later.
* You must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this yields inconsistent inference results.
* If you only plan to keep the best performing model, `best_model_state = model.state_dict()` returns a reference to the state, not its copy. You should use `best_model_state = deepcopy(model.state_dict())`, otherwise `best_model_state` will continually get updated.
## Saving and Loading Entire Model
* Save
` torch.save(model, PATH)`
* Load
```
# Model class must be defined somewhere
model = torch.load(PATH, weights_only=False)
model.eval()
```
* Saving a model in this way saves the entire module, but the disadvantahe is the serialized data is bound to specific classes and exact directory structure used when the model is saved (as the path to the file with the class is saved, not the class itself).
* Remember, call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.

## Export/Load Model in TorchScript
* TorchScript is the recommended format for scaled inference and deployment. It allows you to load a model and run inference without defining the model class.
* Export
```
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save
```
* Load
```
model = torch.jit.load('model_scripted.pt')
model.eval()
```
* Remember to call model.eval()

## Saving and Loading a General Checkpoint for Inference and/or Resuming Training
* Save
```
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
```
* Load
```
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```
* When saving a general checkpoint, either for inference or resuming training, save both the model and optimizer's sate_dict (as that contains buffers and parameters that are updated as a model trains). You also might want to save the latest recorded training loss, and external `torch.nn.Embedding` layers.
* To save multiple components, organize them in a dictionary and use torch.save() to serialize it (saving checkpoints with the .tar file extension.
* Source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
