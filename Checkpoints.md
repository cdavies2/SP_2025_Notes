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
* When saving a model for inference, it's only required to save the trained model's learned parameters. Saving the model's state_dict with torch.save() will give
