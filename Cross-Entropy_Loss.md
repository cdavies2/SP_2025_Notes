# What is Cross-Entropy Loss Function?
* Cross-entropy loss (aka log loss) is used in machine learning to measure performance of classification models. An ideal score is close to zero, and the goal of a model's optimizer is to get the model as close to 0 as possible.
* Cross entropy measures the difference between predicted probability and true probability.
* Cross-Entropy-Loss is derived from the principles of maximum likelihood estimation when applied to the task of classification. Mazimizing likelihood is equivalent to minimizing the negative log-likelihood. Likelihood can be expressed at the product of the probabilities of the correct classes.
* Binary Cross-Entropy Loss is usually used for binary classification problems.
* Multiclass Cross-Entropy Loss, also known as categorical cross-entropy or softmax loss, is widely used for training models in multiclass classification problems.
## Why Not MCE for All Cases?
* In binary classification, the output layer utilizes the sigmoid activation function, resulting in the neural network producing a single probability score (p) ranging between 0 and 1 for the two classes. The unique approach in binary classification involves not encoding binary predicted values as different for class 0 and class 1. They are instead stored as single values, saving model parameters. However, in multiclass classification, the softmax activation is employed in the output layer to obtain a vector of predicted probabilities (p).
* The standard definition of cross-entropy cannot be directly applied to binary classification problems where computed and correct probabilities are stored as singular values.

## How to Interpret Cross Entropy Loss?
* Cross-entropy loss is a scalar value that quantifies how well a model's prediction matches the true label, higher loss means higher discrepancy.
  * _Interpretability with Binary Classification_: in binary classification, since there are two classes (0 and 1), if the true label is 1, loss is primarily influenced by how close the predicted probability for class 1 is to 1.0. If the true label is 0, the loss is influenced by how close the predicted probability for class 1 is to 0.0
  * _Interpretability with Multiclass Classification_: in multiclass classification, only the true label contributes towards the loss. Lower loss indicates that the model is assigning high probabilities to the correct class and low probabilities to incorrect classes.

## Key Features of Cross Entropy Loss
* _Probabilistic Interpretation_: cross-entropy loss encourages the model to output predicted probabilities that are close to the true class probabilities
* _Gradient Descent Optimization_: the mathematical properties of Cross-Entropy Loss make it well-suited for optimization algorithms like gradient descent. The gradient of the loss concerning the model parameters is relatively simple to compute.
* _Commonly Used in Neural Networks_: standard choice for training networks for deep learning because it is supported in those frameworks and aligns well with the softmax activation function.
* _Ease of Implementation_: comparison implementing cross-entropy loss is straightforward and readily available in most machine learning libraries.

## Implementation
* torch.nn.BCELoss is suitable for binary classification problems and is commonly used in PyTorch for tasks where each input sample belongs to one of two classes.
* The inputs to torch.nn.BCELoss are the predicted probabilities and the target labels
* The predicted probabilities should be a tensor of shape (batch_size, 1)
* The target labels should be either 0 or 1.
```
import torch
import torch.nn as nn
import torch.optim as optim

#Define your dataloader

# Define a simple neural network
class Net(nn.Module):

# Instantiate the network, loss function, and optimizer
model = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Inside the training loop
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(torch.sigmoid(outputs), targets)
    loss.backward()
    optimizer.step()
```
* Multi-class Cross-Entropy Loss can be implemented using the PyTorch library _torch.nn.CrossEntropyLoss_
  * **torch.nn.CrossEntropyLoss** combines the functionalities of the softmax activation and the negative log-likelihood loss
  * The input to this loss function is typically raw output scores from the last layer of a neural network, without applying an explicit activation function like softmax. It expects the raw scores (logits) for each class.
  * Internally, CrossEntropyLoss applies the softmax activation function to the raw scores before computing the loss. This ensures that the predicted values are transformed into probabilities, and the sum of probabilities for all classes equals 1
  * CrossEntropyLoss supports the use of class weights to handle class imbalance
  * The targets should contain class indices (integers) for each input sample, and the number of classes should match the number of output neurons in the last layer of the network.
* Source: https://www.geeksforgeeks.org/what-is-cross-entropy-loss-function/
