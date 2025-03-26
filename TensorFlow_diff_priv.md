# Machine Learning with Differential Privacy in Tensorflow
* Learning with Differential Privacy not only has provable privacy guarantees, but also mitigates the risk of exposing sensitive training data in machine learning
* Differentially Private Stoachastic Gradient Descent (DP-SGD) is a modification of the stochastic gradient descent algorithm. The TensorFlow Privacy library provides an implementation of DP-SGD, but in order to do so, you must first train a simple neural network in TensorFlow.

## Training a Neural Network using Keras API in Tensorflow
* Keras is a user-friendly, high-level API for building and training neural networks. The steps involved in using it are...
1. Import Libraries
```
import tensorflow as tf
from tensorflow.keras.models import Sequential #models are objects that group layers together and can be trained on data
# A Sequential model is a linear stack of layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# layers encapsulate a state (weights) and some computation, its the fundamental abstraction
# Dense is used for densely-connected NN layers
# Conv2D is used for 2D convolution layers, MaxPooling2D is the max pooling operation for 2D spatial data
# Dropout applies dropout to the input, Flatten flattens input
from tensorflow.keras.optimizers import Adam
# optimizers are used to compile Keras models
# Adam optimization is a stochastic gradient descent method based on adaptive estimation of first-order and second-order moments
```
2. Prepare the Data (load and preprocess it). MNIST will be used in this example.
```
from tensorflow.keras.datasets import mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

```
3. Build the Model (define the architecture of the neural network)
```
# for Conv2D, the first value is the number of filters in the convolution, kernel_size specifies the size of the convolution window, relu (rectified linear unit) is the activation function, and input_shape specifies a 4D tensor with batch size, height, and width
# for MaxPooling2S, the pool_size is the factors by which to downscale
# for Dropout, the value is the fraction of the input units to drop
# for Dense, the first value is the dimensionality of the output space
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') #softmax transforms the neural network output into a vector of probabilities
])

```
4. Compile the model with an optimizer, loss function, and metrics:
```
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy', # this is softmax loss, which is used for multiclass classification and trains neural networks to output a probability over the N classes for each image.
              metrics=['accuracy'])

```
5. Train the Model using the training data
```
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

```
6. Evaluate the model using test data
```
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

```
Source: https://www.geeksforgeeks.org/training-a-neural-network-using-keras-api-in-tensorflow/

## Getting Started with Privacy
* The link below is to the code used for training the initial model
  * https://github.com/tensorflow/privacy/blob/master/tutorials/walkthrough/mnist_scratch.py
* This model contains two convolutional layers coupled with max pooling layers, a fully-connected layer, and a softmax.
* The model's output is a vector where each component indicates how likely the input is to be in one of the 10 classes of the problem.
```
input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
y = tf.keras.layers.Conv2D(16, 8,
                           strides=2,
                           padding='same',
                           activation='relu').apply(input_layer)
y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
y = tf.keras.layers.Conv2D(32, 4,
                           strides=2,
                           padding='valid',
                           activation='relu').apply(y)
y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
y = tf.keras.layers.Flatten().apply(y)
y = tf.keras.layers.Dense(32, activation='relu').apply(y)
logits = tf.keras.layers.Dense(10).apply(y)
predicted_labels = tf.argmax(input=logits, axis=1)
```
* The `tf.Estimator` API forms minibatches used to train and evaluate the model. The loop is exposed over several epochs of learning (an epoch is one pass over all of the training points included in the training set).
```
steps_per_epoch = 60000 // FLAGS.batch_size
for epoch in range(1, FLAGS.epochs + 1):
  # Train the model for one epoch.
  mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  test_accuracy = eval_results['accuracy']
  print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))
```
* Without privacy, the model should achieve above 99% test accuracy after 15 epochs at a learning rate of 0.15 on minibatches of 256 training points.

## Stochastic Gradient Descent    
* Stochastic gradient descent is an iterative procedure where at each iteration, a batch of data is randomly sampled from the training set. The error between the model's prediction and the training labels (known as the "loss") is computed and differentiated with respect to the model's parameters. These derivatives, or gradients, tell us how each parameter should be updated to bring the model closer to predicting the correct label.
* These steps are repeated for this algorithm...
  1. Sample a minibatch of training points `(x, y)` where `x` is an input and `y` is a label
  2. Compute loss/error `L(theta, x, y)` between the model's prediction `f_theta(x)` and label `y` where `theta` represents model parameters
  3. Compute gradient of the loss `L(theta, x, y)` with respect to the model parameters `theta`.
  4. Multiply these gradients by the learning rate and apply the product to update model parameters `theta`.

## Modifications Needed to Make Stochastic Gradient Descent a Differentially Private Algorithm
* The sensitivity of each gradient needs to be bounded, meaning we must limit how much each individual training point sampled in a minibatch can influence the resulting gradient computation.
* This is done by clipping each gradient computed on each training point, allowing us to bound how much each training point can possibly impact model parameters.
* The algorithm's behavior must be randomized to make it statistically impossible to know whether or not a particular point was included in the training set by comparing the update's stochastic gradient descent applies when it operates with or without a particular point.
* This is done by sampling random noise and adding it to the clipped gradients.
* The differentially private stochastic gradient descent algorithm has the following steps...
  1. Sample a minibatch of training points `(x, y)` where `x` is an input and `y` is a label
  2. Compute loss/error `L(theta, x, y)` between the model's prediction `f_theta(x)` and label `y` where `theta` represents the model parameters
  3. Compute gradient of the loss `L(theta, x, y)` with respect to the model parameters `theta`
  4. Clip gradients, per training example included in the minibatch, to ensure each gradient has a known maximum Euclidean norm
  5. Add random noise to the clipped gradients
  6. Multiply these clipped and noised gradients by the learning rate and apply the product to update model parameters `theta`.

## Implementing DP-SGD with TF Privacy
* TF privacy provides code that wraps an existing TF optimizer to create a variant that performs gradient clipping and noising.
* Forming minibatches of training data and labels is performed by the `tf.Estimator`. Computing loss/error between predictions and labels is done below
```
vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits)
```
* The loss is computed as a vector, where each component corresponds to an individual training point and label, in order to support per example gradient manipulation.
* In TensorFlow, an optimizer object can be instantiated by passing it a learning rate value. Below is the code without differential privacy
```
optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
train_op = optimizer.minimize(loss=scalar_loss)
```
* Below, the `optimizers.dp_optimizer` module implements the optimizer with differential privacy, going through steps 3-6 of the algorithm
```
optimizer = optimizers.dp_optimizer.DPGradientDescentGaussianOptimizer(
    l2_norm_clip=FLAGS.l2_norm_clip,
    noise_multiplier=FLAGS.noise_multiplier,
    num_microbatches=FLAGS.microbatches,
    learning_rate=FLAGS.learning_rate,
    population_size=60000)
train_op = optimizer.minimize(loss=vector_loss)
```
* Here, the stochastic gradient descent optimizer is used. Most optimizers which are objects from a child class of `tf.train.Optimizer` can be made differentially privacy by calling `optimizers.dp_optimizer.make_gaussian_optimizer_class()`
* In addition to the learning rate, the size of the training set is passed as the `population_size` parameter. This is used to measure the strength of privacy achieved.
* TF Privacy introduces three new hyperparameters to the optimizer object....
    * `l2_norm_clip`: maximum Euclidean norm of each individual gradient that is computed on an individual training example from a minibatch. This bounds the optimizer's sensitivity to individual training points (must be a vector loss).
    * `noise_multiplier`: used to control how much noise is sampled and added to gradients before they are applied by the optimizer (more noise is more private, but weakens utility).
    * `num_microbatches`: clipping gradients on a per example basis can be detrimental to performance because computations cannot be batched and parallelized at the granualarity of minibatches. Here, rather than clipping gradients on a per example basis, we clip them on a microbatch basis. For instance, if we have a minibatch of 256 training examples, rather than clipping each of the 256 gradients individually, we clip 32 gradients averaged over microbatches of 8 training examples when `num_microbatches=32`. This allows for parallelism, and trades off performance (when a small value) with utility (when the parameter is close to the minibatch size)
* The hyperparameter values below result in a model with 95% test accuracy.
```
learning_rate=0.25
noise_multiplier=1.3
l2_norm_clip=1.5
batch_size=256
epochs=15
num_microbatches=256
```
## Measuring the Privacy Guarantee Achieved
* It is intuitive to machine learning practitioners how clipping gradients limits the ability of a model to overfit to any of its trainng points.
* Additional randomization is needed to make it hard to tell which behavioral aspects of a model defined by learned parameters came from randomness and which came from the training data.
* With randomness in the training algorithm, we ask "what is the probability that the learning algorithm will choose parameters in this set of possible parameters, when we train it on this specific dataset?"
* We use a version of differential privacy which requires that the probability of learning any particular set of parameters stays about the same if we change (add, remove, or modify) a single training example in the training set. If one point does not ffect the outcome of learning, information contained in that point cannot be memorized, and privacy of the individual who contributed the training point is respected.
* Google's DP library can be used to compute privacy budget spent to train the machine learning model. This allows us to compare two models and determine which of the two is more privacy-preserving than the other.
* First, identify all the parameters that are relevant to measuring possible privacy loss induced by training. These are...
    * `noise_multipler`
    * Sampling ratio `q` (probability of an individual training point being included in a minibatch)
    * `steps` the optimizer takes over the training data.
* `noise_multipler` is reported as provided to the optimizer, and the sampling ratio and number of steps are computed as follows...
```
noise_multiplier = FLAGS.noise_multiplier
sampling_probability = FLAGS.batch_size / 60000
steps = FLAGS.epochs * 60000 // FLAGS.batch_size
```
* At a high level, privacy analysis measures tthe difference between the distributions of model parameters on neighboring training sets.
* TF privacy is performed in the framework of Rényi Differential Privacy (RDP).
* Differential privacy guarantee is expressed using two parameters...
    * `delta`: bounds the probability of our privacy guarantee not holding. Rule of thumb is to set it to be less than the inverse of the training data size (the population size). Here, set it to 10^-5 because MNIST has 60,000 training points
    * `epsilon`: measures the strength of our privacy guarantee. It gives a bound on how much the probability of a particular model output can vary by including (or removing) a single training example. We usually want it to be a small constant.
* To compute privacy spent using the GoogleDP libraru, define two things...
    * `PrivacyAccountant`: specifies what method of privacy accounting will be used (if RDP is used, use the `RdpAccountant`.
    * `DpEvent`: representation of the log of privacy-impacting actions that have occurred (like repeated sampling of records and estimation of their mean with Gaussian noise added).
* To initialize `PrivacyAccountant`, remember the following...
    * There is very little downside in expanding the list of order for which RDP is computed
    * The computed privacy budget is typically not very sensitive to the exact value of the order (close enough works)
    * If you are targeting a particular range of epsilons (like 1-10) and your delta is fixed (like 10^-5), your order must cover the range between `1+ln(1/delta)/10≈2.15` and `1+ln(1/delta)/1≈12.5`. To achieve this, one or two adjustments of the range of orders usually suffices.
```
orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
accountant = privacy_accountant.RdpAccountant(orders)
```
* Next, create a `DPEvent` and feed it to the accountant for processing using its `compose` method
```
event = dp_event.SelfComposedDpEvent(
    event=dp_event.PoissonSampledDpEvent(
        sampling_probability=q,
        event=dp_event.GaussianDpEvent(noise_multiplier)
    ),
    count=steps)
accountant.compose(event)
```
* Finally, query the accountant for best `epsilon` at the given `target_delta` by calling the `get_epsilon` method which takes the minimum over all orders
`epsilon = accountant.get_epsilon(target_delta)`
* Running the code above with the hyperparameter values used during training estimates the `epsilon` value that was achieved by the differentially private optimizer, and thus the strength of the privacy guarantee, which comes with the model we trained.
* To determine effectiveness, insert secrets in the model's training set and measure how likely they are to be leaked by a differentially private model (compared to a non-private one) at inference time.
* Source: https://github.com/tensorflow/privacy/blob/master/tutorials/walkthrough/README.md
