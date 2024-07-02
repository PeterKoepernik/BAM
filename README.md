*Note: This code was last maintained in March 2023, and was not tested with more recent versions of TensorFlow.*

## Introduction

The idea of this little project is to automate architecture search with a genetic algorithm. On the one hand, this alleviates the sometimes troublesome task of choosing a good number and size of layers, and secondly, since the networks slowly grow as they train, it might help with issues that can arise when training a very big randomly initialised neural network, rooted in the exploding and vanishing gradient problem. The final motivation is that the simultaneous training of several networks in a population can make great use of parallelism, whereas the training of a single network is hard to parallelise well.

#### Usage Example

The usage of the package is easiest explained with an example. Suppose we want to train a neural network for image classification on the mnist dataset.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

ds_train, ds_val = tfds.load('mnist', split=['train','test'], shuffle_files=True)

def process(el):
    return (el['image'], el['label'])

ds_train = ds_train.map(process).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(process).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
```

To be able to define a networks architecture without specyfing the number and size of each layer, we can make use of so-called `Stack`'s. A `Stack` is a sequence of compatible layers of fixed type `dense` or `conv2d`, optionally with a distinguished final layer of fixed size. 
The abstract shape of a convolutional neural network for the MNIST classification task could look like this:
```
input (28x28x1) -> Stack (conv2d) -> flatten -> Stack (dense) -> output (size 10, softmax)
```
In general, to specify such a network template we need to subclass the `Architecture` class, but the structure above is already implemented in the `ConvDense` architecture, which can be initialised as follows. For later use, we have to wrap it into a factory function.
```python
from src.architectures import ConvDense

def architecture_factory():
    return ConvDense(
            input_shape = (28,28,1),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'],
            output_size = 10,
            output_activation = 'softmax',
            initial_width = [5,10],
            initial_depth = [2,2],
            )
```
The `ConvDense` class consists of two stacks, one of type `conv2d` and one of type `dense`, initially consisting of two layers of width `5`, and two layers of width `10`, respectively. The latter includes the output layer, which is fixed to size `10` and a softmax activation. The remaining activations and convolution kernel size are default-initialised to `'relu'` and `5`, respectively, but they could be changed using the keywords `conv2d_activation`, `layer_activation`, and `kernel_size`. The number and size of layers in the stacks are mutable, and the `ConvDense` class internally makes sure that after any such change the flattened output of the last `conv2d` layer aligns with the input size to the first `dense` layer.

Now we can initialise a `Population`, which keeps a pool of neural networks initialised from a given `Architecture`.

```python
from src.trainer import Population

pop = Population(
        architecture_factory,
        pop_size = 8,
        ds_train,
        ds_val,
        train_data_per_epoch = .7,
        mutations_per_generation = 2
        )
```

The `Population` exposes to main functions. The first is `epoch()`, which trains each of the networks in the pool on a random fraction `train_data_per_epoch` of the full training set, and records their performance on the validation set. If multiple cores are availabe, this step is parallelised. The second is `mutate()`, which, in short, kills some of the poor performing networks, duplicates some of the well performing networks, and randomly mutates every member of the new generation. We go into a bit more detail on the type of mutations in a subsequent section.

```python
epochs = 50

for e in range(epochs):
    pop.mutate()
    pop.epoch()
    pop.print_population()
```

The `print_population()` command prints a summary of the current pool of networks to the console, including their ranking, their structure (i.e. number and size of layers), their training and validation losses and metrics. The population can be saved using `pop.save()`, and either loaded entirely for later use, or for extraction of the top performing network or its weights.

## Mutations
There are four main kinds of mutations: shrinking, widening, or deleting existing layers, and inserting new layers.
