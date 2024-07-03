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

To be able to define a networks architecture without specyfing the number of layers and their sizes, we can make use of so-called `Stack`'s. A `Stack` is a sequence of compatible layers of fixed type `dense` or `conv2d`, optionally with a distinguished final layer of fixed size. 
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
This `ConvDense` class consists of two stacks, one of type `conv2d` and one of type `dense`, initially consisting of two layers of width `5`, and two layers of width `10`, respectively. The latter includes the output layer, which is fixed to size `10` and a softmax activation. The remaining activations and the convolution kernel size are default-initialised to `'relu'` and `5`, respectively, but they could be changed using the keywords `conv2d_activation`, `layer_activation`, and `kernel_size`. The number and size of layers in the stacks are mutable (except for the output layer of the `dense` stack), and the `ConvDense` class internally makes sure that after any such change the flattened output of the last `conv2d` layer aligns with the input size of the first `dense` layer.

Now we can initialise a `Population`, which keeps a pool of neural networks initialised from a given `Architecture`.

```python
from src.trainer import Population

pop = Population(
        architecture_factory,
        population_size = 8,
        ds_train = ds_train,
        ds_val = ds_val,
        train_data_per_epoch = .7,
        mutations_per_generation = 2
        )
```

The `Population` exposes two key functions. The first is `epoch()`, which trains each of the networks in the pool on a random fraction `train_data_per_epoch` of the full training set, and records their performance on the validation set. If multiple cores are availabe, this step is parallelised. The second is `mutate()`, which, in short, kills some of the poor performing networks, duplicates some of the well performing networks, and applies a random number (with mean `mutations_per_generation`) of mutations to every member of the new generation. We go into a bit more detail on the type of mutations in a subsequent section. There are many other parameters that can be passed to a `Population` to fine-tune its behaviour during these two methods. The basic usage of a `Population` is as follows.

```python
epochs = 50

for e in range(epochs):
    pop.mutate()
    pop.epoch()
    pop.print_population()
```

The `print_population()` command prints a summary of the current pool of networks to the console, including their ranking, their structure (i.e. number and size of layers), their training and validation losses and metrics. The population can be saved using `pop.save()`, and either loaded entirely for later use, or for extraction of the top performing network or its weights.

Note that, by default, every `Architecture` is initialised with a small (adjustable) $L^2$ regularisation; this doesn't only regularise the size of individual parameters, but also the size of the network, since larger networks have more parameters. This is what keeps the networks in the population from growing indefinitely.

## Mutations
There are four main kinds of mutations: shrinking or widening a layer (i.e. inserting or deleting individual nodes), and inserting and deleting whole layers. In each of these mutations, except for the deletion of individual nodes, there is a choice to be made about the weights of the mutated network. This should optimally be done in such a way that the outputs of the network immediately before and after the mutation are as close as possible.

### Dense Layers

We can formally describe a dense layer with input dimension $n_\text{in}$ and output dimension (i.e. width) $n_\text{out}$ by a tuple $(W,b)$ where $W\in \mathbb{R}^{n_\text{out}\times n_\text{in}}$ are the weights and $b\in \mathbb{R}^{n_\text{out}}$ are the bias terms.

#### Insertion of Nodes
When a node is inserted, there is a simple way to ensure that the network output is unchanged, which is to set its outgoing weights and bias to zero, and randomly initialise its incoming weights (or the other way around). Formally, if $(W,b)$ describes the mutated layer, $(W',b')$ the following one, and $n_\text{new}$ nodes are inserted, then

$$
    (W,b) \to (W',b') \quad \stackrel{\text{mutation}}{\longrightarrow}\quad (\begin{bmatrix} W \\ W_\text{new}\end{bmatrix}, \begin{bmatrix} b \\ 0_{n_\text{new}}\end{bmatrix}) \to ([ W' \, 0_{n_\text{out}' \times n_\text{new}}], b'),
$$

where $W_\text{new} \in \mathbb{R}^{n_\text{new} \times n_\text{in}}$ is randomly initialised.

#### Insertion of a Layer
If a layer is inserted between two layers $(W,b)$ and $(W',b')$, then we choose its width to be equal to that of the first layer, initialise the weights with the identity matrix, and its bias with zero. That is,

$$
    (W,b) \to (W',b') \quad \stackrel{\text{mutation}}{\longrightarrow}\quad (W,b) \to (I_{n_\text{out} \times n_\text{out}},0_{n_\text{out}}) \to (W',b')
$$

Note that if $x\in \mathbb{R}^{n_\text{in}}$ is the input of $(W,b)$, and our activation function is $f$, then the output of the first layer is $f(Wx + b)$, and the output of the inserted layer is $f(f(Wx + b))$ (here the application of $f$ is component-wise). These are equal if $f\circ f = f$, that is if the activation is idempotent. This is the case for the ReLU activation, which makes it particularly suited to this setting.

#### Deletion of a Layer
We delete a layer by linearising it: suppose we have layers $(W,b) \to (W',b')$, and we replace the first layer's activation function with the identity. Then if the input of the first layer is $x$, the output of the second layer is

$$
    f(W'(Wx + b) + b') = f(W'Wx + W'b + b'),
$$

which is the same as the output of a single layer $(W'W,W'b+b')$ with activation function $f$. Thus our deletion is as follows.

$$
    (W,b) \to (W',b') \quad \stackrel{\text{mutation}}{\longrightarrow}\quad (W' W,W' b + b').
$$

### Convolutional Layers
Convolutional layers are a bit more complicated than dense layers, but the ideas remain the same. Node insertion works just as it did in the dense setting. Layer insertion does as well, and it translates to choosing convolutional kernels of the form

$$
    \begin{bmatrix}
    0 & 0 & 0\\
    0 & 1 & 0\\
    0 & 0 & 0
    \end{bmatrix}
$$

for the newly inserted layer. Deletion of layers is the most interesting. Suppose we have two subsequent convolutional layers of width one (that is, just a single node), the first with an $s \times s$ kernel $F$, the second with an $s' \times s'$ kernel $F'$. If we linearise the first layer, then an input picture $P$ is subsequently convolved with $F$ and then $F'$ before the activation function of the second layer is applied. Writing $G(P)$ for the result of convolving the picture $P$ with a kernel $G$ (with padding), we get

$$
    F'(F(P))(x,y) = \sum_{i',j'} F'(i',j') \sum_{i,j}F(i,j) P(x+i+i',y+j+j') = G(P)(x,y),
$$

where $G = F \star F'$ is the convolution of $F$ with $F'$, that is, a kernel of size $(s+s')\times(s+s')$ defined by

$$
    G(k,l) = \sum_{i,j}F(i,j) F'(k-i,k-j).
$$

Now if the two layers have width $n$ and $n'$ instead of one, and the input size of the first layer is $k$, then they are defined by filters $(F_{pq})_{p\le k,q\le n}$ and $(F'_{pq})_{p\le n,q\le n'}$, and the filters of the linearised layer are 

$$
    G_{pq} = \sum_{r=1}^n F_{pr} \star F_{rq}',\qquad p\le k,q\le n'.
$$
