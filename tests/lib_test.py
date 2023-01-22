from src.lib import pick_from_dict, pick_from_list
import src.lib as lib

import numpy as np
def test_pick_from_list():
    N = 1000000
    n = 10
    l = list(np.random.random(n))
    #d = [(np.random.random(), score) for score in l]

    picks = [0] * n
    for i in range(N):
        picks[pick_from_list(l)] += 1
    picks = [k / N for k in picks]

    EPS = 0.01
    for i in range(n):
        assert np.abs(picks[i] - l[i]/sum(l)) < EPS

def test_pick_from_dict():
    N = 1000000
    n = 10
    l = list(np.random.random(n))
    d = {np.random.random(): score for score in l}

    picks = {key: 0 for key in d.keys()}
    for i in range(N):
        picks[pick_from_dict(d)] += 1
    picks = [k / N for (key, k) in picks.items()]

    EPS = 0.01
    for i in range(n):
        assert np.abs(picks[i] - l[i]/sum(l)) < EPS

from src.lib import Stack
import tensorflow as tf
import tensorflow.keras as keras

def generate_input(shape):
    return np.random.random(shape)

"""
class Stack:
    def __init__(self,
            input_tensor,
            layer_type = \'dense\',
            layer_activation = \'relu\',
            output_size = 1,
            output_activation = \'sigmoid\',
            initial_width = 10,
            initial_depth = 1
            )
    def deep_copy(self, input_tensor = None)
    def build(self)
    def model(self)
    def update_input(self, input_tensor)
"""

def test_stack(verbose = 0, layer_type = 'dense'):
    if layer_type == 'both':
        test_stack(verbose, 'dense')
        test_stack(verbose, 'conv2d')
        return
    n_in = 10
    n_out = 3
    if layer_type == 'conv2d':
        w = 20
        h = 20
        input_shape = (w,h,n_in)
    else:
        input_shape = (n_in,)
    input_layer = keras.Input(shape = input_shape)
    stack = Stack(input_layer, output_size = n_out, layer_type = layer_type)
    
    model = stack.model()
    if verbose:
        model.summary()
    N = 10
    inp = generate_input((N,) + input_shape)
    out = model(inp)
    
    stack2 = stack.deep_copy()
    model2 = stack2.model()
    if verbose:
        model2.summary()
    layer = stack2.layers[0]
    W, b = layer.get_weights()
    layer.set_weights([W, b + 2])
    out2 = model2(inp)

    assert (out2.numpy() != out.numpy()).any()
    assert (out.numpy() == model(inp).numpy()).all()

def test_mutate_stacks(verbose = 0, layer_type = 'dense'):
    if layer_type == 'both':
        test_mutate_stacks(verbose, 'dense')
        test_mutate_stacks(verbose, 'conv2d')
        return
    n_in = 10
    n_out = 3
    n_batch = 100
    if layer_type == 'dense':
        input_shape = (n_in,)
    else:
        w = 20
        h = 20
        input_shape = (w,h,n_in)
    input_layer = keras.Input(shape = input_shape)

    s0 = Stack(input_layer, output_size = n_out, layer_type = layer_type)
    s = s0.deep_copy()

    ###### grow
    if verbose:
        s0.model().summary()
    lib.insert_layer(s,1)
    lib.insert_layer(s,1)
    s1 = s.deep_copy()
    new_ind1 = lib.widen_layer(s, 1, 1.5)
    new_ind0 = lib.widen_layer(s, 0, 2)
    if verbose:
        s.model().summary()

    assert same_output(s0.model(), s1.model(), n_batch)
    assert same_output(s0.model(), s.model(), n_batch)

    s1 = s0.deep_copy()
    lib.insert_layer(s1, 1)
    lib.insert_nodes(s1, 1, [1,3,5], index_mode = 'after')

    s2 = s1.deep_copy()
    lib.delete_nodes(s1, 1, [1,3,5])
    assert same_output(s0.model(), s1.model(), n_batch)

    lib.delete_nodes(s2, 1, [1,2,5])
    assert not same_output(s0.model(), s2.model(), n_batch)

    ###### shrink and delete
    s1 = s0.deep_copy()

    lib.delete_nodes(s,1,new_ind1)
    lib.delete_nodes(s,0,new_ind0)

    if verbose:
        s.model().summary()

    assert same_output(s.model(), s1.model(), n_batch)

    if layer_type == 'conv2d':
        # for conv2d deleting has an error at the boundary; the deleting operation for conv2d layers is instead checked by test_fold_layers()
        return

    lib.delete_layer(s, 1, convolution_mode = 'exact')
    assert same_output(s.model(), s1.model(), n_batch)

    ###### delete is designed such that it has no effect if the activation is linear
    s = Stack(input_layer,
            layer_activation = 'linear',
            output_size = 3,
            output_activation = 'linear',
            initial_depth = 5)
    s0 = s.deep_copy()

    lib.delete_layer(s,2, convolution_mode = 'exact')
    lib.delete_layer(s,2, convolution_mode = 'exact')

    assert same_output(s.model(), s0.model(), n_batch)


from src.lib import Architecture, SimpleArchitecture#, ArchitectureMutator

def test_simple_architecture(verbose = 0, layer_type = 'dense'):
    if layer_type == 'both':
        test_simple_architecture(verbose, 'dense')
        test_simple_architecture(verbose, 'conv2d')
        return
    n_in = 15
    n_out = 3
    if layer_type == 'dense':
        input_shape = (n_in,)
    else:
        w, h = 20, 20
        input_shape = (w,h,n_in)
    arch = SimpleArchitecture(
            input_shape,
            'binary_cross_entropy',
            layer_type = layer_type,
            output_size = n_out,
            initial_depth = 5
            )
    if verbose:
        print(arch.summary())
    m = arch.get_model()
    batch = 13
    out = m(generate_input((batch,) + input_shape))
    if layer_type == 'dense':
        assert out.shape == (batch, n_out)
    else:
        assert out.shape == (batch, w, h, n_out)

    a = SimpleArchitecture(
            input_shape,
            'binary_cross_entropy',
            layer_type = layer_type,
            output_size = n_out,
            output_size_mutable = True
            )
    ###### test mutations and deep copy
    a0 = a.deep_copy()
    print(a.summary())
    for i in range(10):
        a.mutate()
        print(a.summary())
        print('')

"""
class Population:
    def __init__(self,
        architecture_factory,
        population_size,
        x_train,
        y_train,
        x_val,
        y_val,
        train_data_per_epoch = 0.5,
        val_data_per_epoch = 1.0,
        mutations_per_generation = 1.0
        )

    def epoch(self, verbose = 0, mutate_first = True)

    def mutate(self)
"""

from src.trainer import Population
import matplotlib.pyplot as plt

def test_population(verbose = 0):
    def factory():
        n_in = 1
        n_out = 1
        return SimpleArchitecture(
                (n_in,),
                'mse',
                output_size = n_out,
                output_size_mutable = False,
                output_activation = 'linear')
    
    def f(x):
        #return 2*np.cos(np.sqrt(np.abs(x))) + 1
        return 2*np.sin(x)*np.cos(2*x)**2 + 1

    def dataset(size = 100000, vsplit = .1):
        l = -5
        r = 5
        sigma = 0
        N_val = int(vsplit * size)
        N_train = size - N_val
        x_train = l + (r-l) * np.random.random(N_train)
        y_train = f(x_train) + np.random.normal(loc=0,scale=sigma,size=N_train)
        x_val = l + (r-l) * np.random.random(N_val)
        y_val = f(x_val) + np.random.normal(loc=0,scale=sigma,size=N_val)

        
        ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(16)
        ds_val = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(16)

        return ds_train, ds_val

    ds_train, ds_val = dataset()

    population = Population(
            factory,
            10,
            ds_train,
            ds_val,
            train_data_per_epoch = 0.8,
            mutations_per_generation = .8)

    ###### print best one
    def print_population():
        xs = np.arange(-5,5,.1)
        ys = [f(x) for x in xs]
        plt.plot(xs,ys,label='Truth')
        xs_t = tf.expand_dims(tf.convert_to_tensor(xs), 1)
        ranks = [1,5,10]
        for i in ranks:
            model = population.population[i-1].architecture.model
            ys = tf.squeeze(model(xs_t)).numpy()
            plt.plot(xs,ys,label=f'Rank {i}')
        plt.legend()
        plt.show()


    epochs = 50
    for e in range(epochs):
        population.mutate()
        population.epoch(verbose = 1, parallelise = False)
        print_population()


from src.lib import fold_kernels, fold_conv2d_layers
def test_fold_kernels():
    trials = 10
    kernel_dims = [3,5,7] # just some odd numbers
    for i in range(trials):
        shape1 = (np.random.choice(kernel_dims), np.random.choice(kernel_dims))
        shape2 = (np.random.choice(kernel_dims), np.random.choice(kernel_dims))
            
        k1 = np.random.random(shape1)
        k2 = np.random.random(shape2)

        shape = (shape1[0]+shape2[0]-1,shape1[1]+shape2[1]-1)
        subshape = (np.random.choice(list(range(1,shape[0]+1,2))),
                np.random.choice(list(range(1,shape[1]+1,2))))

        k12 = fold_kernels(k1, k2, shape = subshape)
        k21 = fold_kernels(k2, k1, shape = subshape)

        assert k12.shape == subshape
        assert k21.shape == subshape
        assert (np.all(np.abs(k12-k21) < EPS))

def test_fold_layers():
    k1size = 3
    k2size = 3
    n_in = 5
    n_hidden = 6
    n_out = 7
    layer1 = tf.keras.layers.Conv2D(
            n_hidden,
            k1size,
            padding='same',
            activation = 'linear',
            bias_initializer = 'glorot_uniform'
            )
    layer2 = tf.keras.layers.Conv2D(
            n_out,
            k2size,
            padding='same',
            activation = 'linear',
            bias_initializer = 'glorot_uniform'
            )
    w = 20
    h = 20
    input_layer = tf.keras.Input(shape = (w,h,n_in))
    x = input_layer
    x = layer1(x)
    x = layer2(x)
    
    model1 = tf.keras.Model(input_layer,x)

    layer12 = tf.keras.layers.Conv2D(
            n_out,
            k1size+k2size-1,
            padding='same',
            activation = 'linear'
            )
    x = input_layer
    x = layer12(input_layer)

    W1, b1 = layer1.get_weights()
    W2, b2 = layer2.get_weights()
    W, b = fold_conv2d_layers(W1,b1,W2,b2)
    layer12.set_weights([W,b])

    model2 = tf.keras.Model(input_layer,x)

    N = 10
    input_shape = (N,w,h,n_in)
    offset = max(k1size//2,k2size//2) # they will only be equal away from the boundary
    padded_indices = []
    for x in range(offset, w - offset):
        for y in range(offset, h - offset):
            for n in range(N):
                for i in range(n_out):
                    padded_indices.append((n,x,y,i))
    assert same_output(model1, model2, N, index_range = padded_indices)

import os

def test_stack_saveload(verbose = 0, layer_type = 'dense'):
    if layer_type == 'both':
        test_stack_mutator(verbose, 'dense')
        test_stack_mutator(verbose, 'conv2d')
        return
    n_in = 10
    n_out = 3
    n_batch = 100
    if layer_type == 'dense':
        input_shape = (n_in,)
    else:
        w = 20
        h = 20
        input_shape = (w,h,n_in)
    input_layer = keras.Input(shape = input_shape)

    s0 = Stack(input_layer, output_size = n_out, layer_type = layer_type)
    name = 'my stack boobieboo'
    path = os.path.join(os.getcwd(),'saves','stacks',name)
    s0.save(path)
    s1 = Stack.load(path, s0.input)

    s0.model().summary()
    s1.model().summary()

    assert same_output(s0.model(), s1.model())

    lib.shrink_layer(s1, 0, 0.5)

    assert not same_output(s0.model(), s1.model())

from src.architectures import ConvDense
def test_arch_saveload(verbose = 0, layer_type = 'dense'):
    if layer_type == 'both':
        test_architecture_mutator(verbose, 'dense')
        test_architecture_mutator(verbose, 'conv2d')
        return
    n_in = 15
    n_out = 3
    if layer_type == 'dense':
        input_shape = (n_in,)
    else:
        w, h = 20, 20
        input_shape = (w,h,n_in)
    a = ConvDense(
            input_shape,
            'binary_cross_entropy',
            output_size = n_out
            )
    #a = SimpleArchitecture(
    #        input_shape,
    #        'binary_cross_entropy',
    #        layer_type = layer_type,
    #        output_size = n_out,
    #        output_size_mutable = True
    #        )

    ###### test mutations and deep copy
    a0 = a.deep_copy()

    path = os.path.join(os.getcwd(), 'saves', 'architectures', '123haha')

    a0.save(path)
    a1 = Architecture.load(path)
    lib.shrink_layer(a.stacks[0],0,.5)
    a.build()

    assert not same_output(a.get_model(), a0.get_model())
    assert same_output(a1.get_model(), a0.get_model())

EPS = 0.001
def same_output(model1, model2, N = 100, index_range = None):
    if type(model1.input) != list:
        input_shapes = [tuple(model1.input.shape)]
    else:
        input_shapes = [tuple(i.shape) for i in model1.input]

    input_shapes = [(N,) + s[1:] for s in input_shapes]

    inp = [np.random.random(sh) for sh in input_shapes]
    out1 = model1(inp).numpy()
    out2 = model2(inp).numpy()
    if index_range != None:
        for i in index_range:
            if np.abs(out1[i]-out2[i]) >= EPS:
                return False
        return True
    return (np.abs(out1 - out2) < EPS).all()
