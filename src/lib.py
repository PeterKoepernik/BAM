import tensorflow as tf
import numpy as np
import copy as cpy
import random, os

MAX_SCORE = 1000000
MAX_BIAS_FACTOR = 5 # the most likely mutation is at most this much more likely than the most unlikely

def update_bias(biases, index, factor, max_relative_factor):
    if type(biases) == list:
        min_entry = min([val for val in biases if val > 0])
    elif type(biases) == dict:
        min_entry = min([val for val in biases.values() if val > 0])
    else:
        raise Exception('update_bias can only be called on lists and dictionaries')

    biases[index] = min(factor * biases[index], max_relative_factor * min_entry)

    if biases[index] > MAX_SCORE:
        if type(biases) == list:
            for i in range(len(biases)):
                biases[i] = .5 * biases[i]
        else:
            for key in biases.keys():
                biases[key] = .5 * biases[key]


#class StackMutator:
#    """
#    Sits on a stack and has functions to mutate it in various ways
#    """
#
#    def __init__(self, stack):
#        raise Exception('StackMutator is DEPRECATED')
#        self.stack = stack
#
#        ###### mutation bias (linearly weighted) scores that determine how likely a mutation is to change the width of some layer, or insert a layer
#        self.mutation_bias = {
#                'widen': 3,
#                'insert': 2,
#                'shrink': 1,
#                'delete': 1
#                }
#
#        ###### for each operation a list with a score per layer, higher score making it (linearly) more likely to mutate at this index
#        self.layer_bias = {
#                'widen': [1] * (len(self.stack.layers)-1) + [int(self.stack.output_size_mutable)],
#                'insert': [0] + ([1] * (len(self.stack.layers)-1)),
#                'shrink': [1] * (len(self.stack.layers)-1) + [int(self.stack.output_size_mutable)],
#                'delete': [0] + [1] * (len(self.stack.layers)-2) + [0] # can't delete output layer or first layer
#                }
#
#        self.layer_widen_factor = 1.2 # 20% increase in layer size when mutating
#        self.layer_shrink_factor = 1 / 1.2
#        self.bias_momentum = 1 # (formerly 1.2) # when performing a mutation, this mutation is this much more likely to be chosen next time
#
#    def print_bias(self):
#        print('Mutation bias:')
#        for key, val in self.mutation_bias.items():
#            print(f'{key}: {val}')
#        print('')
#
#        print('Layer bias:')
#        for key, l in self.layer_bias.items():
#            print(f'{key}: ' + str(l))
#
#    def mutate(self):
#        """
#        returns list of (mutation_type, info) where info is layer_index for insert/delete and (layer_index, new/old_nodes) for shrink/widen
#        """
#        ###### pick operation
#        #print(f'Called mutate.')
#        #self.print_bias()
#        op = pick_from_dict(self.mutation_bias)
#        while sum(self.layer_bias[op]) == 0:
#            op = pick_from_dict(self.mutation_bias)
#
#        layer_index = pick_from_list(self.layer_bias[op])
#        #print(f'Picked operation {op} and layer {layer_index}')
#
#        ###### update bias
#        update_bias(self.mutation_bias, op, self.bias_momentum, MAX_BIAS_FACTOR)
#        update_bias(self.layer_bias[op], layer_index, self.bias_momentum, MAX_BIAS_FACTOR)
#
#        ###### perform operation
#        if op == 'widen':
#            new_nodes = self.widen_layer(layer_index, self.layer_widen_factor)
#            info = (layer_index, new_nodes)
#        elif op == 'insert':
#            self.insert_layer(layer_index)
#            for key in self.layer_bias.keys():
#                bias = self.layer_bias[key][layer_index] if len(self.layer_bias[key]) > 0 else 1
#                bias = max(1,bias)
#                self.layer_bias[key].insert(layer_index, bias)
#            info = layer_index
#        elif op == 'shrink':
#            deleted_nodes = self.shrink_layer(layer_index, self.layer_shrink_factor)
#            info = (layer_index, deleted_nodes)
#        elif op == 'delete':
#            self.delete_layer(layer_index)
#            for key in self.layer_bias.keys():
#                self.layer_bias[key].pop(layer_index)
#            info = (layer_index, deleted_nodes)
#        else:
#            raise Exception('Tried to mutate unknown operation.')
#
#        return op, info
#
#    def deep_copy(self, stack = None):
#        if stack == None:
#            stack = self.stack.deep_copy()
#
#        mutator_copy = StackMutator(stack)
#        mutator_copy.mutation_bias = cpy.deepcopy(self.mutation_bias)
#        mutator_copy.layer_bias = cpy.deepcopy(self.layer_bias)
#        mutator_copy.layer_widen_factor = self.layer_widen_factor
#        mutator_copy.layer_shrink_factor = self.layer_shrink_factor
#
#        return mutator_copy

import pickle

def save_object(obj, filename):
    if os.path.exists(filename):
        print(f'Warning: file \'{filename}\' already exists. Aborting.')

    try:
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    except:
        print('Something went wrong saving an object.')
        print(f'All Attributes: {self.__dir__()}')
        assert False

def load_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

class Stack:
    """
    A `stack` is a sequence of compatible layers of a fixed type (dense or conv2d).
    """
    LAYER_TYPES = ['dense', 'conv2d']

    def __init__(self,
            input_tensor,
            layer_type = 'dense',
            layer_activation = 'relu',
            distinguished_output = True, # if False the following three will be ignored
            output_size = 1,
            output_activation = 'relu',
            output_size_mutable = False,
            initial_width = 10,
            initial_depth = 2, # includes output layer
            l2reg = .01, # this will also be mutated
            kernel_size = 3 # only used for conv2d
            ):
        """
        If layer type is `dense`, input_tensor is assumed to be flat.
        """
        if layer_type not in Stack.LAYER_TYPES:
            raise Exception(f'Layer type {layer_type} not supported.')

        self.input = input_tensor
        self.layer_type = layer_type
        self.layer_activation = layer_activation
        self.distinguished_output = distinguished_output
        self.output_size = output_size
        self.output_activation = output_activation
        self.output_size_mutable = output_size_mutable
        self.initial_width = initial_width
        self.initial_depth = initial_depth
        self.l2reg = l2reg
        self.kernel_size = kernel_size

        self.layers = []
        for d in range(initial_depth):
            out_layer = distinguished_output and (d == initial_depth - 1)
            if self.layer_type == 'dense':
                self.layers.append(tf.keras.layers.Dense(
                    output_size if out_layer else initial_width,
                    activation = output_activation if out_layer else layer_activation,
                    kernel_regularizer = tf.keras.regularizers.L2(l2reg)
                    ))
            elif self.layer_type == 'conv2d':
                self.layers.append(tf.keras.layers.Conv2D(
                    output_size if out_layer else initial_width,
                    kernel_size,
                    activation = output_activation if out_layer else layer_activation,
                    padding = 'same',
                    kernel_regularizer = tf.keras.regularizers.L2(l2reg)
                    ))
            else: assert False

        ###### mutation bias (linearly weighted) scores that determine how likely a mutation is to change the width of some layer, or insert a layer
        self.mutation_bias = {
                'widen': 2,
                'insert': 1,
                'shrink': 2,
                'delete': 1,
                'change_reg': 2 if (self.l2reg > 0) else 0 # if we were initialised without regularisation there is no point changing it
                }

        ###### for each operation a list with a score per layer, higher score making it (linearly) more likely to mutate at this index
        self.layer_bias = {
                'widen': [1] * (len(self.layers)-1) + [int(self.output_size_mutable or not self.distinguished_output)],
                'insert': [0] + ([1] * (len(self.layers)-int(self.distinguished_output))),
                'shrink': [1] * (len(self.layers)-1) + [int(self.output_size_mutable or not self.distinguished_output)],
                'delete': [0] + [1] * (len(self.layers)-2) + [1-int(self.distinguished_output)] # can't delete first layer, last layer only if not distinguished
                }

        self.layer_widen_factor = 1.2 # 20% increase in layer size when mutating
        self.layer_shrink_factor = 1 / 1.2
        self.bias_momentum = 1 # (formerly 1.2) # when performing a mutation, this mutation is this much more likely to be chosen next time

        self.change_reg_factor = 1.5

        self.build()

    def load(directory, input_tensor = None):
        if not os.path.exists(directory):
            raise Exception('Given directory does not exist.' )

        weight_path = os.path.join(directory, 'weights.npz')
        obj_path = os.path.join(directory, 'stack.pkl')

        if not os.path.exists(weight_path):
            raise Exception('weights.npy does not exist.')
        if not os.path.exists(obj_path):
            raise Exception('stack.pkl does not exist.')

        stack = load_object(obj_path)

        ###### for legacy reasons; some old stacks don't have this after being loaded
        if not hasattr(stack, 'l2reg'):
            stack.l2reg = 0.0

        ###### now stack has empty layers and input = None
        if input_tensor != None:
            stack.input = input_tensor
        
        with open(weight_path, 'rb') as f:
            weights = np.load(f)

            if input_tensor != None:
                x = stack.input
            for i in range(len(weights.files) // 2):
                W = weights[f'W{i}']
                b = weights[f'b{i}']
                n_out = W.shape[-1]
                n_in = W.shape[-2]

                out_layer = stack.distinguished_output and (i == (len(weights) // 2) - 1)
                if stack.layer_type == 'dense':
                    layer = tf.keras.layers.Dense(
                            n_out,
                            activation = stack.output_activation if out_layer else stack.layer_activation,
                            kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                            )
                    layer.build(input_shape = (1, n_in))
                elif stack.layer_type == 'conv2d':
                    layer = tf.keras.layers.Conv2D(
                        n_out,
                        W.shape[0],
                        activation = stack.output_activation if out_layer else stack.layer_activation,
                        padding = 'same',
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
                    layer.build(input_shape = (1, 1, 1, n_in))
                else: assert False
                if input_tensor != None:
                    x = layer(x)
                layer.set_weights([W,b])
                stack.layers.append(layer)

        if input_tensor != None:
            stack.output = x

        return stack

    def save(self, directory):
        """
        directory should be a folder name
        """
        if not os.path.exists(directory):
            os.mkdir(directory)

        weight_path = os.path.join(directory, 'weights.npz')
        obj_path = os.path.join(directory, 'stack.pkl')

        weights = {}
        for i, l in enumerate(self.layers):
            W, b = l.get_weights()
            weights[f'W{i}'] = W
            weights[f'b{i}'] = b

        with open(weight_path, 'wb') as f:
            np.savez(f, **weights)

        tmplayers = self.layers
        tmpinp = self.input
        tmpout = self.output
        self.layers = []
        self.input = None
        self.output = None

        save_object(self, obj_path)

        self.layers = tmplayers
        self.input = tmpinp
        self.output = tmpout

    def mutate(self):
        """
        returns (mutation_type, info) where info is layer_index for insert/delete and (layer_index, new/old_nodes) for shrink/widen
        """
        ###### pick operation
        #print(f'Called mutate.')
        #self.print_bias()
        op = pick_from_dict(self.mutation_bias)
        while op != 'change_reg' and sum(self.layer_bias[op]) == 0:
            op = pick_from_dict(self.mutation_bias)

        update_bias(self.mutation_bias, op, self.bias_momentum, MAX_BIAS_FACTOR)

        if op == 'change_reg':
            if random.random() > .5:
                self.l2reg = self.l2reg * self.change_reg_factor
            else:
                self.l2reg = self.l2reg / self.change_reg_factor
            return op, None

        ###### update bias
        layer_index = pick_from_list(self.layer_bias[op])
        update_bias(self.layer_bias[op], layer_index, self.bias_momentum, MAX_BIAS_FACTOR)

        #print(f'Picked operation {op} and layer {layer_index}')

        ###### perform operation
        if op == 'widen':
            new_nodes = widen_layer(self, layer_index, self.layer_widen_factor)
            info = (layer_index, new_nodes)
        elif op == 'insert':
            insert_layer(self, layer_index)
            for key in self.layer_bias.keys():
                bias = self.layer_bias[key][layer_index-1] if len(self.layer_bias[key]) > 0 else 1
                bias = max(1,bias)
                self.layer_bias[key].insert(layer_index, bias)
            info = layer_index
        elif op == 'shrink':
            deleted_nodes = shrink_layer(self, layer_index, self.layer_shrink_factor)
            info = (layer_index, deleted_nodes)
        elif op == 'delete':
            delete_layer(self, layer_index)
            for key in self.layer_bias.keys():
                self.layer_bias[key].pop(layer_index)
            info = layer_index
        else:
            raise Exception('Tried to perform unknown mutation.')

        return op, info

    def build(self, input_tensor = None):
        """
        optionally with new input_tensor
        """
        if input_tensor != None:
            self.input = input_tensor
        x = self.input
        for l in self.layers:
            x = l(x)
        self.output = x

        return self.output

    def model(self):
        return tf.keras.Model(self.input, self.output)

#    def update_input(self, input_tensor):
#        """ replace by build(input_tensor) """
#        #assert self.input.shape == input_tensor.shape
#        self.input = input_tensor
#        self.build()
       
    def deep_copy(self, input_tensor = None):
        """
        if input_tensor = None the copy will have the same input tensor
        """
        if input_tensor == None:
            input_tensor = self.input

        stack_copy = Stack(input_tensor, 
                layer_type = self.layer_type,
                layer_activation = self.layer_activation,
                distinguished_output = self.distinguished_output,
                output_size = self.output_size,
                output_activation = self.output_activation,
                output_size_mutable = self.output_size_mutable,
                initial_width = self.initial_width,
                initial_depth = self.initial_depth,
                l2reg = self.l2reg,
                kernel_size = self.kernel_size
                )

        ###### mutation biases
        stack_copy.mutation_bias = cpy.deepcopy(self.mutation_bias)
        stack_copy.layer_bias = cpy.deepcopy(self.layer_bias)
        stack_copy.layer_widen_factor = self.layer_widen_factor
        stack_copy.layer_shrink_factor = self.layer_shrink_factor
        stack_copy.bias_momentum = self.bias_momentum

        ###### Copy layers
        stack_copy.layers = []

        x = input_tensor
        for l in self.layers:
            _, n = layer_dims(l)
            if self.layer_type == 'dense':
                new_layer = tf.keras.layers.Dense(
                        n,
                        activation = l.activation,
                        kernel_regularizer = tf.keras.regularizers.L2(self.l2reg)
                        )
            elif self.layer_type == 'conv2d':
                k = kernel_size(l)
                new_layer = tf.keras.layers.Conv2D(
                        n,
                        k,
                        activation = l.activation,
                        padding = 'same',
                        kernel_regularizer = tf.keras.regularizers.L2(self.l2reg)
                        )
            else: assert False
            x = new_layer(x)
            new_layer.set_weights(l.get_weights())
            stack_copy.layers.append(new_layer)

        stack_copy.build()

        return stack_copy

    def print_bias(self):
        print('Mutation bias:')
        for key, val in self.mutation_bias.items():
            print(f'{key}: {val}')
        print('')

        print('Layer bias:')
        for key, l in self.layer_bias.items():
            print(f'{key}: ' + str(l))


class Architecture:
    """
    An `Architecture` defines a set of stacks, and how a model is built from it.
    """

    def __init__(self):
        self.stack_bias = [1] * len(self.stacks)
        self.bias_momentum = 1 # no bias change for now (formerly 1.2)

    def get_model(self):
        return self.model

    def compile(self):
        self.build()
        self.model.compile(optimizer = 'adam', loss = self.loss, metrics = self.metrics, run_eagerly=True)
        return self.model

    def summary(self):
        pass

    def print_weights(self):
        print('\n===== Printing weights of Model =====')
        self.summary()
        print('\n==== WEIGHTS ====')
        for i, stack in enumerate(self.stacks):
            print(f'\n=== STACK {i} ===')
            print(f'Layer type: {stack.layer_type}')
            print(f'Layer activation: {stack.layer_activation}')
            if stack.distinguished_output:
                print(f'Output activation: {stack.output_activation}')
            for j, layer in enumerate(stack.layers):
                print(f'\n== LAYER {j} ==')
                W, b = layer.get_weights()
                if len(W.shape) == 2:
                    n_in, n_out = W.shape
                    print(f'n_in = {n_in}\nn_out = {n_out}')
                elif len(W.shape) == 4:
                    k, l, n_in, n_out = W.shape
                    print(f'n_in = {n_in}\nn_out = {n_out}')
                    print(f'kernel = {k}x{l}')
                print(f'W = {W}')
                print(f'b = {b}')
        print('\n===== DONE =====')

    def build(self):
        pass
    
    def shallow_copy(self):
        """ Copies all internal parameters, but leaves stacks of copy uninitialised """
        pass

    def deep_copy(self):
        """ shallow copy + deep copy of all stacks """
        copy = self.shallow_copy()

        copy.stack_bias = cpy.deepcopy(self.stack_bias)
        copy.bias_momentum = self.bias_momentum

        copy.stacks = [stack.deep_copy() for stack in self.stacks]#[m.stack for m in mutator_copies]
        copy.build()

        return copy

    def picklify(self, directory):
        """
        Return a list of unpicklable instance attributes and replace them by picklable ones, and save them in the directory. Restore with unpicklify()
        """
        raise Exception('picklify() not implemented.')

    def unpicklify(self, pickles):
        raise Exception('unpicklify() not implemented.')

    def post_pickle_restore(self, directory):
        """
        Restore all instance attributes that were picklified earlier. Those are the ones not covered by the save/load of the parent class.
        """
        self.unpicklify([]) ###### default implementation

    def save(self, directory):
        """
        directory should be a folder name
        """
        if not os.path.exists(directory):
            os.mkdir(directory)

        stack_paths = [os.path.join(directory, f'stack{i}') for i in range(len(self.stacks))]
        for path, stack in zip(stack_paths, self.stacks):
            stack.save(path)

        tmp_stacks = self.stacks
        self.stacks = []
        self.model = None
        pickles = self.picklify(directory)

        obj_path = os.path.join(directory, 'arch.pkl')
        save_object(self, obj_path)

        self.stacks = tmp_stacks
        self.unpicklify(pickles)

        self.build()

    def load(directory):
        if not os.path.exists(directory):
            raise Exception('Given directory not found.')

        def path(i):
            return os.path.join(directory, f'stack{i}')

        stacks = []
        i = 0
        while os.path.exists(path(i)):
            stacks.append(Stack.load(path(i)))
            i += 1

        obj_path = os.path.join(directory, 'arch.pkl')
        arch = load_object(obj_path)
        arch.stacks = stacks

        ###### now it has everything except the objects that were picklified
        arch.post_pickle_restore(directory)
        arch.build()

        return arch

    def mutate(self, num_mutations = 1, build = True):
        """
        returns list [(stack_index, infos)]
        where infos is as returned by Stack.mutate()
        """
        res = []
        for i in range(num_mutations):
            ###### pick stack
            stack_index = pick_from_list(self.stack_bias)

            ###### update bias
            update_bias(self.stack_bias, stack_index, self.bias_momentum, MAX_BIAS_FACTOR)

            ###### perform mutation
            infos = self.stacks[stack_index].mutate()

            res.append((stack_index, infos))

        if build:
            self.build()

        return res


class SimpleArchitecture(Architecture):
    def __init__(
            self,
            input_shape,
            loss,
            metrics = [],
            layer_type = 'dense',
            layer_activation = 'relu',
            distinguished_output = True, #if false the following three are ignored
            output_size = 1,
            output_activation = 'sigmoid',
            output_size_mutable = False,
            initial_width = 10,
            initial_depth = 2,
            kernel_size = 3
            ):

        assert initial_width >= 1
        assert initial_width >= 1
        assert output_size >= 1

        self.input_shape = input_shape
        self.loss = loss
        self.metrics = metrics
        self.layer_type = layer_type
        self.layer_activation = layer_activation
        self.distinguished_output = distinguished_output
        self.output_size = output_size
        self.output_activation = output_activation
        self.output_size_mutable = output_size_mutable
        self.initial_width = initial_width
        self.initial_depth = initial_depth
        self.kernel_size = kernel_size
 
        ###### set up stack and initial layer
        self.input_layer = tf.keras.Input(shape = input_shape)

        if len(self.input_shape) > 1 and self.layer_type == 'dense':
            self.flatten = True
            self.input_layer = tf.keras.layers.Flatten()(self.input_layer)
        else:
            self.flatten = False

        stack = Stack(
                input_tensor = self.input_layer,
                layer_type = self.layer_type,
                layer_activation = layer_activation,
                distinguished_output = distinguished_output,
                output_size = output_size,
                output_activation = output_activation,
                output_size_mutable = output_size_mutable,
                initial_width = initial_width,
                initial_depth = initial_depth,
                kernel_size = self.kernel_size
                )

        self.stacks = [stack]

        self.build()

        super().__init__()

    def picklify(self, directory):
        self.input_layer = None
        return []

    def unpicklify(self, pickles):
        self.input_layer = tf.keras.Input(shape = self.input_shape)
        if self.flatten:
            self.input_layer = tf.keras.layers.Flatten()(self.input_layer)

    def post_pickle_restore(self, pickles):
        self.unpicklify([])

    def compile(self):
        self.build()
        self.model.compile(optimizer = 'adam', loss = self.loss, metrics = self.metrics)

    def shallow_copy(self):
        return SimpleArchitecture(
                self.input_shape,
                self.loss,
                self.metrics,
                self.layer_type,
                self.layer_activation,
                self.distinguished_output,
                self.output_size,
                self.output_activation,
                self.output_size_mutable,
                self.initial_width,
                self.initial_depth,
                self.kernel_size
                )

    def deep_copy(self):
        copy = self.shallow_copy()
        copy.stacks = [self.stacks[0].deep_copy(copy.input_layer)]
        copy.build()
        return copy

    def build(self):
        self.output_size = self.stacks[0].output_size
        self.stacks[0].build(self.input_layer)
        self.model = self.stacks[0].model()
        #self.stack.build()
        #self.model = tf.keras.Model(self.stack.input, self.stack.output)

    def summary(self):
        """ returns a list of hidden layer sizes """
        layer_sizes = [layer_dims(l)[1] for l in self.stacks[0].layers]
        string = 'in-'
        if self.flatten:
            string += 'flatten-'
        for l in self.stacks[0].layers[:-1]:
        #for l in layer_sizes[:-1]:
            string += str(layer_dims(l)[1])
            if self.layer_type == 'conv2d':
                k = kernel_size(l)
                string += f'({k}x{k})'
            string += '-'
        string += str(self.stacks[0].output_size) + f'({self.output_activation})'
        return string

""" Helper Functions """

def pick_from_dict(d):
    """
    Takes a dictionary (key, score) and returns a key with probability linear in score.
    """
    Z = sum(d.values())
    x = np.random.random() * Z
    for key, score in d.items():
        if x < score:
            return key
        x -= score
    assert False

def pick_from_list(l):
    """
    Takes a list of scores and returns an index with probability linear in score.
    """
    Z = sum(l)
    x = np.random.random() * Z
    for i, score in enumerate(l):
        if x < score:
            return i
        x -= score
    assert False

def _add(*ts):
    """
    adds tuples element wise
    """
    if len(ts) == 1:
        return ts[0]
    elif len(ts) > 2:
        return _add(_add(ts[0],ts[1]), *(ts[2:]))
    else:
        t1, t2 = ts
        assert type(t1) == type(t2)
        assert len(t1) == len(t2)
        return tuple([t1[i] + t2[i] for i in range(len(t1))])

def fold_kernels(k1, k2, shape = None):
    """
    If shape (of new kernel) is None, it is taken so large that folding with the new kernel is identical to folding with k1 and then k2 (which is actually symmetric)
    """
    shape1 = k1.shape
    shape2 = k2.shape
    assert len(shape1) == 2 and len(shape2) == 2
    if shape == None:
        shape = _add(shape1, shape2, (-1,-1))

    def _padded(arr, ind):
        assert type(ind) == tuple
        if np.any([i < 0 for i in ind]) or np.any([ind[j] >= arr.shape[j] for j in range(len(ind))]):
            return 0
        return arr[ind]
    def _get(arr, ind):
        """
        assumes arr is a numpy array with odd dimensions, and i assigns arr centred, i.e. (0,0,...) is the middle element
        """
        assert sum([s % 2 == 0 for s in arr.shape]) == 0
        widths = tuple([s // 2 for s in arr.shape])
        return _padded(arr, _add(ind,widths))
    def _set(arr, ind, val):
        """
        centred assignment
        """
        assert sum([s % 2 == 0 for s in arr.shape]) == 0
        widths = tuple([s // 2 for s in arr.shape])
        arr[_add(ind,widths)] = val

    kernel = np.zeros(shape)
    for k in range(-(shape[0]//2),shape[0]//2+1):
        for l in range(-(shape[1]//2),shape[1]//2+1):
            for i in range(-(shape1[0]//2),shape1[0]//2+1):
                for j in range(-(shape1[1]//2),shape1[1]//2+1):
                    _set(kernel, (k,l), _get(kernel,(k,l)) + _get(k1,(i,j)) * _get(k2,(k-i,l-j)))

    return kernel

def fold_conv2d_layers(W1, b1, W2, b2, kernel_size = 'sum'):
    """
    kernel_size options:
        integer
        'sum': full convolution
        'max': size is larger of the two previous kernels
        'min': same but min
        'avg': avg of the kernels
    """
    w1, h1, in1, out1 = W1.shape
    w2, h2, in2, out2 = W2.shape
    if type(kernel_size) == int:
        assert kernel_size > 0 and kernel_size <= min(w1+w2-1,h1+h2-1)
        sh = (kernel_size,kernel_size)
    elif kernel_size == 'sum':
        sh = (w1+w2-1,h1+h2-1,in1,out2)
    elif kernel_size == 'max':
        sh = (max(w1,w2),max(h1,h2))
    elif kernel_size == 'min':
        sh = (min(w1,w2),min(h1,h2))
    elif kernel_size == 'avg':
        sh = ((w1+w2)//2,(h1+h2)//2)
    else:
        assert False
    W = np.zeros(sh + (in1,out2))
    assert out1 == in2
    b = np.copy(b2)
    for j in range(out2):
        for k in range(out1):
            for i in range(in1):
                W[:,:,i,j] += fold_kernels(W1[:,:,i,k], W2[:,:,k,j], shape = sh)
            b[j] += np.sum(W2[:,:,k,j]) * b1[k]

    return W, b

############## Layer Operations ##############

def flatten_node_changes(output, changed_nodes, post_shrink = False):
    """
    output is either a stack, a layer, or layer.output. IF IT IS A LAYER/STACK IT NEEDS TO HAVE A WELL-DEFINED AND UP-TO-DATE OUTPUT (this is maintained by mutations but not by widen/shrink_input. This should not be a problem because widen/shrink input is currently only ever applied to dense layers, and this only to cnn layers)

    If a tuple, assumes that output_shape is the shape of a conv2d layer, with or without first batch dimension.

    post_shrink means that the output_shape is the output shape of a layer after changed_nodes have been deleted from it. In that case the output_shape will be changed to what it must have been before the shrink operation.

    """
    if issubclass(type(output), Stack):
        output_shape = output.layers[-1].output.shape
    elif issubclass(type(output), tf.keras.layers.Layer):
        output_shape = output.output.shape
    else:
        ###### now it should be a tensor or tuple
        output_shape = output

    if len(output_shape) == 3:
        w, h, n_out = output_shape
    elif len(output_shape) == 4:
        _, w, h, n_out = output_shape
    else:
        raise Exception('Given output_shape not supported.')

    if post_shrink:
        n_out += len(changed_nodes)

    X = np.zeros((w,h,n_out), dtype = int)
    X[:,:,changed_nodes] = np.ones((w,h,len(changed_nodes)))
    Y = X.reshape(-1) ###### flatten does the same as numpy reshape
    changed_flat_nodes = [i for (i,y) in enumerate(Y) if y == 1]

    return changed_flat_nodes


def shrink_input(stack, deleted_nodes):
    if len(deleted_nodes) == 0:
        return

    layer = stack.layers[0]
    n_in, n_out = layer_dims(layer)
    W, b = layer.get_weights()

    new_in = n_in - len(deleted_nodes)

    for ind in deleted_nodes:
        assert ind >= 0 and ind < n_in

    kept_nodes = [i for i in range(n_in) if i not in deleted_nodes]

    if stack.layer_type == 'dense':
        new_layer = tf.keras.layers.Dense(
                n_out,
                activation = layer.activation,
                kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                )
        new_layer.build(input_shape = (1,new_in))
        W2 = W[kept_nodes,:]
        new_layer.set_weights([W2, b])
    elif stack.layer_type == 'conv2d':
        new_layer = tf.keras.layers.Conv2D(
                n_out,
                W.shape[0], # kernel size
                activation = layer.activation,
                padding = 'same',
                kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                )
        new_layer.build(input_shape = (1,1,1,new_in))
        W2 = W[:,:,kept_nodes,:]
        new_layer.set_weights([W2, b])
    else: assert False
    
    stack.layers[0] = new_layer

def widen_input(stack, new_nodes, index_mode = 'prior'):
    """
    For index_mode see insert_nodes. Adds new inputs (with zero weights) to the first layer.
    """

    if len(new_nodes) == 0:
        return

    ###### transform to index_mode = 'after'
    if index_mode == 'prior':
        new_nodes = [ind + i for (i, ind) in enumerate(new_nodes)]

    layer = stack.layers[0]
    n_in, n_out = layer_dims(layer)
    W, b = layer.get_weights()

    new_in = n_in + len(new_nodes)

    for ind in new_nodes:
        assert ind >= 0 and ind < new_in

    old_nodes = [i for i in range(new_in) if i not in new_nodes]

    if stack.layer_type == 'dense':
        new_layer = tf.keras.layers.Dense(
                n_out,
                activation = layer.activation,
                kernel_initializer = tf.keras.initializers.zeros,
                kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                )
        new_layer.build(input_shape = (1, new_in))
        W2, b2 = new_layer.get_weights()
        W2[old_nodes,:] = W
        new_layer.set_weights([W2, b2])
    elif stack.layer_type == 'conv2d':
        new_layer = tf.keras.layers.Conv2D(
                n_out,
                W.shape[0], # kernel size
                activation = layer.activation,
                padding = 'same',
                kernel_initializer = tf.keras.initializers.zeros,
                kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                )
        new_layer.build(input_shape = (1,1,1,new_in))
        W2, b2 = new_layer.get_weights()
        W2[:,:,old_nodes,:] = W
        new_layer.set_weights([W2, b2])
    else: assert False
    
    stack.layers[0] = new_layer

def widen_layer(stack, layer_id, factor):
    assert factor > 1
    assert layer_id >= 0 and layer_id < len(stack.layers) 

    _, n_out = layer_dims(stack.layers[layer_id])
    extra_nodes = max(1, int(factor*n_out) - n_out)
    new_indices = random.sample(range(n_out+extra_nodes),extra_nodes)
    new_indices.sort()
    #new_indices = list(range(n_out,n_out+extra_nodes))
    insert_nodes(stack, layer_id, new_indices, index_mode = 'after')
    return new_indices

def insert_nodes(stack, layer_id, node_ids, index_mode = 'prior'):
    """
    Insert nodes at given indices;

        index_mode = prior: index = i means the node will be inserted before current node with index i. Repitions mean that more than one node will be inserted before a given node.
        index_mode = after: index = i means the new node will have index i after all insertion operations; that is, after the insertion operation we recover the original state by deleting nodes with given indices
    """
    assert layer_id >= 0 and layer_id < len(stack.layers) 
    if stack.distinguished_output and layer_id == len(stack.layers) - 1 and not stack.output_size_mutable:
        raise Exception('Cannot widen immutable output layer.')

    if len(node_ids) == 0:
        return

    ###### transform to index_mode = 'after'
    if index_mode == 'prior':
        node_ids = [ind + i for (i, ind) in enumerate(node_ids)]

    n_in, n_out = layer_dims(stack.layers[layer_id])
    new_width = n_out + len(node_ids)

    for ind in node_ids:
        assert ind >= 0 and ind < new_width

    old_nodes = [i for i in range(new_width) if i not in node_ids]

    x = stack.input

    for i in range(len(stack.layers)):
        if i == layer_id:
            # create new layer
            if stack.layer_type == 'dense':
                new_layer = tf.keras.layers.Dense(
                        new_width,
                        activation = stack.layers[i].activation,
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            elif stack.layer_type == 'conv2d':
                new_layer = tf.keras.layers.Conv2D(
                        new_width,
                        kernel_size(stack.layers[i]),
                        activation = stack.layers[i].activation,
                        padding = 'same',
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            else: assert False

            x = new_layer(x)

            # copy weights from old layer
            new_weights, new_bias = new_layer.get_weights()
            old_weights, old_bias = stack.layers[i].get_weights()
            if stack.layer_type == 'dense':
                new_weights[:,old_nodes] = old_weights
            elif stack.layer_type == 'conv2d':
                new_weights[:,:,:,old_nodes] = old_weights
            else: assert False
            new_bias[old_nodes] = old_bias
            new_layer.set_weights([new_weights, new_bias])

            stack.layers[i] = new_layer

            if layer_id == len(stack.layers)-1: # this was output layer
                stack.output_size = new_width
            else: # create new after layer with larger input
                if stack.layer_type == 'dense':
                    _, n = layer_dims(stack.layers[i+1])
                    out_layer = tf.keras.layers.Dense(
                            n,
                            activation = stack.layers[i+1].activation,
                            kernel_initializer = tf.keras.initializers.zeros,
                            kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                            )
                elif stack.layer_type == 'conv2d':
                    _, n = layer_dims(stack.layers[i+1])
                    k = kernel_size(stack.layers[i+1])
                    out_layer = tf.keras.layers.Conv2D(
                            n,
                            k,
                            activation = stack.layers[i+1].activation,
                            kernel_initializer = tf.keras.initializers.zeros,
                            padding = 'same',
                            kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                            )
                else: assert False
                x = out_layer(x)

                old_weights, old_bias = stack.layers[i+1].get_weights()
                new_weights, new_bias = out_layer.get_weights()
                if stack.layer_type == 'dense':
                    new_weights[old_nodes,:] = old_weights
                elif stack.layer_type == 'conv2d':
                    new_weights[:,:,old_nodes,:] = old_weights
                else: assert False
                out_layer.set_weights([new_weights, old_bias])
                stack.layers[i+1] = out_layer

        elif i != layer_id + 1: # skip layer after widened one
            x = stack.layers[i](x)

    stack.output = x

def conv2d_identity(shape, dtype = None):
    assert len(shape) == 4 # conv2d weights should have dimensions (kernelsize,kernelsize,n_in,n_out)
    assert shape[2] == shape[3]

    kernel_w = shape[0]
    kernel_h = shape[1]
    assert kernel_w % 2 == 1 # only works for odd kernel sizes
    assert kernel_h % 2 == 1
    mid_w = kernel_w // 2
    mid_h = kernel_h // 2

    kernel = np.zeros(shape)
    for i in range(shape[2]):
        kernel[mid_w,mid_h,i,i] = 1
    return tf.convert_to_tensor(kernel, dtype = dtype)

def insert_layer(stack, layer_id):
    assert layer_id > 0 and layer_id < len(stack.layers) + 1
    if stack.distinguished_output:
        assert layer_id < len(stack.layers) # can't insert behind output layer

    _, n = layer_dims(stack.layers[layer_id-1])
    if stack.layer_type == 'dense':
        new_layer = tf.keras.layers.Dense(
                n,
                activation = stack.layer_activation,
                kernel_initializer = tf.keras.initializers.identity,
                kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                )
    elif stack.layer_type == 'conv2d':
        k = kernel_size(stack.layers[layer_id-1])
        new_layer = tf.keras.layers.Conv2D(
                n,
                k,
                activation = stack.layer_activation,
                kernel_initializer = conv2d_identity,
                padding = 'same',
                kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                )
    else: assert False
   
    stack.layers.insert(layer_id, new_layer)
    stack.build()

def delete_layer(stack, layer_id, convolution_mode = 'performance'):
    """
    Deletes a layer (not the output layer). If the deleted layer has parameters (A,b) and the subsequent layer has (A',b'), then the subsequent layers weights are replaced by (A*A',A'*b + b') (ie if the deleted layer had an identity activation it would not change the output).
    """
    if convolution_mode == 'exact':
        kernel_size = 'sum'
    elif convolution_mode == 'performance':
        kernel_size = 'min'
    else:
        raise Exception(f'Unknown convolution mode {convolution_mode}.')

    assert layer_id >= 0 and layer_id < len(stack.layers)

    if stack.distinguished_output:
        assert layer_id < len(stack.layers) - 1 # can't delete output layer

    x = stack.input

    i = 0
    while i < len(stack.layers):
        if i == layer_id-1:
            l_prev = stack.layers[i]
            l_cur = stack.layers[i+1]

            W_prev, b_prev = l_prev.get_weights()
            W, b = l_cur.get_weights() # these are numpy arrays; W.shape = (in,out)

            _, n = layer_dims(l_cur)
            if stack.layer_type == 'dense':
                newW = W_prev@W
                newb = b + b_prev@W

                new_layer = tf.keras.layers.Dense( # layer that will replace l_prev
                        n,
                        activation = l_prev.activation,
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            elif stack.layer_type == 'conv2d':
                newW, newb = fold_conv2d_layers(W_prev,b_prev,W,b,kernel_size = kernel_size)
                new_layer = tf.keras.layers.Conv2D(
                        n,
                        newW.shape[0],
                        activation = l_prev.activation,
                        padding = 'same',
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            else: assert False
            x = new_layer(x)
            new_layer.set_weights([newW, newb])
            stack.layers.pop(i+1)
            stack.layers[i] = new_layer
        else:
            x = stack.layers[i](x)
        i += 1

    stack.output = x


def delete_layer_old(stack, layer_id):
    """
    Deletes a layer (not the output layer). If the deleted layer has parameters (A,b) and the subsequent layer has (A',b'), then the subsequent layers weights are replaced by (A*A',A'*b + b') (ie if the deleted layer had an identity activation it would not change the output).
    """
    assert layer_id >= 0 and layer_id < len(stack.layers) - 1 # can't delete output layer

    x = stack.input

    i = 0
    while i < len(stack.layers):
        if i == layer_id:
            l_cur = stack.layers[i]
            l_next = stack.layers[i+1]

            W, b = l_cur.get_weights() # these are numpy arrays; W.shape = (in,out)
            W2, b2 = l_next.get_weights()

            _, n = layer_dims(l_next)
            if stack.layer_type == 'dense':
                newW = W@W2
                newb = b2 + b@W2

                new_layer = tf.keras.layers.Dense( # layer that will replace l_next
                        n,
                        activation = l_next.activation,
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            elif stack.layer_type == 'conv2d':
                newW, newb = fold_conv2d_layers(W,b,W2,b2, kernel_size = 'min')
                new_layer = tf.keras.layers.Conv2D(
                        n,
                        newW.shape[0],
                        activation = l_next.activation,
                        padding = 'same',
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            else: assert False
            x = new_layer(x)
            new_layer.set_weights([newW, newb])
            stack.layers.pop(i)
            stack.layers[i] = new_layer
        else:
            x = stack.layers[i](x)
        i += 1

    stack.output = x

def shrink_layer(stack, layer_id, factor):
    assert factor > 0 and factor < 1
    assert layer_id >= 0 and layer_id < len(stack.layers)

    _, n_out = layer_dims(stack.layers[layer_id])
    new_width = max(1,min(n_out-1,int(factor * n_out)))
    deleted_nodes = random.sample(range(new_width,n_out), n_out-new_width)
    deleted_nodes.sort()
    #deleted_nodes = list(range(new_width,n_out)) 
    delete_nodes(stack, layer_id, deleted_nodes)
    return deleted_nodes

def delete_nodes(stack, layer_id, node_ids):
    """
    deletes nodes with given ids in given layer
    """
    assert layer_id >= 0 and layer_id < len(stack.layers)
    if stack.distinguished_output and layer_id == len(stack.layers)-1 and not stack.output_size_mutable:
        raise Exception('Cannot shrink immutable output layer.')

    if len(node_ids) == 0:
        return

    n_in, n_out = layer_dims(stack.layers[layer_id])
    new_width = n_out - len(node_ids)

    assert len(set(node_ids)) == len(node_ids)
    assert new_width > 0
    for ind in node_ids:
        assert ind >= 0 and ind < n_out

    surviving_nodes = [i for i in range(n_out) if i not in node_ids]

    x = stack.input

    for i in range(len(stack.layers)):
        if i == layer_id:
            # create new layer
            if stack.layer_type == 'dense':
                new_layer = tf.keras.layers.Dense(
                        new_width,
                        activation = stack.layers[i].activation,
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            elif stack.layer_type == 'conv2d':
                new_layer = tf.keras.layers.Conv2D(
                        new_width,
                        kernel_size(stack.layers[i]),
                        activation = stack.layers[i].activation,
                        padding = 'same',
                        kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                        )
            else: assert False

            x = new_layer(x)

            # choose the deleted nodes
            W, b = stack.layers[i].get_weights()
            if stack.layer_type == 'dense':
                newW = W[:,surviving_nodes]
            elif stack.layer_type == 'conv2d':
                newW = W[:,:,:,surviving_nodes]
            else: assert False
            newb = b[surviving_nodes]
            new_layer.set_weights([newW, newb])
            stack.layers[i] = new_layer

            if layer_id < len(stack.layers) - 1: # create new after layer with larger input
                _, n = layer_dims(stack.layers[i+1])
                if stack.layer_type == 'dense':
                    out_layer = tf.keras.layers.Dense(
                            n,
                            activation = stack.layers[i+1].activation,
                            kernel_initializer = tf.keras.initializers.zeros,
                            kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                            )
                elif stack.layer_type == 'conv2d':
                    k = kernel_size(stack.layers[i+1])
                    out_layer = tf.keras.layers.Conv2D(
                            n,
                            k,
                            activation = stack.layers[i+1].activation,
                            kernel_initializer = tf.keras.initializers.zeros,
                            padding='same',
                            kernel_regularizer = tf.keras.regularizers.L2(stack.l2reg)
                            )
                else: assert False
                x = out_layer(x)
                W, b = stack.layers[i+1].get_weights()
                if stack.layer_type == 'dense':
                    newW = W[surviving_nodes,:]
                elif stack.layer_type == 'conv2d':
                    newW = W[:,:,surviving_nodes,:]
                else: assert False
                newb = b
                out_layer.set_weights([newW, newb])
                stack.layers[i+1] = out_layer

        elif i != layer_id + 1: # skip layer after widened one
            x = stack.layers[i](x)

    stack.output = x

def layer_dims(layer):
    """
    returns n_in, n_out for a layer that has been built
    """
    sh = layer.get_weights()[0].shape # shape of weight matrix
    return sh[-2], sh[-1]

def kernel_size(layer):
    """
    returns kernel size of a conv2d layer
    """
    sh = layer.get_weights()[0].shape
    assert len(sh) == 4
    return sh[0]
