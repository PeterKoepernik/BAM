from src.lib import Architecture, Stack
import src.lib as lib
import tensorflow as tf
import numpy as np

class ConvDense(Architecture):
    def __init__(
            self,
            input_shape,
            loss,
            metrics = [],
            conv2d_activation = 'relu',
            layer_activation = 'relu',
            output_size = 1,
            output_activation = 'sigmoid',
            initial_width = [5,5],
            initial_depth = [2,2], #includes output layer
            kernel_size = 5
            ):

        self.input_shape = input_shape
        self.loss = loss
        self.metrics = metrics
        self.conv2d_activation = conv2d_activation
        self.layer_activation = layer_activation
        self.output_size = output_size
        self.output_activation = output_activation
        self.initial_width = initial_width
        self.initial_depth = initial_depth
        self.kernel_size = kernel_size

        self.input_layer = tf.keras.Input(shape=input_shape)

        stack_conv2d = Stack(
                input_tensor = self.input_layer,
                layer_type = 'conv2d',
                layer_activation = conv2d_activation,
                distinguished_output = False,
                #output_size = initial_width[0],
                #output_activation = conv2d_activation,
                #output_size_mutable = True,
                initial_width = initial_width[0],
                initial_depth = initial_depth[0],
                kernel_size = kernel_size
                )

        stack_conv2d.build()

        self.flatten = tf.keras.layers.Flatten()

        x = stack_conv2d.output
        x = self.flatten(x)

        stack_dense = Stack(
                input_tensor = x,
                layer_type = 'dense',
                layer_activation = layer_activation,
                output_size = output_size,
                output_activation = output_activation,
                output_size_mutable = False,
                initial_width = initial_width[1],
                initial_depth = initial_depth[1]
                )
        stack_dense.build()

        self.stacks = [stack_conv2d, stack_dense]

        self.build()

        super().__init__()

    def picklify(self, directory):
        self.input_layer = None
        self.flatten = None
        return []

    def unpicklify(self, pickles):
        self.input_layer = tf.keras.Input(shape=self.input_shape)
        self.flatten = tf.keras.layers.Flatten()

    def post_pickle_restore(self, directory):
        self.unpicklify([])

    def build(self):
        self.stacks[0].build(self.input_layer)
        x = self.flatten(self.stacks[0].output)
        self.stacks[1].build(x)
        self.model = tf.keras.Model(self.input_layer, self.stacks[1].output)

    def compile(self):
        self.build()
        self.model.compile(optimizer = 'adam', loss = self.loss, metrics = self.metrics)

    def mutate(self, num_mutations = 1):
        for i in range(num_mutations):
            info_list = super().mutate(num_mutations = 1, build = False)
            assert len(info_list) == 1
            stack_index, infos = info_list[0]
            #self.adapt_layers(stack_index, infos, build = False)
            self.adapt_layers(stack_index, infos)

        #self.build()
            
    def adapt_layers(self, stack_index, infos, build = True):
        """
        If a mutation changes the output size of the first stack, the input size of the second stack has to be adapted accordingly
        """
        mutation_type, infos = infos
        if stack_index == 1:
            return
        if mutation_type not in ['widen', 'shrink']:
            return

        ###### only have to do something if output size of conv2d stack has changed
        layer_index, changed_nodes = infos # changed_nodes is either new_nodes or old_nodes
        if layer_index + 1 < len(self.stacks[0].layers):
            return

        ###### adapt_input(out_layer = l, changed_nodes, delete/widen continue here then make architectures more modular
        changed_flat_nodes = lib.flatten_node_changes(self.stacks[0], changed_nodes, post_shrink = mutation_type == 'shrink')

        if mutation_type == 'widen':
            ###### now we need to check which new nodes there are after the flattening operation
            lib.widen_input(self.stacks[1], changed_flat_nodes, index_mode = 'after')
        if mutation_type == 'shrink':
            lib.shrink_input(self.stacks[1], changed_flat_nodes)

        if build:
            self.build()

    def shallow_copy(self):
        return ConvDense(
                input_shape = self.input_shape,
                loss = self.loss,
                metrics = self.metrics,
                conv2d_activation = self.conv2d_activation,
                layer_activation = self.layer_activation,
                output_size = self.output_size,
                output_activation = self.output_activation,
                initial_width = self.initial_width,
                initial_depth = self.initial_depth,
                kernel_size = self.kernel_size
                )

    def deep_copy(self):
        copy = self.shallow_copy()
        copy.stacks = []
        x = copy.input_layer
        s0 = self.stacks[0].deep_copy(x)
        copy.stacks.append(s0)
        x = copy.flatten(s0.output)
        s1 = self.stacks[1].deep_copy(x)
        copy.stacks.append(s1)
        copy.build()
        return copy

    def summary(self):
        """ returns a list of hidden layer sizes """
        string = 'in-'
        for i in range(2):
            if i == 1:
                string += 'flatten-'
            for l in self.stacks[i].layers[:len(self.stacks[i].layers)-i]:
                string += str(l.output.shape[-1])
                if i == 0:
                    kernel_size = l.get_weights()[0].shape[0]
                    string += f'({kernel_size}x{kernel_size})'
                string += '-'
            if i == 1:
                string += str(self.stacks[i].output_size) + f'({self.output_activation})'
        return string

class ConvDenseMixed(Architecture):
    def __init__(
            self,
            dense_input_shape,
            cnn_input_shape,
            loss,
            metrics = [],
            cnn_activation = 'relu',
            dense_activation = 'relu',
            output_size = 1,
            output_activation = 'sigmoid',
            initial_width = [5,5],
            initial_depth = [2,2],
            kernel_size = 5
            ):

        self.dense_input_shape = dense_input_shape
        self.cnn_input_shape = cnn_input_shape
        self.loss = loss
        self.metrics = metrics
        self.cnn_activation = cnn_activation
        self.dense_activation = dense_activation
        self.output_size = output_size
        self.output_activation = output_activation
        self.initial_width = initial_width
        self.initial_depth = initial_depth
        self.kernel_size = kernel_size

        self.dense_input_layer = tf.keras.Input(shape = dense_input_shape)
        self.cnn_input_layer = tf.keras.Input(shape = cnn_input_shape)

        stack_cnn = Stack(
                input_tensor = self.cnn_input_layer,
                layer_type = 'conv2d',
                layer_activation = cnn_activation,
                distinguished_output = False,
                initial_width = initial_width[0],
                initial_depth = initial_depth[0],
                kernel_size = kernel_size
                )
        stack_cnn.build()

        self.flatten = tf.keras.layers.Flatten()
        self.concatenate = tf.keras.layers.Concatenate()

        x = stack_cnn.output
        x = self.flatten(x)
        x = self.concatenate([x,self.dense_input_layer])

        stack_dense = Stack(
                input_tensor = x,
                layer_type = 'dense',
                layer_activation = dense_activation,
                output_size = output_size,
                output_activation = output_activation,
                output_size_mutable = False,
                initial_width = initial_width[0],
                initial_depth = initial_depth[0]
                )
        stack_dense.build()

        self.stacks = [stack_cnn, stack_dense]

        self.build()

        super().__init__()

    def picklify(self, directory):
        self.dense_input_layer = None
        self.cnn_input_layer = None
        self.flatten = None
        self.concatenate = None
        return []

    def unpicklify(self, pickles):
        self.dense_input_layer = tf.keras.Input(shape = self.dense_input_shape)
        self.cnn_input_layer = tf.keras.Input(shape = self.cnn_input_shape)
        self.flatten = tf.keras.layers.Flatten()
        self.concatenate = tf.keras.layers.Concatenate()

    def post_pickle_restore(self, directory):
        self.unpicklify([])

    def build(self):
        self.stacks[0].build(self.cnn_input_layer)
        x = self.flatten(self.stacks[0].output)
        x = self.concatenate([x, self.dense_input_layer])
        self.stacks[1].build(x)
        self.model = tf.keras.Model([self.dense_input_layer, self.cnn_input_layer], self.stacks[1].output)

    def compile(self):
        self.build()
        self.model.compile(optimizer = 'adam', loss = self.loss, metrics = self.metrics)

    def mutate(self, num_mutations = 1):
        for i in range(num_mutations):
            info_list = super().mutate(num_mutations = 1, build = False)
            assert len(info_list) == 1
            stack_index, infos = info_list[0]
            self.adapt_layers(stack_index, infos)
            #self.adapt_layers(stack_index, infos, build = False)

        #self.build()

    def adapt_layers(self, stack_index, infos, build = True):
        mutation_type, infos = infos
        if stack_index == 1 or mutation_type not in ['widen', 'shrink']:
            return

        layer_index, changed_nodes = infos
        if layer_index + 1 < len(self.stacks[0].layers):
            return

        changed_flat_nodes = lib.flatten_node_changes(self.stacks[0], changed_nodes, post_shrink = mutation_type == 'shrink')

        if mutation_type == 'widen':
            lib.widen_input(self.stacks[1], changed_flat_nodes, index_mode = 'after')
        elif mutation_type == 'shrink':
            lib.shrink_input(self.stacks[1], changed_flat_nodes)

        if build:
            self.build()

    def shallow_copy(self):
        return ConvDenseMixed(
            dense_input_shape = self.dense_input_shape,
            cnn_input_shape = self.cnn_input_shape,
            loss = self.loss,
            metrics = self.metrics,
            cnn_activation = self.cnn_activation,
            dense_activation = self.dense_activation,
            output_size = self.output_size,
            output_activation = self.output_activation,
            initial_width = self.initial_width,
            initial_depth = self.initial_depth,
            kernel_size = self.kernel_size
            )

    def deep_copy(self):
        copy = self.shallow_copy()
        copy.stacks = []
        x = copy.cnn_input_layer
        s0 = self.stacks[0].deep_copy(x)
        copy.stacks.append(s0)
        x = copy.flatten(s0.output)
        x = copy.concatenate([x,copy.dense_input_layer])
        s1 = self.stacks[1].deep_copy(x)
        copy.stacks.append(s1)
        copy.build()
        return copy

    def summary(self):
        """ returns a list of hidden layer sizes """
        string = 'in-'
        for i in range(2):
            if i == 1:
                string += 'flatten-concatenate-'
            for l in self.stacks[i].layers[:len(self.stacks[i].layers)-i]:
                string += str(l.output.shape[-1])
                if i == 0:
                    kernel_size = l.get_weights()[0].shape[0]
                    string += f'({kernel_size}x{kernel_size})'
                string += '-'
            if i == 1:
                string += str(self.stacks[i].output_size) + f'({self.output_activation})'
        return string
