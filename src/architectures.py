import os
if os.path.join(os.getcwd(),'src') == os.path.dirname(os.path.realpath(__file__)):
    ###### we are being called from within this package; if we are called from terminal, BAM/src is already in the global searchpath
    import src.lib as lib

from lib import Architecture, Stack, flatten_node_changes, widen_input, shrink_input
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
        changed_flat_nodes = flatten_node_changes(self.stacks[0], changed_nodes, post_shrink = mutation_type == 'shrink')

        if mutation_type == 'widen':
            ###### now we need to check which new nodes there are after the flattening operation
            widen_input(self.stacks[1], changed_flat_nodes, index_mode = 'after')
        if mutation_type == 'shrink':
            shrink_input(self.stacks[1], changed_flat_nodes)

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
                initial_width = initial_width[1],
                initial_depth = initial_depth[1]
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

        changed_flat_nodes = flatten_node_changes(self.stacks[0], changed_nodes, post_shrink = mutation_type == 'shrink')

        if mutation_type == 'widen':
            widen_input(self.stacks[1], changed_flat_nodes, index_mode = 'after')
        elif mutation_type == 'shrink':
            shrink_input(self.stacks[1], changed_flat_nodes)

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
                string += str(self.stacks[i].output_size) + f'({self.output_activation}) / (l2reg: {round(self.stacks[0].l2reg,4)}/{round(self.stacks[1].l2reg,4)})'
        return string

class ValueArchSimple(Architecture):
    """
    No convolutional network, just symmetrised dense network
    """
    def __init__(
            self,
            num_stats,
            activation = 'tanh',
            initial_width = 2,
            initial_depth = 2
            ):

        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.num_stats = num_stats
        self.activation = activation
        self.initial_width = initial_width
        self.initial_depth = initial_depth

        self.input1 = tf.keras.Input(shape = (num_stats,))
        self.input2 = tf.keras.Input(shape = (num_stats,))

        self.concat = tf.keras.layers.Concatenate()

        stack = Stack(
                input_tensor = self.concat([self.input1, self.input2]),
                layer_type = 'dense',
                layer_activation = self.activation,
                output_size = 1,
                output_activation = 'sigmoid',
                initial_width = initial_width,
                initial_depth = initial_depth,
                l2reg = 0.0
                )
        self.stacks = [stack]

        self.build()
        super().__init__()

    def picklify(self, directory):
        self.input1 = None
        self.input2 = None
        self.concat = None
        return []

    def unpicklify(self, pickles):
        self.input1 = tf.keras.Input(shape = (self.num_stats,))
        self.input2 = tf.keras.Input(shape = (self.num_stats,))
        self.concat = tf.keras.layers.Concatenate()

    def build(self):
        inp1 = self.concat([self.input1,self.input2])
        inp2 = self.concat([self.input2,self.input1])

        self.stacks[0].build(inp1)
        out1 = self.stacks[0].output

        self.stacks[0].build(inp2)
        out2 = self.stacks[0].output

        out = (out1 + 1 - out2) / 2

        self.model = tf.keras.Model([self.input1, self.input2], out)

    def shallow_copy(self):
        return ValueArchSimple(
            num_stats = self.num_stats,
            activation = self.activation,
            initial_width = self.initial_width,
            initial_depth = self.initial_depth
            )

    def deep_copy(self):
        copy = self.shallow_copy()
        input_tensor = copy.concat([copy.input1, copy.input2])
        copy.stacks = [self.stacks[0].deep_copy(input_tensor)]
        copy.build()
        return copy

    def summary(self):
        """ returns a list of hidden layer sizes """
        string = 'in-'
        for l in self.stacks[0].layers[:-1]:
            string += str(l.output.shape[-1])
            string += '-'
        string += str(self.stacks[0].output_size) + f'(sigmoid) / (l2reg: {round(self.stacks[0].l2reg,4)})'

        return string

class ValueArchConv(Architecture):
    """
    Symmetrised dense network, and convolutional layers
    """

    def __init__(
            self,
            num_stats,
            dense_activation = 'relu',
            conv2d_activation = 'relu',
            initial_width = [3, 6],
            initial_depth = [3, 3],
            kernel_size = 5
            ):

        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.num_stats = num_stats
        self.dense_activation = dense_activation
        self.conv2d_activation = conv2d_activation
        self.initial_width = initial_width
        self.initial_depth = initial_depth
        self.kernel_size = kernel_size

        self.denseinput1 = tf.keras.Input(shape = (num_stats,))
        self.denseinput2 = tf.keras.Input(shape = (num_stats,))
        self.planeinput1 = tf.keras.Input(shape = (28,14,6))
        self.planeinput2 = tf.keras.Input(shape = (28,14,6))

        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

        convstack = Stack(
                input_tensor = self.concat([self.planeinput1, self.planeinput2]),
                layer_type = 'conv2d',
                layer_activation = self.conv2d_activation,
                distinguished_output = False,
                initial_width = initial_width[0],
                initial_depth = initial_depth[0],
                kernel_size = self.kernel_size
                )
        x = convstack.build()
        x = self.flatten(x)
        dense_input = self.concat([self.denseinput1,self.denseinput2])
        x = self.concat([x, dense_input])
        
        outstack = Stack(
                input_tensor = x,
                layer_type = 'dense',
                layer_activation = self.dense_activation,
                output_size = 1,
                output_activation = 'sigmoid',
                initial_width = initial_width[1],
                initial_depth = initial_depth[1]
                )
        self.stacks = [convstack, outstack]

        self.build()

        super().__init__()

    def picklify(self, directory):
        self.denseinput1 = None
        self.denseinput2 = None
        self.planeinput1 = None
        self.planeinput2 = None
        self.concat = None
        self.flatten = None
        return []

    def unpicklify(self, pickles):
        self.denseinput1 = tf.keras.Input(shape = (self.num_stats,))
        self.denseinput2 = tf.keras.Input(shape = (self.num_stats,))
        self.planeinput1 = tf.keras.Input(shape = (28,14,6))
        self.planeinput2 = tf.keras.Input(shape = (28,14,6))

        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()

    def build(self):
        convin1 = self.concat([self.planeinput1, self.planeinput2])
        convin2 = self.concat([self.planeinput2, self.planeinput1])

        densein1 = self.concat([self.denseinput1,self.denseinput2])
        densein2 = self.concat([self.denseinput2,self.denseinput1])

        ###### (1,2)
        x = self.stacks[0].build(convin1)
        x = self.flatten(x)
        x = self.concat([x, densein1])
        out1 = self.stacks[1].build(x)

        ###### (2,1)
        x = self.stacks[0].build(convin2)
        x = self.flatten(x)
        x = self.concat([x, densein2])
        out2 = self.stacks[1].build(x)

        ###### symmetrise
        out = (out1 + 1 - out2) / 2

        self.model = tf.keras.Model(
                [self.denseinput1, self.denseinput2, self.planeinput1, self.planeinput2],
                out)

    def mutate(self, num_mutations = 1):
        for i in range(num_mutations):
            info_list = super().mutate(num_mutations = 1, build = False)
            assert len(info_list) == 1
            stack_index, infos = info_list[0]
            self.adapt_layers(stack_index, infos)

    def adapt_layers(self, stack_index, infos, build = True):
        mutation_type, infos = infos
        if stack_index == 1 or mutation_type not in ['widen', 'shrink']:
            return

        layer_index, changed_nodes = infos
        if layer_index + 1 < len(self.stacks[0].layers):
            return

        changed_flat_nodes = flatten_node_changes(self.stacks[0], changed_nodes, post_shrink = mutation_type == 'shrink')

        if mutation_type == 'widen':
            widen_input(self.stacks[1], changed_flat_nodes, index_mode = 'after')
        elif mutation_type == 'shrink':
            shrink_input(self.stacks[1], changed_flat_nodes)

        if build:
            self.build()

    def shallow_copy(self):
        return ValueArchConv(
            num_stats = self.num_stats,
            dense_activation = self.dense_activation,
            conv2d_activation = self.conv2d_activation,
            initial_width = self.initial_width,
            initial_depth = self.initial_depth,
            kernel_size = self.kernel_size
            )

    def deep_copy(self):
        copy = self.shallow_copy()
        conv_in = copy.concat([copy.planeinput1, copy.planeinput2])
        copy.stacks[0] = self.stacks[0].deep_copy(conv_in)
        dense_in = copy.concat([copy.flatten(copy.stacks[0].output), copy.concat([copy.denseinput1, copy.denseinput2])])
        copy.stacks[1] = self.stacks[1].deep_copy(dense_in)
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
                string += str(self.stacks[i].output_size) + f'(sigmoid) - (l2reg: {round(self.stacks[0].l2reg,4)}/{round(self.stacks[1].l2reg,4)})'
        return string

class PriorArchSimple(Architecture):
    """
    Takes (x0,xl,xr) where x0 are inputs with no left/right association (like mp or hp or turn), and xl and xr have same length one for left and one for right (for example turret sp on the left/right side). Then we have one set of dense layers that map to scores
    (x0,xl,xr) -> (s0,sl,sr)
    (x0,xr,xl) -> (s0',sl',sr')
    Here s0 are actions that have no side (like idle; or when we dont distinguish), and sl/sr are actions with a side (like scouts left or scouts right)
    Then the final scores are
    (s0+s0', sl + sr', sr + sl')
    This way the output is a homeomorphism wrt flipping l/r.
    """
    def __init__(
            self,
            num_stats0, # stats that are not positional (hp,mp,turn etc)
            num_statslr, # like sp on the left
            num_outputs0,
            num_outputslr,
            activation = 'relu',
            initial_width = 2,
            initial_depth = 2,
            l2reg = 0.002
            ):

        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.num_stats0 = num_stats0
        self.num_statslr = num_statslr
        self.num_outputs0 = num_outputs0
        self.num_outputslr = num_outputslr
        self.activation = activation
        self.initial_width = initial_width
        self.initial_depth = initial_depth
        self.l2reg = l2reg

        self.input0 = tf.keras.Input(shape = (num_stats0,))
        self.inputl = tf.keras.Input(shape = (num_statslr,))
        self.inputr = tf.keras.Input(shape = (num_statslr,))

        self.concat = tf.keras.layers.Concatenate()

        self.ind0 = list(range(num_outputs0))
        self.indl = list(range(num_outputs0,num_outputs0+num_outputslr))
        self.indr = list(range(num_outputs0+num_outputslr, num_outputs0+2*num_outputslr))

        stack = Stack(
                input_tensor = self.concat([self.input0, self.inputl, self.inputr]),
                layer_type = 'dense',
                layer_activation = self.activation,
                output_size = num_outputs0 + 2 * num_outputslr,
                output_activation = activation,
                initial_width = initial_width,
                initial_depth = initial_depth,
                l2reg = l2reg
                )
        self.stacks = [stack]

        self.build()
        super().__init__()

    def picklify(self, directory):
        self.input0 = None
        self.inputl = None
        self.inputr = None
        self.concat = None
        return []

    def unpicklify(self, pickles):
        self.input0 = tf.keras.Input(shape = (self.num_stats0,))
        self.inputl = tf.keras.Input(shape = (self.num_statslr,))
        self.inputr = tf.keras.Input(shape = (self.num_statslr,))
        self.concat = tf.keras.layers.Concatenate()

    def build(self):
        inp1 = self.concat([self.input0,self.inputl,self.inputr])
        inp2 = self.concat([self.input0,self.inputr,self.inputl])

        self.stacks[0].build(inp1)
        out1 = self.stacks[0].output

        self.stacks[0].build(inp2)
        out2 = self.stacks[0].output

        ###### now we combine them in the funny way; first dimension is batch dimension
        s0 = tf.gather(out1, self.ind0, axis=1) + tf.gather(out2, self.ind0, axis=1)
        sl = tf.gather(out1, self.indl, axis=1) + tf.gather(out2, self.indr, axis=1)
        sr = tf.gather(out1, self.indr, axis=1) + tf.gather(out2, self.indl, axis=1)
        scores = self.concat([s0,sl,sr])

        out = tf.nn.softmax(scores)

        self.model = tf.keras.Model([self.input0, self.inputl, self.inputr], out)

    def shallow_copy(self):
        return PriorArchSimple(
            num_stats0 = self.num_stats0,
            num_statslr = self.num_statslr,
            num_outputs0 = self.num_outputs0,
            num_outputslr = self.num_outputslr,
            activation = self.activation,
            initial_width = self.initial_width,
            initial_depth = self.initial_depth,
            l2reg = self.l2reg
            )

    def deep_copy(self):
        copy = self.shallow_copy()
        input_tensor = copy.concat([copy.input0, copy.inputl, copy.inputr])
        copy.stacks = [self.stacks[0].deep_copy(input_tensor)]
        copy.build()
        return copy

    def summary(self):
        """ returns a list of hidden layer sizes """
        string = 'in-'
        for l in self.stacks[0].layers:
            string += str(l.output.shape[-1])
            string += '-'
        string += 'symmetrise-softmax'
        string += f' (l2reg: {round(self.stacks[0].l2reg,4)})'

        return string

