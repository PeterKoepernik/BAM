from src.architectures import ConvDense, ConvDenseMixed
import src.lib as lib
from tests.lib_test import same_output

def test_convdense():
    input_shape = (3,5,5)
    a0 = ConvDense(input_shape, 'binary_cross_entropy')
    a = a0.deep_copy()
    assert same_output(a.model, a0.model)
    
    ###### mutate then adapt layers then build
    s = a.stacks[0] # conv2d stack
    print(a.summary())
    layer_index = len(s.layers) - 1
    indices = lib.widen_layer(s,layer_index, 2)
    print(a.summary())
    a.adapt_layers(0, ('widen', (layer_index, indices)))
    print(a.summary())
    assert same_output(a.model, a0.model)
    lib.delete_nodes(s, layer_index, indices)
    a.adapt_layers(0, ('shrink', (layer_index, indices)))
    print(a.summary())
    assert same_output(a.model, a0.model)

def test_convdensemixed():
    cnn_input_shape = (3,5,5)
    dense_input_shape = (10,)
    a0 = ConvDenseMixed(dense_input_shape, cnn_input_shape, 'binary_cross_entropy')
    a = a0.deep_copy()
    assert same_output(a.model, a0.model)
    
    ###### mutate then adapt layers then build
    s = a.stacks[0] # conv2d stack
    print(a.summary())
    layer_index = len(s.layers) - 1
    indices = lib.widen_layer(s,layer_index, 2)
    print(a.summary())
    a.adapt_layers(0, ('widen', (layer_index, indices)))
    print(a.summary())
    assert same_output(a.model, a0.model)
    lib.delete_nodes(s, layer_index, indices)
    a.adapt_layers(0, ('shrink', (layer_index, indices)))
    print(a.summary())
    assert same_output(a.model, a0.model)
