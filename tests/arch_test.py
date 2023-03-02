from src.architectures import ConvDense, ConvDenseMixed, PriorArchSimple
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

import numpy as np
def test_priorarch():
    num_stats0 = 2
    num_statslr = 3
    num_outputs0 = 1
    num_outputslr = 2
    a = PriorArchSimple(
            num_stats0 = num_stats0,
            num_statslr = num_statslr,
            num_outputs0 = num_outputs0,
            num_outputslr = num_outputslr
            )
    N = 3000
    inp0 = np.random.random((N,num_stats0))
    inpl = np.random.random((N,num_statslr))
    inpr = np.random.random((N,num_statslr))
    m = a.model

    out1 = m((inp0,inpl,inpr)).numpy()
    out2 = m((inp0,inpr,inpl)).numpy()

    ind0 = list(range(num_outputs0))
    indl = list(range(num_outputs0, num_outputs0+num_outputslr))
    indr = list(range(num_outputs0+num_outputslr, num_outputs0+2*num_outputslr))

    assert close(np.take(out1,ind0,axis=1), np.take(out2,ind0,axis=1))
    assert close(np.take(out1,indl,axis=1), np.take(out2,indr,axis=1))
    assert close(np.take(out1,indr,axis=1), np.take(out2,indl,axis=1))

    for i in range(min(N,5)):
        print(f'\nout1 = {list([round(x,3) for x in out1[i,:]])}')
        print(f'out2 = {list([round(x,3) for x in out2[i,:]])}')

EPS = 1e-4
def close(x,y):
    return (np.abs(x-y) < EPS).all()
