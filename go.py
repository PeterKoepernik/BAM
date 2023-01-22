from tests.lib_test import test_pick_from_list, test_pick_from_dict

#test_pick_from_list()
#test_pick_from_dict()

import tests.lib_test as test
#test.test_stack(verbose = 1, layer_type = 'both')
#test.test_mutate_stacks(verbose = 1, layer_type = 'both')
#test.test_simple_architecture(verbose = 1, layer_type = 'both')
#test.test_fold_kernels()
#test.test_fold_layers()
#
#test.test_stack_saveload()
#test.test_arch_saveload(layer_type = 'conv2d')

import tests.arch_test as atest
#atest.test_convdense()
#atest.test_convdensemixed()

import applications.mnist
