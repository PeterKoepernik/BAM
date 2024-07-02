from tests.lib_test import test_pick_from_list, test_pick_from_dict

#test_pick_from_list()
#test_pick_from_dict()

import tests.lib_test as test
test.test_stack(verbose = 0, layer_type = 'both')
test.test_mutate_stacks(verbose = 0, layer_type = 'both')
#test.test_simple_architecture(verbose = 1, layer_type = 'both')
#test.test_fold_kernels()
#test.test_fold_layers()
#
test.test_stack_saveload(verbose = 1)
test.test_arch_saveload(verbose = 0, layer_type = 'conv2d')
#test.test_population(mode = 'value_arch')
#test.test_value_arch()
#test.test_value_arch_conv()

import tests.arch_test as atest
#atest.test_convdense()
#atest.test_convdensemixed()
#atest.test_priorarch()

#import applications.mnist
