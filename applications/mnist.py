import tensorflow as tf
import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load('mnist', split=['train','test'], shuffle_files=True)

def process(el):
    return (el['image'], el['label'])

ds_train = ds_train.map(process).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(process).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

input_shape = (28,28,1)
output_shape = (10,)
#for example in ds_train.take(1):
#    image = example['image']
#    label = example['label']
#    print(label)
#    _, w, h, ch = image.shape
#    input_shape = (w,h,ch)

from src.trainer import Population
from src.architectures import ConvDense

def architecture_factory():
    return ConvDense(
            input_shape,
            'sparse_categorical_crossentropy',
            metrics = ['accuracy'],
            output_size = 10,
            output_activation = 'softmax',
            initial_width = [5,10],
            initial_depth = [2,2],
            )

pop_size = 8
pop = Population(
        architecture_factory,
        pop_size,
        ds_train,
        ds_test,
        train_data_per_epoch = .7,
        mutations_per_generation = 2
        )

epochs = 50

for e in range(epochs):
    pop.mutate(verbose = 1)
    pop.restart_session()
    pop.epoch(verbose = 0)
    pop.print_population()
