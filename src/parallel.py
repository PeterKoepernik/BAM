"""
for testing parallelisation with tf
"""

#!/usr/bin/env python
# coding: utf-8

# # Week 2: Implementing Callbacks in TensorFlow using the MNIST Dataset
#
# In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.
#
# Write an MNIST classifier that trains to 99% accuracy and stops once this threshold is achieved. In the lecture you saw how this was done for the loss but here you will be using accuracy instead.
#
# Some notes:
# 1. Your network should succeed in less than 9 epochs.
# 2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!" and stop training.
# 3. If you add any additional variables, make sure you use the same names as the ones used in the class. This is important for the function signatures (the parameters and names) of the callbacks.

# In[1]:


import os

def work(_id):
    import tensorflow as tf
    from tensorflow import keras
    
    # Get current working directory
    current_dir = os.getcwd()
    
    # Append data/mnist.npz to the previous path to get the full path
    data_path = os.path.join(current_dir, "mnist.npz")
    
    # Discard test set
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path=data_path)
    
    # Normalize pixel values
    x_train = x_train / 255.0
    
    
    # Now take a look at the shape of the training data:
    
    # In[3]:
    
    
    data_shape = x_train.shape
    
    print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
    
    
    # Remember to inherit from the correct class
    class myCallback(keras.callbacks.Callback):
            # Define the correct function signature for on_epoch_end
            def on_epoch_end(self, epochs, logs={}):
                if logs.get('accuracy',0) > 0.99:
                    print("\nReached 99% accuracy so cancelling training!")
    
                    # Stop training once the above condition is met
                    self.model.stop_training = True
    
    ### END CODE HERE
    
    ### START CODE HERE
    
    # Instantiate the callback class
    callback = myCallback()
    
    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fit the model for 10 epochs adding the callbacks
    # and save the training history
    history = model.fit(x_train, y_train, epochs=1, callbacks=[callback])

    print(f'Worker {_id} finished!')

def print_me(number):
    print(f'Number {number}!')

from multiprocessing import Pool

import time
t = time.time()
work(2)
t_single = time.time() - t
t = time.time()
with Pool(4) as pool:
    pool.map(work, [1,2,3,4])
print(f'Sequential: {round(t_single)}s')
print(f'Parallel: {round((time.time()-t)/4)}s')
