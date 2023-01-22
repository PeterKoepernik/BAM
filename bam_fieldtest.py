import tensorflow as tf
import tensorflow.keras as keras
import sys
import random
import numpy as np

width = 10

def softmax(scores):
    exps = np.exp(scores)
    Z = np.sum(exps)
    return exps / Z

def pick(scores, num = 1, mode = 'softmax'):
    """
    mode either `softmax` or `linear`
    """
    if num > 1:
        i = pick(scores, 1)
        scores = scores[:i] + scores[i+1:]
        other_picks = pick(scores, num-1)
        if type(other_picks) == int:
            other_picks = [other_picks]
        other_picks = [(j if j < i else (j+1)) for j in other_picks]
        return [i] + other_picks

    if mode == 'softmax':
        probs = softmax(scores)
    elif mode == 'linear':
        Z = sum(scores)
        probs = [sc / Z for sc in scores]
    else:
        raise Exception('Unsupported mode')

    x = random.random()
    i = 0
    cum = probs[i]
    while x > cum:
        i += 1
        cum += probs[i]
    return i

def clone_model(model):
    model_copy= keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    return model_copy

def mutate_model(model):
    # assume layers are [input, flatten, [hidden,...] output]
    n_layers = len(model.layers)
    if random.random() < .7: # widen layer
        i = random.randrange(1 + flatten_layer, n_layers - 1)
        return widen_layer(model, i, max(1,int(.2*model.layers[i].output.shape[1])))
    else: # new layer
        i = random.randrange(2 + flatten_layer, n_layers)
        return insert_layer(model, i)

def architecture_of(model):
    # returns a list of hidden layer sizes, assuming it is [input, flatten, hidden, output]
    layer_sizes = [l.output.shape[1] for l in model.layers[1+flatten_layer:-1]]
    string = ''
    for l in layer_sizes:
        string += str(l) + '-'
    return string[:-1]

def perform_generation(models, num_epochs = 1, print_summary = False):
    sorted_models = [] # elements (test_loss, model.name)
    model_dict = {} # model.name -> (train_loss, train_acc, test_loss, test_acc)
    scores = []
    for m in models:
        hist = train(m, epochs = num_epochs, verbose = 0)
        train_loss = hist.history['loss'][0]
        train_acc = hist.history['accuracy']
        test_loss, test_acc = m.evaluate(x_test, y_test, verbose = 0)
        model_dict[m.name] = (train_loss, train_acc, test_loss, test_acc)
        sorted_models.append((test_loss, m))
        scores.append(-4*np.log(test_loss))
    sorted_models.sort()

    if print_summary:
        print('\n===Generation done===')
        for i, (_,model) in enumerate(sorted_models):
            print(f'{i+1}.: {architecture_of(model)} (train {model_dict[model.name][1][0]} test {model_dict[model.name][3]}) (score {1.0 / model_dict[model.name][2]**4})')

    models = mutate_population(models, scores)
    if print_summary:
        print(f'\nNew generation:')
        for m in models:
            print(architecture_of(m))

    return models

def init_population(num_networks = 10, initial_hidden_layers = 1, initial_width = 20):
    models = [seq_model(num_hidden_layers = initial_hidden_layers, width = initial_width) for i in range(num_networks)]
    return models

def mutate_population(models, scores, top_quantile = .1):
    # pick half of them according to given scores
    n = len(models)
    always_include = int(top_quantile*n + 1)
    sorted_models = list(zip(scores, range(len(models))))
    sorted_models.sort()
    sorted_models.reverse()
    ind = [i for (sc,i) in sorted_models[:always_include]]
    remaining_indices = pick(scores, num = int(n / 2) - always_include)
    if type(remaining_indices) == int:
        remaining_indices = [remaining_indices]
    ind += remaining_indices
    if type(ind) == int:
        ind = [ind]
    new_models = [models[i] for i in ind]
    for i in range(int(n / 2)):
        mutated_model = clone_model(new_models[i])
        mutated_model = mutate_model(mutated_model)
        new_models.append(mutated_model)
    return new_models

def widen_layer(model, layer_id, extra_nodes):
    layers = [l for l in model.layers]

    x = layers[0].output
    assert layer_id > 0 and layer_id < len(layers) - 1

    for i in range(1, len(layers)):
        if i == layer_id:
            # create new layer
            n_in = layers[i].input.shape[1]
            n_out = layers[i].output.shape[1]
            new_layer = tf.keras.layers.Dense(
                    n_out + extra_nodes,
                    activation = 'relu'
                    )

            x = new_layer(x)

            # copy weights from old layer
            new_weights, new_bias = new_layer.get_weights()
            old_weights, old_bias = layers[i].get_weights()
            new_weights[:,:n_out] = old_weights
            new_bias[:n_out] = old_bias
            new_layer.set_weights([new_weights, new_bias])

            # create new after layer with larger input
            out_layer = tf.keras.layers.Dense(
                    layers[i+1].output.shape[1],
                    activation = layers[i+1].activation,
                    kernel_initializer = keras.initializers.zeros
                    )
            x = out_layer(x)
            old_weights, old_bias = layers[i+1].get_weights()
            new_weights, new_bias = out_layer.get_weights()
            new_weights[:n_out,:] = old_weights
            out_layer.set_weights([new_weights, old_bias])
        elif i != layer_id + 1: # skip layer after widened one
            x = layers[i](x)

    new_model = keras.models.Model(layers[0].input, x)
    return new_model

def insert_layer(model, layer_id, new_layer = None, layer_name = None):
    """
    new_layer will have index layer_id in the new model. If new_layer is None, it is initialized as a dense layer with same width as the previous layer, relu activation, and identity weights.
    """
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            if new_layer == None:
                new_layer = tf.keras.layers.Dense(
                        layers[i-1].output.shape[1],
                        activation = 'relu',
                        kernel_initializer = tf.keras.initializers.identity
                        )
            if layer_name != None:
                new_layer._name = layer_name
            x = new_layer(x)
        x = layers[i](x)

    new_model = keras.models.Model(layers[0].input, x)
    return new_model

def compile(model):
    model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']) 

def train(model, epochs = 1, verbose = 1):
    compile(model)
    return model.fit(x_train, y_train, epochs = epochs, verbose = verbose)

def load_mnist():
    import os
    # Load the data

    # Get current working directory
    current_dir = os.getcwd()
    
    # Append data/mnist.npz to the previous path to get the full path
    #data_path = os.path.join(current_dir, "data/mnist.npz")
    data_path = "mnist.npz"
    
    global x_train, y_train, x_test, y_test, num_classes, input_shape, flatten_layer
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=data_path)
    
    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    input_shape = (28,28,1)
    flatten_layer = True

def load_reuters(max_words = 1000):
    global x_train, y_train, x_test, y_test, num_classes, input_shape, flatten_layer

    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=max_words, test_split=0.2, path = 'reuters.npz')

    num_classes = np.max(y_train) + 1
    
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = (1000,)
    flatten_layer = False

def seq_model(num_hidden_layers = 1, width = 5):
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    if flatten_layer:
        x = tf.keras.layers.Flatten()(x)

    for i in range(num_hidden_layers):
        x = tf.keras.layers.Dense(width, activation = 'relu')(x)

    x = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(x)
    model = keras.models.Model(inp, x)
    return model

load_reuters()
#load_mnist()

models = init_population(num_networks = 8, initial_width = 10)
models = perform_generation(models, print_summary = True)
while True:
    input('Press Enter for next generation')
    models = perform_generation(models, print_summary = True)

#model = seq_model(num_hidden_layers = 2)
#
#train(model, epochs = 10)
#
#num_layers = 1
#model = seq_model(num_hidden_layers = 1, width = 5)
#model.summary()
#
#max_depth = 10
#
#for depth in range(max_depth):
#    if depth > 0:
#        insert_index = random.randint(3,len(model.layers)-1) # 0 = input, 1 = flatten, 2 = first hidden
#        model = insert_layer(model, insert_index, layer_name = f'hidden_{num_layers+1}')
#        num_layers += 1
#        #print(f'\nNew layer inserted at position {insert_index}.')
#        #model.summary()
#    train(model)
