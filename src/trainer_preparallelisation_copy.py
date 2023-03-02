import os
import numpy as np
import tensorflow as tf

if os.path.join(os.getcwd(),'src') == os.path.dirname(os.path.realpath(__file__)):
    ###### we are being called from within this package
    from src.lib import Architecture, pick_from_list
else:
    ###### we are being called from Terminal, in which case the BAM/src dir was added to global environment
    from lib import Architecture, pick_from_list

class Population:
    """
    Manages a set of architectures
    """
    def load(directory,
            ds_train = None, # if these are none then the population can only be inspected, not trained further
            ds_val = None,
            **kwargs
            ):
        """
        Loads a population from a previous save. All keywords that are listed below are, if not provided, initialised as is implemented as default in Population.__init__.

        this directory is .../population_name and in it are 'generation_i/'
        """
        for key in kwargs.keys():
            assert key in [
                    'train_size',
                    'val_size',
                    'train_data_per_epoch',
                    'val_data_per_epoch',
                    'mutations_per_generation',
                    'saveload']

        print(f'Loading population at {directory}')

        generations = os.listdir(directory)
        assert len(generations) > 0
        for path in generations:
            assert path.startswith('generation_')


        population_size = len(os.listdir(os.path.join(directory, generations[0])))
        assert population_size > 0

        print(f'Found {len(generations)} generations of size {population_size}.')

        from pathlib import Path
        directory = str(Path(directory)) # convert it to the natural OS representation (fex remove trailing / delimiter)
        name = directory.split('/')[-1]

        print(f'Name is {name}.')

        pop = Population(
                architecture_factory = lambda: None,
                population_size = population_size,
                ds_train = ds_train,
                ds_val = ds_val,
                name = name,
                directory = directory,
                #directory = os.path.join(*(directory.split('/')[:-1])),
                **kwargs
                )

        i = 0
        while os.path.exists(os.path.join(directory, f'generation_{i+1}')):
            i += 1
        pop.generation = i
        print(f'Set generation to {i}.')

        pop.reload()

        return pop

    def __init__(self,
            architecture_factory,
            population_size,
            ds_train,
            ds_val,
            train_size = 0,
            val_size = 0,
            name = '',
            directory = None,
            train_data_per_epoch = 0.5,
            val_data_per_epoch = 1.0,
            mutations_per_generation = 1.0,
            saveload = True,
            num_cores = 1 # if more than one we use parallelisation
            ):
        """
            train_data_per_epoch: is the fraction of the training data which is used for training of one epoch. Every member of the population gets an independently sampled random subset of the data with that size. If the given number is an integer, it is assumed that it is the total size of train samples used per epoch (rather than a fraction)
            mutations_per_generation: average number of mutations per individual per generation (epoch)
        """
        if population_size == 'singleton':
            population_size = 1
            train_data_per_epoch = 1.0
            val_data_per_epoch = 1.0
            mutations_per_generation = 0.0

        self.name = name

        if directory == None and saveload:
            base = os.path.join(os.getcwd(), 'saves')
            i=0
            def path(i):
                return os.path.join(base, f'population_{i}')
            while os.path.exists(path(i)):
                i += 1
            directory = path(i)

        self.saveload = saveload
        if saveload:
            self.directory = directory
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.generation = 0
        self.population_size = population_size
        
        if ds_train != None and ds_val != None:
            self.update_datasets(ds_train, ds_val, train_size, val_size)
        else:
            print('Population loaded in inspect-only mode. Training calls will fail')
       
        if train_data_per_epoch > 1:
            assert type(train_data_per_epoch) == int
            self.train_data_per_epoch = min(1.0, train_data_per_epoch / self.train_size)
        else:
            self.train_data_per_epoch = train_data_per_epoch

        assert self.train_data_per_epoch > 0 and self.train_data_per_epoch <= 1

        if val_data_per_epoch > 1:
            assert type(val_data_per_epoch) == int
            self.val_data_per_epoch = min(1.0, val_data_per_epoch / self.val_size)
        else:
            self.val_data_per_epoch = val_data_per_epoch

        assert self.val_data_per_epoch > 0 and self.val_data_per_epoch <= 1

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.train_data_per_epoch = train_data_per_epoch
        self.val_data_per_epoch = val_data_per_epoch
        self.mutations_per_generation = mutations_per_generation

        self.population = [architecture_factory() for i in range(population_size)]
        self.prev_population = [] # before last mutation+epoch

    def update_datasets(self, ds_train, ds_val, train_size = 0, val_size = 0):
        self.train_size = train_size        
        if train_size == 0:
            self.train_size = ds_train.cardinality()
            
        self.val_size = val_size
        if val_size == 0:
            self.val_size = ds_val.cardinality()
            
        assert self.train_size > 0
        assert self.train_size > 0        

    def epoch(self, verbose = 0):
        assert self.ds_train != None and self.ds_val != None

        ###### used for subsampling datasets
        def filter_factory(present_samples):
            def _filter(i, el):
                return present_samples[i]
            return _filter

        def project(i, el):
            return el

        def process(arch):
            arch.compile()
            model = arch.model

            ###### sample dataset
            present_datasets = tf.convert_to_tensor(np.random.random(self.train_size) < self.train_data_per_epoch) # list of True/False of length of training data with fraction of True's on average self.train_data_per_epoch
            _filter = filter_factory(present_datasets)
            ds_train_sampled = self.ds_train.enumerate().filter(_filter).map(project)

            ###### train
            hist = model.fit(ds_train_sampled, epochs = 1, verbose = verbose)
            train_loss = hist.history['loss'][-1]
            arch.train_loss = train_loss
            #todo test the above at an example, later test how much slower is this than using x_train when data_per_epcoh = 1 (ie how much overhead does the subsampling have)

            ###### compute validation error
            present_datasets = tf.convert_to_tensor(np.random.random(self.val_size) < self.val_data_per_epoch) # list of True/False of length of training data with fraction of True's on average self.train_data_per_epoch
            _filter = filter_factory(present_datasets)
            ds_val_sampled = self.ds_val.enumerate().filter(_filter).map(project)
            val_losses = model.evaluate(ds_val_sampled, verbose = 0)
            if len(arch.metrics) == 0:
                val_loss = val_losses
                arch.val_loss = val_loss
            else:
                val_loss, acc = val_losses[:2]
                arch.val_loss = val_loss
                arch.acc = acc # this is not renamed to val_acc for historical reasons
                arch.train_acc = hist.history['accuracy'][-1]

        ###### train
        print(f'Epoch: 0/{len(self.population)}', end = '\r')
        for i, arch in enumerate(self.population):
            process(arch)
            print(f'Epoch: {i+1}/{len(self.population)}', end = '\r')
        print('')

        ###### add old population if they were already evaluated (only not in the very first run)
        for a in self.prev_population:
            if hasattr(a, 'val_loss'):
                self.population.append(a)

        ###### sort according to performance and cut to best half
        self.population.sort(key = lambda a: a.val_loss)
        self.population = self.population[:self.population_size]

        self.generation += 1

        self.save()

    def print_population(self):
        print(f'\n=== Population (Gen {self.generation}) ===')
        num = 3
        for i, a in enumerate(self.population):
            if i == num:
                print('...\n')
            if i >= num and i < self.population_size - num:
                continue
            string = f'{i+1}: {a.summary()}\nloss/acc train - '# (train: {round(a.train_loss,3)}'
            if hasattr(a, 'train_loss'):
                string += f'{round(a.train_loss,3)} / '
            if hasattr(a, 'train_acc'):
                string += f'{round(100*a.train_acc,2)}%'
            string += '\nloss/acc val - '
            if hasattr(a, 'val_loss'):
                string += f'{round(a.val_loss,3)} / '
            if hasattr(a, 'acc'):
                string += f'{round(100*a.acc,2)}%'
            string += '\n'
            print(string)

    def print_weights(self, model_id = 0):
        assert model_id >= 0 and model_id < self.population_size
        self.population[model_id].print_weights()

    def save(self):
        if not self.saveload:
            return False

        base = os.path.join(self.directory, f'generation_{self.generation}')
        if os.path.exists(base):
            return False #already saved this generation
        os.mkdir(base)

        def path(i):
            return os.path.join(base, f'individual_{i}')

        for i, a in enumerate(self.population):
            a.save(path(i))

        return True

    def reload(self):
        assert self.saveload
        base = os.path.join(self.directory, f'generation_{self.generation}')

        assert os.path.exists(base)

        def path(i):
            return os.path.join(base, f'individual_{i}')

        self.population = [Architecture.load(path(i)) for i in range(self.population_size)]

    def restart_session(self):
        if not self.saveload:
            return
        self.save()
        tf.keras.backend.clear_session()
        self.reload()

    def mutate(self, verbose = 0):
        """
        To form the new generation, we sample n times from the current generation (with repitition) and mutate every new individual Poisson(mutations_per_generation) times
        """
        self.prev_population = self.population
        new_population = []
        parent_picks = [0] * self.population_size
        for i in range(self.population_size):
            # pick and mutate new individual
            i, parent = self.sample_individual()
            parent_picks[i] += 1
            child = parent.deep_copy()

            from scipy.stats import poisson
            num_mutations = poisson.rvs(mu=self.mutations_per_generation)
            #num_mutations = 1
            child.mutate(num_mutations)

            new_population.append(child)

        self.population = new_population

        if verbose > 0:
            print('\n==Mutation==')
            print(f'Number of children per parent: {parent_picks}.')
            print('')

    def sample_individual(self):
        """
        Sample an individual from the current generation, biased towards those with high ranking.
        """
        def weight(x, s = 2):
            assert x >= 0 and x <= 1
            x = 1 - x
            if x < .5:
                return .5*(2*x)**s
            return 1-.5*(2*(1-x))**s
        
        sample_index = pick_from_list([weight(i / self.population_size) for i in range(self.population_size)])
        return sample_index, self.population[sample_index]

if __name__ == '__main__':
    """
    This means we are tasked with training one individual of one generation
    """
    import sys, os

    assert len(sys.argv) == 5
    process_id = int(sys.argv[1])
    path_to_train_ds = sys.argv[2]
    path_to_val_ds = sys.argv[3]
    path_to_individual = sys.argv[4]
    
    assert os.path.exists(path_to_train_ds)
    assert os.path.exists(path_to_val_ds)
    assert os.path.exists(path_to_individual)

    train_ds = tf.data.Dataset.load(path_to_train_ds)
    print(f'This is process number {process_id}. This is the first element of my training set.')
    for el in train_ds.take(1):
        print(el)
    import time
    time.sleep(3)
    
