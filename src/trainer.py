import tensorflow as tf
import src.lib as lib
import os
import numpy as np

class Population:
    """
    Manages a set of architectures
    """
    def __init__(self,
            architecture_factory,
            population_size,
            ds_train,
            ds_val,
            name = '',
            directory = None,
            train_data_per_epoch = 0.5,
            val_data_per_epoch = 1.0,
            mutations_per_generation = 1.0,
            saveload = True
            ):
        """
            train_data_per_epoch: is the fraction of the training data which is used for training of one epoch. Every member of the population gets an independently sampled random subset of the data with that size. If the given number is an integer, it is assumed that it is the total size of train samples used per epoch (rather than a fraction)
            mutations_per_generation: average number of mutations per individual per generation (epoch)
        """
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
                os.mkdirs(directory)

        self.generation = 0
        self.population_size = population_size

        self.train_size = ds_train.cardinality()
        assert self.train_size > 0
        self.val_size = ds_val.cardinality()
        assert self.val_size > 0

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

    def epoch(self, verbose = 0):
        ###### used for subsampling datasets
        def filter_factory(present_samples):
            def _filter(i, el):
                return present_samples[i]
            return _filter

        def project(i, el):
            return el

        sorted_population = []
        ###### for parallelisation
        #def callback(result):
        #    val_loss, arch = result
        #    sorted_population.append((val_loss, arch))

        def process(arch):
            arch.compile()
            model = arch.model

            ###### sample dataset
            present_datasets = tf.convert_to_tensor(np.random.random(self.train_size) < self.train_data_per_epoch) # list of True/False of length of training data with fraction of True's on average self.train_data_per_epoch
            _filter = filter_factory(present_datasets)
            ds_train_sampled = self.ds_train.enumerate().filter(_filter).map(project)

            ###### train
            model.fit(ds_train_sampled, epochs = 1, verbose = verbose)
            #todo test the above at an example, later test how much slower is this than using x_train when data_per_epcoh = 1 (ie how much overhead does the subsampling have)

            ###### compute validation error
            present_datasets = tf.convert_to_tensor(np.random.random(self.val_size) < self.val_data_per_epoch) # list of True/False of length of training data with fraction of True's on average self.train_data_per_epoch
            _filter = filter_factory(present_datasets)
            ds_val_sampled = self.ds_val.enumerate().filter(_filter).map(project)
            losses = model.evaluate(ds_val_sampled, verbose = 0)
            if len(arch.metrics) == 0:
                val_loss = losses
                arch.val_loss = val_loss
            else:
                val_loss, acc = losses[:2]
                arch.val_loss = val_loss
                arch.acc = acc

        ###### train
        for arch in self.population:
            process(arch)

        ###### sort according to performance
        self.population.sort(key = lambda a: a.val_loss)

        self.generation += 1

        self.save()

    def print_population(self):
        print('\n=== Population ===')
        for i, a in enumerate(self.population):
            string = f'{i+1}: {a.summary()}'
            if hasattr(a, 'val_loss'):
                string += f' ({round(a.val_loss,3)})'
            if hasattr(a, 'acc'):
                string += f' (accuracy {round(100*a.acc)}%)'
            print(string)

    def save(self):
        if not self.saveload:
            return
        base = os.path.join(self.directory, f'generation_{self.generation}')
        if os.path.exists(base):
            return #already saved this generation
        os.mkdir(base)

        def path(i):
            return os.path.join(base, f'individual_{i}')

        for i, a in enumerate(self.population):
            a.save(path(i))

    def load(self):
        assert self.saveload
        base = os.path.join(self.directory, f'generation_{self.generation}')

        assert os.path.exists(base)

        def path(i):
            return os.path.join(base, f'individual_{i}')

        self.population = [lib.Architecture.load(path(i)) for i in range(self.population_size)]

    def restart_session(self):
        if not self.saveload:
            return
        self.save()
        tf.keras.backend.clear_session()
        self.load()

    def mutate(self, verbose = 0):
        """
        To form the new generation, we sample n times from the current generation (with repitition) and mutate every new individual Poisson(mutations_per_generation) times
        """
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
        
        sample_index = lib.pick_from_list([weight(i / self.population_size) for i in range(self.population_size)])
        return sample_index, self.population[sample_index]

