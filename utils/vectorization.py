import numpy as np
import pandas as pd

class Vectorization:
    def __init__(self):
        self.vector_size =None
        self.features = None
        self.train_vector = []
        self.test_vector = []

    def single_vector(self, input_arr, vector_num):
        result = np.zeros((len(input_arr), vector_num))
        for i in range(len(input_arr)):
            result[i, int(input_arr[i])] = 1
        return result


    def To_vector(self, train, test, features, vector_size, input_length):
        self.vector_size = vector_size
        self.features = features
        for idx, var in enumerate(features):
            train_x = train[:, :, idx]
            test_x = test[:, :, idx]
            if vector_size[var]==1:
                vector_train = np.expand_dims(train_x, axis=2)
                vector_test = np.expand_dims(test_x, axis=2)
            else :
                vector_train = np.zeros((train_x.shape[0], train_x.shape[1], vector_size[var]))
                vector_test = np.zeros((test_x.shape[0], test_x.shape[1], vector_size[var]))
                for i in range(input_length):
                    vector_train[:, i] = self.single_vector(train_x[:,i], vector_num = vector_size[var])
                    vector_test[:, i] = self.single_vector(test_x[:,i], vector_num = vector_size[var])

            self.train_vector.append(vector_train)
            self.test_vector.append(vector_test)

        return self.train_vector, self.test_vector
