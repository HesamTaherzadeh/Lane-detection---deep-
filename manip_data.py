import numpy as np 
from functools import wraps
from time import time


class Data_Manipulation():
    def __init__(self, address_data, address_label):
        """ the init method for the Data Manipulation Class that is targeted at 

        Args:
            address_data (_type_): _description_
            address_label (_type_): _description_
        """
        self.address_data = address_data 
        self.address_label = address_label

    def timing(function):
        @wraps(function)
        def wrapper(*args, **kw):
            start_time = time()
            result = function(*args, **kw)
            end_time = time()
            print(f"function {function.__name__} took {end_time - start_time} second ")
            return result
        return wrapper

    def _read_data(self):
        self.data = np.load(self.address_data)
        self.label = np.expand_dims(np.load(self.address_label), axis=-1)
        self.data = np.concatenate([self.data, self.label], axis=3)
        self.shape_of_data = self.data.shape


    
    def shuffle_split(self, ratio_of_test2train, random_seed = 0):
        self._read_data()
        np.random.seed(random_seed)
        # np.random.shuffle(self.data)
        test_count = int(self.shape_of_data[0] * ratio_of_test2train)
        test_data = self.data[:test_count, :, :, :3]
        train_data = self.data[test_count:, :, :, :3]
        test_label = self.data[:test_count, :, :, 3]
        train_label = self.data[test_count:, :, :, 3]
        return train_data, train_label, test_data, test_label



        
