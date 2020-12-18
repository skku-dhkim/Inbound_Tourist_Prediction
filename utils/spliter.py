import numpy as np
import pandas as pd


class WindowGenerator:
    def __init__(self, data, input_width, label_width, shift):
        self.data = data
        self.label_data = data[:, 0]

        # Set window size
        self.total_window_size = (input_width + shift + label_width) - 1

        # Input and label index slices
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self):
        _x = []
        _y = []
        for idx in range(len(self.data)-self.total_window_size+1):
            _x.append(self.data[self.input_indices + idx])
            _y.append(self.label_data[self.label_indices + idx])
        return np.array(_x), np.array(_y)

    def to_pandas(self, data, date_list, name="value"):
        df = pd.DataFrame(data, columns=[name]).set_index(pd.Series(date_list[self.total_window_size:]))
        return df

    def reshape(self, d):
        shape = d.shape
        d = d.reshape([-1, 1, shape[1], shape[2]])
        return d

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}'])

