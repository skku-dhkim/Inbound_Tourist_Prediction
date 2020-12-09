import pandas as pd
import numpy as np
import utils.normalization as norm
import tensorflow as tf
import random


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 data_df, label_columns=None):
        # Store the raw data.
        self.train_df = None
        self.val_df = None
        self.skeleton_df = data_df
        self.skeleton_np = data_df.to_numpy()
        self.label_np = data_df[['Total_entry']].to_numpy()

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.skeleton_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift + label_width

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        data_df.min().to_csv("./min.csv")
        data_df.max().to_csv("./max.csv")

    def split_window(self, normalization=True):
        skeleton_x = []
        skeleton_y = []

        if normalization:
            self.skeleton_df = norm.min_max(self.skeleton_df)
            self.skeleton_np = self.skeleton_df.to_numpy()
            self.label_np = norm.min_max(self.label_np)

        for idx in range(len(self.skeleton_df)-self.total_window_size+1):
            skeleton_x.append(self.skeleton_np[self.input_indices+idx])
            skeleton_y.append(self.label_np[self.label_indices+idx])
        return np.array(skeleton_x), np.array(skeleton_y)

    def augmentation(self, times: int, shuffle: bool):
        def make_random_variable(half_std):
            _rnd = [random.uniform(-value, value) for _, value in enumerate(half_std)]
            _rnd[0] = 0     # Not change
            __rnd = []
            for _ in range(self.input_width):
                __rnd.append(_rnd)
            return np.array(__rnd)

        skeleton_x, skeleton_y = self.split_window()

        X, Y = [], []
        half_std_value = self.skeleton_df.std(axis=0)

        for i in range(times):
            for batch in skeleton_x:
                rnd = make_random_variable(half_std_value)
                X.append(batch+rnd)

            for batch in skeleton_y:
                Y.append(batch)

        X = np.array(X)
        Y = np.array(Y)
        Y = np.squeeze(Y, axis=2)

        if shuffle:
            rnd_idx = np.array([i for i in range(len(X))])
            np.random.shuffle(rnd_idx)
            X = X[rnd_idx]
            Y = Y[rnd_idx]

        return X, Y

    def test_set(self):
        skx, sky = self.split_window()
        sky = np.squeeze(sky, axis=2)
        return skx, sky

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

