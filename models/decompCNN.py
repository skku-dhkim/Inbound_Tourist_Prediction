import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations

tf.keras.backend.set_floatx('float64')


class CNN(tf.keras.Model):
    def __init__(self, units, output_seq, dropout=0.25):
        super(CNN, self).__init__()
        self.conv2d_1_1 = layers.Conv2D(units, kernel_size=[1, 3], strides=[1, 2], padding='valid',
                                        activation=activations.tanh)
        self.conv2d_1_2 = layers.Conv2D(units, kernel_size=[1, 3], strides=[1, 1], padding='valid',
                                        activation=activations.tanh)
        self.conv2d_2_1 = layers.Conv2D(units, kernel_size=[1, 5], strides=[1, 2], padding='valid',
                                        activation=activations.tanh)
        self.conv2d_2_2 = layers.Conv2D(units, kernel_size=[1, 2], strides=[1, 1], padding='valid',
                                        activation=activations.tanh)

        self.conv2d_3 = layers.Conv2D(units*2, kernel_size=[2, 1], strides=[1, 1], activation=activations.tanh)
        self.conv2d_4 = layers.Conv2D(units*2, kernel_size=[1, 3], strides=[1, 2], activation=activations.tanh)
        self.max_pool_1 = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 1))

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(300, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.dense2 = layers.Dense(output_seq, activation='linear')

    def call(self, input_X, **kwargs):
        _x1 = self.conv2d_1_1(input_X)
        _x1 = self.conv2d_1_2(_x1)

        _x2 = self.conv2d_2_1(input_X)
        _x2 = self.conv2d_2_2(_x2)

        concat_1 = layers.concatenate([_x1, _x2], axis=1)

        _x = self.conv2d_3(concat_1)
        _x = self.conv2d_4(_x)
        _x = self.max_pool_1(_x)

        _x = self.flatten(_x)
        _x = self.dense1(_x)
        _x = self.dropout(_x)
        _x = self.dense2(_x)

        return _x


# class Disease(tf.keras.Model):
#     def __init__(self):
#         super(Disease, self).__init__()
#         self.disease = layers.Dense(1, activation=activations.relu)
#
#     def call(self, disease_x, **kwargs):
#         _xd = self.disease(disease_x)
#         return _xd
#
#
# class Politics(tf.keras.Model):
#     def __init__(self):
#         super(Politics, self).__init__()
#         self.politics = layers.Dense(1, activation=activations.tanh)
#
#     def call(self, politics_x, **kwargs):
#         _xp = self.politics(politics_x)
#         return _xp
#
#
# class Season(tf.keras.Model):
#     def __init__(self):
#         super(Season, self).__init__()
#         self.season = layers.Dense(1, activation=activations.tanh)
#
#     def call(self, season_x, **kwargs):
#         _xs = self.season(season_x)
#         return _xs
#
#
# class LatentCNN(tf.keras.Model):
#     def __init__(self, out_sequences):
#         super(LatentCNN, self).__init__()
#         self.dense = layers.Dense(out_sequences, activation=activations.linear)
#         self.CNN = CNN()
#         self.LatentVector = LatentVector()
#
#     def call(self, input_x, **kwargs):
#         out_cnn = self.CNN(input_x[0])
#         out_lv = self.LatentVector(
#             input_x[1]['disease'],
#             input_x[1]['politics'],
#             input_x[1]['season']
#         )
#         out = layers.concatenate([out_cnn, out_lv])
#         out = self.dense(out)
#         return out

