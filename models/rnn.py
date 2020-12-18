import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed, InputLayer, Flatten
from tensorflow.keras.models import Sequential


def bidirectional_RNN(units: list, num_of_stack: int, input_shape: list, dropout=0.25):
    def is_last_layer(stack_id, num_of_stack):
        if stack_id == num_of_stack-1:
            return True
        else:
            return False
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(num_of_stack):
        if is_last_layer(units[i], num_of_stack):
            model.add(Bidirectional(LSTM(
                units=units,
                return_sequences=False)))
            continue
        model.add(Bidirectional(LSTM(
            units=units[i],
            return_sequences=True,
            input_shape=input_shape)))

    # Adding Dropout
    model.add(Dropout(dropout))
    model.add(Dense(units=60, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(units=1, activation='linear'))
    return model


def LSTM_RNN(units: list, num_of_stack: int, input_shape: list, dropout=0.25):
    def is_last_layer(stack_id, num_of_stack):
        if stack_id == num_of_stack-1:
            return True
        else:
            return False
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(num_of_stack):
        if is_last_layer(units[i], num_of_stack):
            model.add(LSTM(
                units=units,
                return_sequences=False))
            continue
        model.add(LSTM(
            units=units[i],
            return_sequences=True,
            input_shape=input_shape))

    # Adding Dropout
    model.add(Dropout(dropout))
    model.add(Dense(units=60, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(units=1, activation='linear'))
    return model
