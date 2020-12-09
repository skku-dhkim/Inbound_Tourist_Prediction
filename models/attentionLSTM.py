import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, **kwargs):
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)

        self.V = tf.keras.layers.Dense(1)

    def call(self, query_h, query_c, values):
        query_with_time_axis_h = tf.expand_dims(query_h, 1)
        query_with_time_axis_c = tf.expand_dims(query_c, 1)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis_h) + self.W3(query_with_time_axis_c) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, output_size, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(output_size)

        # used for attention
        self.attention = Attention(self.dec_units)

    def call(self, hidden_h, hidden_c, enc_output):
        context_vector, attention_weights = self.attention(hidden_h, hidden_c, enc_output)
        x = tf.expand_dims(context_vector, 1)
        output, state_h, state_c = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state_h, state_c, attention_weights


class AttentionLSTM(tf.keras.Model):
    def __init__(self, units, BATCH_SIZE, output_seq):
        super(AttentionLSTM, self).__init__()
        self.encoder = Encoder(units, BATCH_SIZE)
        self.decoder = Decoder(output_seq, units, BATCH_SIZE)

    def call(self, X, **kwargs):
        sample_output, sample_hidden_h, sample_hidden_c = self.encoder(X)
        sample_decoder_output, _, _, _ = self.decoder(sample_hidden_h, sample_hidden_c, sample_output)
        return sample_decoder_output
