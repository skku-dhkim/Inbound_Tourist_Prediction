from utils.data_loader import DataLoader
from utils.spliter import WindowGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from models.rnn import bidirectional_RNN, LSTM_RNN
from models.attentionLSTM import AttentionLSTM
from models.decompCNN import CNN
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from utils.plot_util import plot
import pandas as pd

data_loader = DataLoader()
train, test, date_lists = data_loader.load_csv('./data/tourist_daily.csv', features=['Date', 'Total_entry', 'seoul_hotel', 'incheon_airport'], split_date='2018-12-31')

train = data_loader.minmaxScale(train)
test = data_loader.minmaxScale(test)

wg = WindowGenerator(train, input_width=30, label_width=1, shift=30)
twg = WindowGenerator(test, input_width=30, label_width=1, shift=30)

train_x, train_y = wg.split_window()
test_x, test_y = twg.split_window()

# NOTE: This line for CNN models.
# train_x = wg.reshape(train_x)
# test_x = twg.reshape(test_x)

# NOTE: Number of units list should be same with stacked layer
# model = bidirectional_RNN([128, 64], 2, [30, train.shape[-1]])
model = LSTM_RNN([128, 64], 2, [30, train.shape[-1]])
# model = AttentionLSTM(128, 64, 1)
# model = CNN(6, 1, dropout=0.5)

lr = 0.001
optimizer = Adam(learning_rate=lr)

model.compile(optimizer=optimizer, loss="mean_squared_error")

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
tb = TensorBoard('logs')

history = model.fit(train_x,
                    train_y,
                    shuffle=True,
                    epochs=50,
                    callbacks=[es, rlr, tb],
                    validation_split=0.2,
                    verbose=1,
                    batch_size=64)

train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

inv_train_predict = data_loader.inverseScale(train_predict)
inv_test_predict = data_loader.inverseScale(test_predict)

TRUE = pd.DataFrame(data_loader.original[:, 0:1], columns=["True"]).set_index(pd.Series(date_lists[0]))
prediction_train = wg.to_pandas(inv_train_predict[1:], date_lists[1], name="Train")
prediction_test = twg.to_pandas(inv_test_predict[1:], date_lists[2], name="Test")

plot(TRUE, prediction_train, prediction_test, show=True)


