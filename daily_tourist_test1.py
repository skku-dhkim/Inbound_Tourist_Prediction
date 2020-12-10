import pandas as pd
import tensorflow as tf
import datetime

from utils.preprocess import WindowGenerator
from models.attentionLSTM import AttentionLSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from utils.plot_util import plot


TEST_TYPE = 1  # 0: Before Corona Virus, 1: After Corona Virus, 2: All data use
# Load data from CSV
df = pd.read_csv("./data/tourist_daily.csv")
df = df[["Date", "Total_entry", "KOSPI", "Oil_price", "China_exchange_rate", "Japan_exchange_rate", "USA_exchange_rate",
         "Japan_GDP", "China_GDP", "USA_GDP"]]

if TEST_TYPE == 0:
    _df = df.loc[df['Date'] <= '2019-12-31']
    train_df = _df.loc[df['Date'] <= '2018-12-31']
    test_df = _df.loc[df['Date'] > '2018-12-31']
elif TEST_TYPE == 1:
    train_df = df.loc[df['Date'] <= '2020-07-31']
    test_df = df.loc[df['Date'] > '2020-07-31']
else:
    train_df = df
    test_df = df

datelist_train = list(train_df['Date'])
datelist_test = list(test_df['Date'])
datelist_train = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]
datelist_test = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_test]

date = df['Date']

train_df = train_df.drop('Date', axis=1)
test_df = test_df.drop('Date', axis=1)
print('Training set shape == {}'.format(train_df.shape))
print(train_df.head())
print('Test set shape == {}'.format(test_df.shape))
print(test_df.head())
print('All timestamps == {}'.format(len(datelist_train)))

'Hyper-parameters'
PAST_DATA = 10
FUTURE_DATA = 1
SHIFT = 30
BATCH_SIZE = 32
HIDDEN_UNITS = 64
EPOCHS = 1
UPSAMPLE_LEVEL = 10


def evaluation(x, y, m, date_list):
    r = m.predict(x)
    df_r = pd.DataFrame(r, columns=['Total_entry']).set_index(pd.Series(date_list[PAST_DATA + SHIFT + FUTURE_DATA - 1:]))
    df_real = pd.DataFrame(y, columns=['Total_entry']).set_index(pd.Series(date_list[PAST_DATA + SHIFT + FUTURE_DATA - 1:]))
    return df_r, df_real


train_wg = WindowGenerator(input_width=PAST_DATA,
                           label_width=FUTURE_DATA,
                           shift=SHIFT,
                           label_columns=['Total_entry'],
                           data_df=train_df)

test_wg = WindowGenerator(input_width=PAST_DATA,
                          label_width=FUTURE_DATA,
                          shift=SHIFT,
                          label_columns=['Total_entry'],
                          data_df=test_df,
                          references=['./min.csv', './max.csv'])

print(train_wg)

# # Augmentation and Make Train set
X, Y = train_wg.augmentation(UPSAMPLE_LEVEL, shuffle=True)
print("Train X shape: {}".format(X.shape))
print("Train Y shape: {}".format(Y.shape))
#
# Make Test set
TX, TY = test_wg.test_set()
print("Test X shape: {}".format(TX.shape))
print("Test Y shape: {}".format(TY.shape))

# Model
model = AttentionLSTM(units=HIDDEN_UNITS, BATCH_SIZE=BATCH_SIZE, output_seq=FUTURE_DATA)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MSE,
              metrics=['mape', 'mae'])

# Training Callbacks
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
tb = TensorBoard('./logs')

history = model.fit(X,
                    Y,
                    epochs=EPOCHS,
                    callbacks=[es, rlr, tb],
                    validation_split=0.2,
                    verbose=1,
                    batch_size=BATCH_SIZE)

# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=SHIFT + FUTURE_DATA, freq='1d').tolist()

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())


# Training set performance
test_x, test_y = train_wg.test_set()
train_result, train_real = evaluation(test_x, test_y, model, datelist_train)
print(train_result.head(10))
print(train_real.head(10))
# Plot
img_path = "./result-{}-{}-{}-ALL".format(PAST_DATA, SHIFT, FUTURE_DATA)
plot(train_real, train_result, show=True, save_path=img_path)

if TEST_TYPE != 2:
    # Perform predictions
    test_result, test_real = evaluation(TX, TY, model, datelist_test)
    print(test_result.head(10))
    print(test_real.head(10))
    # Plot
    img_path = "./result-{}-{}-{}-TEST".format(PAST_DATA, SHIFT, FUTURE_DATA)
    plot(test_real, test_result, show=True, save_path=img_path)
