import pandas as pd
from utils.preprocess import WindowGenerator
from models.attentionLSTM import AttentionLSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import matplotlib.dates as mdates

# Load data from CSV
df = pd.read_csv("./data/tourist_daily.csv")
df = df[["Date", "Total_entry", "KOSPI", "Oil_price", "China_exchange_rate", "Japan_exchange_rate", "USA_exchange_rate",
         "Japan_GDP", "China_GDP", "USA_GDP"]]

datelist_train = list(df['Date'])
datelist_train = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]
date = df['Date']

df = df.drop('Date', axis=1)
print('Training set shape == {}'.format(df.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print(df.head())

'Hyper-parameters'
PAST_DATA = 10
FUTURE_DATA = 1
SHIFT = 7
BATCH_SIZE = 32
HIDDEN_UNITS = 64
EPOCHS = 30

wg = WindowGenerator(input_width=PAST_DATA, label_width=FUTURE_DATA, shift=SHIFT, label_columns=['Total_entry'],
                     data_df=df)

print(wg)
# Augmentation and Make Train set
X, Y = wg.augmentation(10, shuffle=True)
print(X.shape)
print(Y.shape)

# Model
model = AttentionLSTM(units=HIDDEN_UNITS, BATCH_SIZE=BATCH_SIZE, output_seq=FUTURE_DATA)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MAE,
              metrics=['mape', 'mse'])

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

test_x, test_y = wg.test_set()

# Perform predictions
predictions = model.predict(test_x)
print(test_x.shape)
print(predictions.shape)

df_prediction = pd.DataFrame(predictions, columns=['Total_entry']).set_index(pd.Series(datelist_train[PAST_DATA
                                                                                                      + SHIFT
                                                                                                      + FUTURE_DATA - 1:]))

df_real = pd.DataFrame(test_y, columns=['Total_entry']).set_index(pd.Series(datelist_train[PAST_DATA
                                                                                           + SHIFT
                                                                                           + FUTURE_DATA - 1:]))
print(df_prediction.head(10))
print(df_real.head(10))

fig, ax = plt.subplots(figsize=(50, 10))

# Plot parameters
START_DATE_FOR_PLOTTING = '2010-01-01'

plt.plot(df_real.index,
         df_real.loc[:]['Total_entry'],
         color='b',
         label='True')
plt.plot(df_prediction.index,
         df_prediction.loc[:]['Total_entry'],
         color='orange',
         label='Training predictions')

plt.grid(which='major', color='#cccccc', alpha=0.5)
plt.legend(shadow=True)
plt.title('Total_entry, KOSPI, Oil_price', fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.grid(True)
plt.savefig("./result-{}-{}-{}".format(PAST_DATA, SHIFT, FUTURE_DATA))
# plt.show()
