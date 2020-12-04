from sqlalchemy import create_engine
from config.db_config import mysql_info
import pandas as pd
import numpy as np
from models.DA_RNN import DA_rnn
import matplotlib.pyplot as plt

# DB connection
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       .format(user=mysql_info['user'], pw=mysql_info['pw'], host=mysql_info['host'], db=mysql_info['db']))

# Load data from DB
df = pd.read_sql_query("select "
                       "Date, "
                       "Total_entry, "
                       "KOSPI, "
                       "Oil_price, "
                       "China_exchange_rate, "
                       "Japan_exchange_rate, "
                       "USA_exchange_rate, "
                       "Japan_GDP, "
                       "China_GDP, "
                       "USA_GDP from T_DAILY_FOREIGNERS", engine)
df.set_index("Date", inplace=True, drop=True)
df = (df-df.min())/(df.max()-df.min())
X = df.loc[:, [x for x in df.columns.tolist() if x != 'Total_entry']].values
y = np.array(df.Total_entry)

batchsize = 32
nhidden_encoder = 64
nhidden_decoder = 64
ntimestep = 10
lr = 0.001
epochs = 1

# Initialize model
print("==> Initialize DA-RNN model ...")
model = DA_rnn(
    X,
    y,
    ntimestep,
    nhidden_encoder,
    nhidden_decoder,
    batchsize,
    lr,
    epochs
)

# Train
print("==> Start training ...")
model.train()

# Prediction
y_pred = model.test()

fig1 = plt.figure()
plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
plt.savefig("1.png")
plt.close(fig1)

fig2 = plt.figure()
plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
plt.savefig("2.png")
plt.close(fig2)

fig3 = plt.figure()
plt.plot(y_pred, label='Predicted')
plt.plot(model.y[model.train_timesteps:], label="True")
plt.legend(loc='upper left')
plt.savefig("3.png")
plt.close(fig3)
print('Finished Training')
