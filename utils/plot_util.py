import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot(TRUE, df_train, df_prediction, save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(50, 10))

    plt.plot(TRUE.loc[:].index,
             TRUE.loc[:]["True"],
             color='b',
             label='TRUE')
    plt.plot(df_train.index,
             df_train.loc[:]['Train'],
             color='orange',
             label='True')
    plt.plot(df_prediction.index,
             df_prediction.loc[:]['Test'],
             color='red',
             label='Training predictions')

    plt.grid(which='major', color='#cccccc', alpha=0.5)
    plt.legend(shadow=True)
    plt.title('Total_entry, KOSPI, Oil_price', fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
