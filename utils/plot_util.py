import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot(df_real, df_prediction, save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(50, 10))
    #
    # # Plot parameters
    # START_DATE_FOR_PLOTTING = '2010-01-01'

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

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
