import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataLoader:
    def __init__(self):
        self.original = None  # Original data
        self._scaler = None  # Scaling Inst
        self.scaler = None

    def load_csv(self, path: str, features: list, split_date: str, with_datelist=True):
        df = pd.read_csv(path)
        df = df[features]

        date_list = list(df['Date'])
        date_list = [datetime.strptime(date, '%Y-%m-%d').date() for date in date_list]

        df = df.set_index(df['Date'])

        train = df.loc[:split_date].copy()
        test = df.loc[split_date:].copy()

        datelist_train = list(train['Date'])
        datelist_train = [datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]
        datelist_test = list(test['Date'])
        datelist_test = [datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_test]

        train.drop('Date', axis=1, inplace=True)
        test.drop('Date', axis=1, inplace=True)

        train = train.astype(float)
        test = test.astype(float)

        # Using multiple features (predictors)
        train_set = train.values
        test_set = test.values

        self.original = df.drop('Date', axis=1).copy()
        self.original = self.original.astype(float).values

        if with_datelist:
            return train_set, test_set, [date_list, datelist_train, datelist_test]
        else:
            return train_set, test_set

    def standardScale(self, d):
        if self.scaler is not StandardScaler:
            self._scaler = StandardScaler()
            self.scaler = StandardScaler()
            self._scaler.fit(self.original)
            self.scaler.fit(self.original[:, 0:1])
        data = self._scaler.transform(d)
        return data

    def minmaxScale(self, d):
        if self.scaler is not MinMaxScaler:
            self._scaler = StandardScaler()
            self.scaler = StandardScaler()
            self._scaler.fit(self.original)
            self.scaler.fit(self.original[:, 0:1])
        data = self._scaler.transform(d)
        return data

    def inverseScale(self, d):
        data = self.scaler.inverse_transform(d)
        return data

