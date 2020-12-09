def min_max(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df
