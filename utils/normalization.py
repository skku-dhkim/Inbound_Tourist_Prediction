import pandas as pd


def min_max(d, references=None):
    if references:
        min_v = pd.read_csv(references[0], header=None).set_index(0, drop=True)
        min_v = min_v.to_numpy().squeeze()
        max_v = pd.read_csv(references[1], header=None).set_index(0, drop=True)
        max_v = max_v.to_numpy().squeeze()
    else:
        min_v = d.min(axis=0)
        max_v = d.max(axis=0)
    d = (d - min_v) / (max_v - min_v)
    return d
