from pathlib import Path
import pandas as pd
import numpy as np

COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

def load_train(path="data/raw/train_FD001.txt"):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLS
    return df

def add_rul(df):
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = max_cycle - out["cycle"]
    return out

def add_features(df):
    out = df.copy()
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    for c in sensor_cols:
        out[f"{c}_delta"] = out.groupby("unit")[c].diff().fillna(0)
        out[f"{c}_roll_mean_10"] = (
            out.groupby("unit")[c].transform(lambda s: s.rolling(10, min_periods=1).mean())
        )
        out[f"{c}_roll_std_10"] = (
            out.groupby("unit")[c].transform(lambda s: s.rolling(10, min_periods=1).std())
            .fillna(0)
        )
    return out

def feature_columns():
    base = ["cycle", "setting1", "setting2", "setting3"]
    sensors = [f"s{i}" for i in range(1, 22)]
    extra = []
    for c in sensors:
        extra += [f"{c}_delta", f"{c}_roll_mean_10", f"{c}_roll_std_10"]
    return base + sensors + extra
