import pandas as pd
import numpy as np


def rvi(src, length, mamode='sma', scalar=1):
    _diff = src - src.shift(1)
    _pos = pd.Series(np.where(_diff > 0, 1, 0))
    _neg = pd.Series(np.where(_diff < 0, 1, 0))
    _std = _diff.rolling(length).std()
    if mamode.upper() == 'EMA':
        _pos_std = (_pos * _std).ewm(alpha=1/length).mean()
        _neg_std = (_neg * _std).ewm(alpha=1/length).mean()
    else:  # 默认SMA
        _pos_std = (_pos * _std).rolling(length).mean()
        _neg_std = (_neg * _std).rolling(length).mean()
    _rvi = scalar * _pos_std / (_pos_std + _neg_std)
    return _rvi
