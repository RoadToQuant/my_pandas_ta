import numpy as np
from pandas import DataFrame, Series

from pandas_ta.overlap import ma
from pandas_ta.momentum import roc
from pandas_ta.utils import get_offset, verify_series


def vdma(volume: Series, fast=5, slow=20, mamode='sma'):
    _f_vol: Series = ma(mamode, volume, length=fast)
    _s_vol: Series = ma(mamode, volume, length=slow)
    return (_f_vol > _s_vol).map(int)
