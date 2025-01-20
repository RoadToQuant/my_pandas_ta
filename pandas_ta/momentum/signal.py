# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame, Series

from pandas_ta.overlap import ma
from pandas_ta.momentum import roc
from pandas_ta.utils import get_offset, verify_series, signed_series


def signal(factor: Series, period=126, top_k=0.8, bot_k=0.2) -> Series:
    v_q_upper = factor.rolling(period, min_periods=period).quantile(top_k)
    v_q_lower = factor.rolling(period, min_periods=period).quantile(bot_k)

    arr_s_sig = np.where(factor >= v_q_upper, 1, 
                         np.where(factor <= v_q_lower, -1, 0))
    s_sig = Series(arr_s_sig, index=factor.index).dropna()
    return s_sig
