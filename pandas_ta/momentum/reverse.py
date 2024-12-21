# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame, Series

from pandas_ta.overlap import ma
from pandas_ta.momentum import roc
from pandas_ta.utils import get_offset, verify_series

from pandas_ta.utils import signed_series
from .signal import signal
from .volume import vdma


def anchor_reverse(high: Series, low: Series, close: Series, short=10, long=63):
    """锚定反转因子
    
    - reference：锚定反转因子构建与增强|中银量化多因子选股系列
    """
    _is_short_up = signed_series(close, period=short)

    _long_hhv = high.rolling(long).max()
    _long_llv = low.rolling(long).min()
    _anchor_price = np.where(_is_short_up == 1, _long_llv, _long_hhv)
    return close / _anchor_price - 1


def anchor_rev_s(*args, period=126, top_k=0.8, bot_k=0.2, **kwargs):
    _anchor_rev = anchor_reverse(*args, **kwargs)
    return -1 * signal(_anchor_rev, period, top_k=top_k, bot_k=bot_k)


def anchor_reverse_std(high: Series, low: Series, close: Series, short=10, long=63):
    _anchor_rev = anchor_reverse(high, low, close, short=short, long=long)
    _close_std = close.rolling(long).std()
    return _anchor_rev * _close_std


def anchor_rev_std_s(*args, period=126, top_k=0.8, bot_k=0.2, **kwargs):
    _anchor_rev_std = anchor_reverse_std(*args, **kwargs)
    _arev_std_s = -1 * signal(_anchor_rev_std, period, top_k, bot_k)
    return _arev_std_s


def anchor_rev_std_vdma_s(high, low, close, volume, short=10, long=63, **kwargs):
    _anchor_rev_std_s = anchor_rev_std_s(high, low, close, short=short, long=long, **kwargs)
    _vdma = vdma(volume, fast=short, slow=long)
    return _anchor_rev_std_s * _vdma

def anchor_rev_std_vddma_s(high, low, close, volume, short=10, long=63, 
                           period=126, top_k=0.95, bot_k=0.05, 
                           fast=10, slow=63, offset=10,
                           **kwargs):
    _anchor_rev_std_vdma_s = anchor_rev_std_vdma_s(
        high, low, close, volume, short=short, long=long, 
        period=period, top_k=top_k, bot_k=bot_k, **kwargs
    )
    _ref_vdma = vdma(volume, fast, slow).shift(offset)
    return _anchor_rev_std_vdma_s * _ref_vdma
