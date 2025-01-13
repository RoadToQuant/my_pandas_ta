import numpy as np
import pandas as pd

from pandas_ta import rsi as ta_rsi
from pandas_ta.momentum.rsi import rsi


def vwap(df, prices=('close',), weight='volume'):
    """给定行情数据计算指定价格标签的加权值

    :param df: 行情数据，至少包含OHLCV等基础行情数据
    :type df: pd.DataFrame
    :param prices: 价格标签，默认close收盘价
    :param weight: 权重变量，默认volume成交量
    """
    # 判断是否包含指定数据
    # assert df.columns

    # # 使用pandas函数计算
    # df[prices] = df[prices].multiply(df[weight], axis=0)
    # s_sum = df.sum(numeric_only=True)
    # s_weighted_prices = s_sum[prices] / s_sum[weight]

    # 使用numpy计算
    _weighted_prices = np.average(df[prices], weights=df[weight], axis=0)
    s_weighted_prices = pd.Series(_weighted_prices, index=prices)
    return s_weighted_prices


def roc(prices, price_labels='close', n=1, shift=0, skip=0):
    """给定数据计算百分比变化

    :param prices: 行情数据，至少包含OHLCV等基础行情数据
    :type prices: pd.DataFrame
    :param price_labels: 价格标签，默认close收盘价
    :param n: 期数，默认1期
    :param shift: 偏移量，默认0，即当期对过去n段的变化，可以为负，表示未来变化，可以进一步封装
    :param skip: 间隔量，默认0，即无间隔
    """
    # if not isinstance(price_labels, list):
    #     price_labels = [price_labels]
    if isinstance(price_labels, tuple):
        price_labels = list(price_labels)
    # TODO：支持open-to-close和vwap-to-close的计算
    return prices[price_labels].pct_change(periods=n).shift(shift + skip)


s1 = pd.Series(np.random.rand(10))
s1_rma = rsi(s1, 5)
print(s1_rma)
print(ta_rsi(s1, length=5))
