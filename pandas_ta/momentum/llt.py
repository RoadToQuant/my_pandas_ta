
import pandas as pd


def get_llt(prices: pd.Series, alpha: float) -> pd.Series:
    
    llt = pd.Series(index=prices.index, dtype='float64')

    # 需要至少两个价格点来计算LLT
    if len(prices) < 2:
        return prices

    # 初始化LLT的前两个值
    llt[0] = prices[0]
    llt[1] = prices[1]

    # 使用给定的公式计算接下来的LLT值
    for t in range(2, len(prices)):
        llt[t] = ((alpha - alpha**2 / 4) * prices[t] +
                  (alpha**2 / 2) * prices[t-1] -
                  (alpha - 3 * alpha**2 / 4) * prices[t-2] +
                  2 * (1 - alpha) * llt[t-1] -
                  (1 - alpha)**2 * llt[t-2])
    
    return llt

