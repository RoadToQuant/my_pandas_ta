# -*- coding: utf-8 -*-
from pandas_ta.overlap import ma
from pandas_ta.momentum import roc
from pandas_ta.utils import get_offset, verify_series


def cv(high, low, fast=None, slow=None, scalar=None, mamode=None, offset=None, **kwargs):
    """Chaikin Volatility (cvlt)"""
    # Validate Arguments
    fast = int(fast) if fast and fast > 0 else 10
    slow = int(slow) if slow and slow > 0 else 10
    scalar = float(scalar) if scalar and scalar > 0 else 100
    mamode = mamode if isinstance(mamode, str) else "sma"
    _length = max(fast, slow)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)

    if high is None or low is None: return

    # Calculate Result
    rem = ma(mamode, high - low, length=fast)
    cv = scalar * roc(rem, slow, scalar=1, talib=False)

    # Offset
    if offset != 0:
        cv = cv.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        cv.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cv.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    cv.name = f"CV_{fast}_{slow}"
    cv.category = "volatility"

    return cv
