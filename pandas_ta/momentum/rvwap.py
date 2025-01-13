# -*- coding: utf-8 -*-
from pandas_ta.utils import get_offset, is_datetime_ordered, verify_series


def rvwap(close, volume, length=10, anchor=None, offset=None, **kwargs):
    """Indicator: Rolling Volume Weighted Average Price (RVWAP)"""
    # Validate Arguments
    close = verify_series(close)
    volume = verify_series(volume)
    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
    offset = get_offset(offset)
    # Calculate Result
    wp = close * volume
    rvwap = wp.rolling(min_periods=1, window=length).sum()
    rvwap /= volume.rolling(min_periods=1, window=length).sum()

    # Offset
    if offset != 0:
        rvwap = rvwap.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        rvwap.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        rvwap.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    rvwap.name = f"RVWAP_{anchor}"
    rvwap.category = "momentumr"

    return rvwap


rvwap.__doc__ = \
    """Rolling Volume Weighted Average Price (VWAP)
    
    The Rolling Volume Weighted Average Price that measures the average typical price
    by volume.  It is typically used with intraday charts to identify general
    direction.
    
    Sources:
        https://www.tradingview.com/wiki/Volume_Weighted_Average_Price_(VWAP)
        https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/volume-weighted-average-price-vwap/
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vwap_intraday
    
    Calculation:
        closev = close * volume
        VWAP = closev.cumsum() / volume.cumsum()
    
    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        volume (pd.Series): Series of 'volume's
        anchor (str): How to anchor VWAP. Depending on the index values, it will
            implement various Timeseries Offset Aliases as listed here:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
            Default: "D".
        offset (int): How many periods to offset the result. Default: 0
    
    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method
    
    Returns:
        pd.Series: New feature generated.
    """
