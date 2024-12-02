import sys
sys.path.append(r'F:\projects\quant-libs\my_pandas_ta')

import pandas_ta as ta
import numpy as np


def test_cross():
    # a = np.random.rand(10)
    # b = np.random.rand(10)
    a = np.array([1, 1, 2, 3, 4, 4, 6])
    b = np.array([0, 2, -1, -2, 6, 7, 5])
    _cross_real = np.array([0, 1, 0, 0, 1, 0, 0])
    _cross = ta.snp.cross(b, a)
    print(_cross)
    assert np.allclose(_cross_real, _cross)
