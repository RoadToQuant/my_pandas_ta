import numpy as np
import pandas as pd

from pandas_ta import rsi as ta_rsi
from pandas_ta.momentum.rsi import rsi


s1 = pd.Series(np.random.rand(10))
s1_rma = rsi(s1, 5)
print(s1_rma)
print(ta_rsi(s1, length=5))
