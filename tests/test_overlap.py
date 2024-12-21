import numpy as np
import pandas as pd

from pandas_ta.overlap.rma import rma


s1 = pd.Series(np.random.rand(10))
s1_rma = rma(s1, 5)
print(s1_rma)
