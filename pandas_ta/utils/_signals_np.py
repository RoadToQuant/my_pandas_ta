from sys import float_info as sflt
import numpy as np


def zero(arr: np.ndarray):
    """If the value is close to zero, then return zero. Otherwise return itself."""
    _arr = np.where(abs(arr) < sflt.epsilon, 0, arr)
    return _arr


def shift(arr: np.ndarray, n: int, fill=0):
    _arr = np.zeros_like(arr)
    
    if n > 0:
        _arr[n:] = arr[:-1 * n]
        _arr[:n] = fill
    elif n < 0:
        l = len(_arr)
        _arr[:n] = arr[-1 * n:]
        _arr[n:] = fill
    else:
        _arr = arr
    return _arr


def cross(arr_a: np.ndarray, arr_b: np.ndarray, above: bool = True, asint: bool = True, offset: int = None):
    offset = offset if offset is not None else 0

    arr_a = zero(arr_a)
    arr_b = zero(arr_b)

    # Calculate Result
    current = arr_a > arr_b  # current is above
    previous = shift(arr_a, 1) < shift(arr_b, 1)  # previous is below
    # above if both are true, below if both are false
    cross = current & previous if above else ~current & ~previous

    if asint:
        cross = cross.astype(int)

    # Offset
    if offset != 0:
        cross = shift(cross, shift)

    # Name & Category
    _name = f"a_{'XA' if above else 'XB'}_b"
    return cross
