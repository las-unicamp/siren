from typing import TypedDict

import numpy as np

from other_methods.my_types import Array2Dbool


class OOBReturnType(TypedDict):
    b_e: Array2Dbool
    b_w: Array2Dbool
    b_n: Array2Dbool
    b_s: Array2Dbool


def handle_oob(shape: tuple[int, int]) -> OOBReturnType:
    b_e = np.zeros(shape, dtype=bool)
    b_e[-1, :] = True
    # b_e[:, -1] = True

    b_w = np.zeros(shape, dtype=bool)
    b_w[0, :] = True
    # b_w[:, 0] = True

    b_n = np.zeros(shape, dtype=bool)
    b_n[:, -1] = True
    # b_n[-1, :] = True

    b_s = np.zeros(shape, dtype=bool)
    b_s[:, 0] = True
    # b_s[0, :] = True

    return {
        "b_e": b_e,
        "b_n": b_n,
        "b_s": b_s,
        "b_w": b_w,
    }
