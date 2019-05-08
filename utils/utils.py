import numpy as np


def progress_bar(this, max_num, name=None, width=50):
    if name is None:
        s = ""
    else:
        s = "{}: ".format(name)
    s += "Progress {:3.0f}% ".format(100 * this / max_num)
    s += "|{:<{width}s}|".format("#" * (width * this // max_num), width=width)
    print(s, end="\r")


def clean_progress_bar(width=50):
    print(" " * 2 * width, end="\r")


def quat_matrix(pt1, pt2):
    """ build quaternion matrix from pt1 and pt2 """
    pt1 = np.array(pt1, dtype=np.double)
    pt2 = np.array(pt2, dtype=np.double)
    for pt in [pt1, pt2]:
        if len(pt.shape) != 1 or pt.shape[0] != 3:
            raise ValueError("Arguments should be 3-element vectors")

    xm, ym, zm = tuple(pt1 - pt2)
    xp, yp, zp = tuple(pt1 + pt2)

    matrix = np.array(
        [
            [
                xm * xm + ym * ym + zm * zm,
                yp * zm - ym * zp,
                xm * zp - xp * zm,
                xp * ym - xm * yp,
            ],
            [
                yp * zm - ym * zp,
                yp * yp + zp * zp + xm * xm,
                xm * ym - xp * yp,
                xm * zm - xp * zp,
            ],
            [
                xm * zp - xp * zm,
                xm * ym - xp * yp,
                xp * xp + zp * zp + ym * ym,
                ym * zm - yp * zp,
            ],
            [
                xp * ym - xm * yp,
                xm * zm - xp * zp,
                ym * zm - yp * zp,
                xp * xp + yp * yp + zm * zm,
            ],
        ]
    )
    return matrix


def uptri2sym(vec, n=None, col_based=False):
    """
    Converts upper triangular matrix to a symmetric matrix

    :vec: the upper triangle array/matrix
    :n: the number of rows/columns
    :col_based: if true, triangular matirx is of the form
                    0 1 3
                    - 2 4
                    - - 5
                if false, triangular matrix is of the form
                    0 1 2
                    - 3 4
                    - - 5
    """
    if hasattr(vec[0], "__iter__") and not isinstance(vec[0], str):
        tmp = []
        for v in vec:
            tmp += v
        vec = tmp
    if n is None:
        n = -1 + np.sqrt(1 + 8 * len(vec))
        n = int(n / 2)
    if n * (n + 1) / 2 != len(vec):
        raise RuntimeError("Bad number of rows requested")

    matrix = np.zeros((n, n))
    if col_based:
        i = 0  # vector index
        j = 0  # for column counting
        for c in range(n):
            j += 1
            for r in range(n):
                matrix[r, c] = vec[i]
                matrix[c, r] = vec[i]
                i += 1
                if r + 1 == j:
                    break
    else:
        for r in range(n):
            for c in range(r, n):
                i = n * r + c - r * (1 + r) / 2
                matrix[r, c] = vec[int(i)]
                matrix[c, r] = vec[int(i)]

    return matrix


def float_vec(word):
    """
    Turns strings into floatin point vectors
    :word: a comma-delimited string of numbers

    if no comma or only one element:
        returns just the floating point number
    if elements in word are strings:
        returns word unchanged
    else:
        returns a np.array() of floating point numbers
    """
    val = word
    val = val.split(",")
    try:
        val = [float(v) for v in val]
    except ValueError:
        val = word
    if len(val) == 1:
        return val[0]
    else:
        return np.array(val)


def is_alpha(test):
    rv = re.search("^[a-zA-Z]+$", test)
    return bool(rv)


def is_int(test):
    rv = re.search("^[+-]?\d+$", test)
    return bool(rv)


def is_num(test):
    rv = re.search("^[+-]?\d+\.?\d*", test)
    return bool(rv)
