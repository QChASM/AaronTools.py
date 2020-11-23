import collections.abc
import copy
import os
import re
from collections import OrderedDict

import AaronTools.atoms as Atoms
import numpy as np
from AaronTools.const import AARONTOOLS


def progress_bar(this, max_num, name=None, width=50):
    if name is None:
        s = ""
    else:
        s = "{}: ".format(name)
    s += "Progress {:3.0f}% ".format(100 * this / max_num)
    s += "|{:<{width}s}|".format(
        "#" * int(width * this / max_num), width=width
    )
    print(s, end="\r")


def clean_progress_bar(width=50):
    print(" " * 2 * width, end="\r")


def proj(v, u):
    """projection of u into v"""
    numerator = np.dot(u, v)
    denominator = np.linalg.norm(v) ** 2
    return numerator * v / denominator


def quat_matrix(pt1, pt2):
    """ build quaternion matrix from pt1 and pt2 """
    pt1 = np.array(pt1, dtype=np.longdouble)
    pt2 = np.array(pt2, dtype=np.longdouble)
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
    return matrix.astype(np.double)


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


def add_dict(this, other, skip=[]):
    for key, val in other.items():
        if key in skip:
            continue
        if key in this and isinstance(val, dict):
            add_dict(this[key], val)
        else:
            this[key] = val
    return this


def resolve_concatenation(*args):
    seq = [isinstance(a, collections.abc.MutableSequence) for a in args]
    if any(seq) and not all(seq):
        rv = []
        for i, s in enumerate(seq):
            if s:
                rv.extend(args[i])
            else:
                rv.append(args[i])
        return rv
    else:
        err_msg = "Cannot concatenate" + " {}" * len(args)
        raise TypeError(err_msg.format(*[type(a) for a in args]))


def combine_dicts(*args, case_sensitive=False, dict2_conditional=False):
    """combine dictionaries d1 and d2 to return a dictionary
    with keys d1.keys() + d2.keys()
    if a key is in d1 and d2, the items will be combined:
        if they are both dictionaries, combine_dicts is called recursively
        otherwise, d2[key] is appended to d1[key]
        if case_sensitive=False, the key in the output will be the lowercase
        of the d1 key and d2 key (only for combined items)
    dict2_conditional: bool - if True, don't add d2 keys unless they are also in d1
    """
    d1 = args[0]
    d2 = args[1:]
    if len(d2) > 1:
        d2 = combine_dicts(
            d2[0],
            *d2[1:],
            case_sensitive=case_sensitive,
            dict2_conditional=dict2_conditional
        )
    else:
        d2 = d2[0]

    out = OrderedDict()
    case_keys_1 = list(d1.keys())
    case_keys_2 = list(d2.keys())
    if case_sensitive:
        keys_1 = case_keys_1
        keys_2 = case_keys_2
    else:
        keys_1 = [
            key.lower() if isinstance(key, str) else key for key in case_keys_1
        ]
        keys_2 = [
            key.lower() if isinstance(key, str) else key for key in case_keys_2
        ]

    # go through keys from d1
    for case_key, key in zip(case_keys_1, keys_1):
        # if the key is only in d1, add the item to out
        if key in keys_1 and key not in keys_2:
            out[case_key] = d1[case_key]
        # if the key is in both, combine the items
        elif key in keys_1 and key in keys_2:
            key_2 = case_keys_2[keys_2.index(key)]
            if isinstance(d1[case_key], dict) and isinstance(d2[key_2], dict):
                out[key] = combine_dicts(
                    d1[case_key],
                    d2[key_2],
                    case_sensitive=case_sensitive,
                )
            else:
                try:
                    out[key] = d1[case_key] + d2[key_2]
                except TypeError:
                    out[key] = resolve_concatenation(d1[case_key], d2[key_2])

    # go through keys from d2
    if not dict2_conditional:
        for case_key, key in zip(case_keys_2, keys_2):
            # if it's only in d2, add item to out
            if key in keys_2 and key not in keys_1:
                out[case_key] = d2[case_key]

    return out


def integrate(fun, a, b, n=101):
    """numerical integration using Simpson's method
    fun - function to integrate
    a - integration starts at point a
    b - integration stops at point b
    n - number of points used for integration"""
    import numpy as np

    dx = float(b - a) / (n - 1)
    x_set = np.linspace(a, b, num=n)
    s = -(fun(a) + fun(b))
    i = 1
    max_4th_deriv = 0
    while i < n:
        if i % 2 == 0:
            s += 2 * fun(x_set[i])
        else:
            s += 4 * fun(x_set[i])

        if i < n - 4 and i >= 3:
            sg_4th_deriv = (
                6 * fun(x_set[i - 3])
                + 1 * fun(x_set[i - 2])
                - 7 * fun(x_set[i - 1])
                - 3 * fun(x_set[i])
                - 7 * fun(x_set[i + 1])
                + fun(x_set[i + 2])
                + 6 * fun(x_set[i + 3])
            )
            sg_4th_deriv /= 11 * dx ** 4

            if abs(sg_4th_deriv) > max_4th_deriv:
                max_4th_deriv = abs(sg_4th_deriv)

        i += 1

    s = s * dx / 3.0

    # close enough error estimate
    e = (abs(b - a) ** 5) * max_4th_deriv / (180 * n ** 4)

    return (s, e)


def same_cycle(graph, a, b):
    """
    Determines if Atom :a: and Atom :b: are in the same cycle in a undirected :graph:

    Returns: True if cycle found containing a and b, False otherwise

    :graph: connectivity matrix or Geometry
    :a:, :b: indices in connectivity matrix/Geometry or Atoms in Geometry
    """
    from AaronTools.geometry import Geometry

    if isinstance(a, Atoms.Atom):
        a = graph.atoms.index(a)
    if isinstance(b, Atoms.Atom):
        b = graph.atoms.index(b)
    if isinstance(graph, Geometry):
        graph = [
            [graph.atoms.index(j) for j in i.connected] for i in graph.atoms
        ]
    graph = [[i for i in j] for j in graph]

    graph, removed = trim_leaves(graph)

    if a in removed or b in removed:
        return False
    path = shortest_path(graph, a, b)
    for p, q in zip(path[:-1], path[1:]):
        graph[p].remove(q)
        graph[q].remove(p)

    path = shortest_path(graph, a, b)
    if path is None:
        return False
    return True


def shortest_path(graph, start, end):
    """
    Find shortest path from :start: to :end: in :graph: using Dijkstra's algorithm
    Returns: list(node_index) if path found, None if path not found

    :graph: the connection matrix or Geometry
    :start: the first atom or node index
    :end: the last atom or node index
    """
    from AaronTools.geometry import Geometry

    if isinstance(start, Atoms.Atom):
        start = graph.atoms.index(start)
    if isinstance(end, Atoms.Atom):
        end = graph.atoms.index(end)
    if isinstance(graph, Geometry):
        graph = [
            [graph.atoms.index(j) for j in i.connected if j in graph.atoms]
            for i in graph.atoms
        ]
    graph = [[i for i in j] for j in graph]

    # initialize distance array, parent array, and set of unvisited nodes
    dist = [np.inf for x in graph]
    parent = [-1 for x in graph]
    unvisited = set([i for i in range(len(graph))])

    dist[start] = 0
    current = start
    while True:
        # for all unvisited neighbors of current node, update distances
        # if we update the distance to a neighboring node,
        # then also update its parent to be the current node
        for v in graph[current]:
            if v not in unvisited:
                continue
            if dist[v] == np.inf:
                new_dist = dist[current] + 1
            else:
                new_dist = dist[current] + dist[v]
            if dist[v] > new_dist:
                dist[v] = new_dist
                parent[v] = current
        # mark current node as visited
        # select closest unvisited node to be next node
        # break loop if we found end node or if no unvisited connected nodes
        unvisited.remove(current)
        if end not in unvisited:
            break
        current = None
        for u in unvisited:
            if current is None or dist[u] < dist[current]:
                current = u
        if dist[current] == np.inf:
            break
    # return shortest path from start to end
    path = [end]
    while True:
        if parent[path[-1]] == -1:
            break
        path += [parent[path[-1]]]
    path.reverse()
    if path[0] != start or path[-1] != end:
        return None
    return path


# XXX: for some reason, if _removed defaults to [], it might behave like a global variable
# def trim_leaves(graph, _removed=[]):
def trim_leaves(graph, _removed=None):
    from AaronTools.geometry import Geometry

    # print(_removed)
    if _removed is None:
        _removed = []

    if isinstance(graph, Geometry):
        graph = [
            [graph.atoms.index(j) for j in i.connected] for i in graph.atoms
        ]
    graph = [[i for i in j] for j in graph]
    some_removed = False

    for i, con in enumerate(graph):
        if len(con) == 1:
            graph[con[0]].remove(i)
            graph[i].remove(con[0])
            some_removed = True
            _removed += [i]

    if some_removed:
        graph, _removed = trim_leaves(graph, _removed)

    return graph, set(_removed)


def to_closing(s, p):
    """returns the portion of string s from the beginning to the closing
    paratheses or bracket denoted by p
    p can be '(', '{', or '['
    if the closing paratheses is not found, returns None instead"""
    if p == "(":
        q = ("(", ")")
    elif p == "{":
        q = ("{", "}")
    elif p == "[":
        q = ("[", "]")
    else:
        raise RuntimeError("p must be '(', '{', or '['")

    out = ""
    count = 0
    for x in s:
        if x == q[0]:
            count += 1
        elif x == q[1]:
            count -= 1

        out += x
        if count == 0:
            break

    if count != 0:
        return None
    else:
        return out


def rotation_matrix(theta, axis):
    """rotation matrix for rotating theta radians about axis"""
    # I've only tested this for rotations in R3
    if np.linalg.norm(axis) == 0:
        axis = [1, 0, 0]
    axis = axis / np.linalg.norm(axis)
    dim = len(axis)
    M = np.dot(np.transpose([axis]), [axis])
    M = [[i * (1 - np.cos(theta)) for i in m] for m in M]
    I = np.identity(dim)
    Cos = [[np.cos(theta) * i for i in ii] for ii in I]
    Sin = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            if i != j:
                if (i + j) % 2 != 0:
                    p = 1
                else:
                    p = -1
                if i > j:
                    p = -p
                Sin[i][j] = p * np.sin(theta) * axis[dim - (i + j)]

    return np.array(
        [
            [M[i][j] + Cos[i][j] + Sin[i][j] for i in range(0, dim)]
            for j in range(0, dim)
        ]
    )


def fibonacci_sphere(radius=1, center=np.zeros(3), n=500):
    """
    returns a grid of points that are equally spaced on a sphere
    with the specified radius and center
    number of points can be adjusted with n
    """
    # generate a grid of points on the unit sphere
    grid = np.zeros((n, 3))
    d_theta = np.pi * (3.0 - np.sqrt(5.0))
    dy = 2.0 / (n - 1)

    for i in range(0, n):
        y = 1 - i * dy
        r = np.sqrt(1 - y ** 2)

        theta = i * d_theta

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        grid[i] = np.array([x, y, z])

    # scale the points to the specified radius and move the center
    grid *= radius
    grid += center

    return grid


def lebedev_sphere(radius=1, center=np.zeros(3), n=302):
    """
    returns one of the Lebedev grid points (xi, yi, zi)
    and weights (wi) with the specified radius and center.
    Weights do not include r**2, so integral of F(x,y,z)
    over sphere is 4*pi*r**2\sum_i{F(xi,yi,zi)wi}.  The number
    of points (n) must be one of 110, 194, 302, 590, 1454, 5810
    """
    # read grid data  on unit sphere
    grid_file = os.path.join(
        AARONTOOLS, "utils", "quad_grids", "Leb" + str(n) + ".grid"
    )
    grid_data = np.loadtxt(grid_file)
    grid = grid_data[:, [0, 1, 2]]
    weights = grid_data[:, 3]

    # scale the points to the specified radius and move the center
    grid *= radius
    grid += center

    return grid, weights


def gauss_legendre_grid(a=-1, b=1, n=32):
    """
    returns a Gauss-Legendre grid points (xi) and weights
    (wi)for the range a to b. Integral over F(x) is
    \sum_i{F(xi)wi}. The number of points (n) must be one
    of 20, 32, 64, 75, 99, 127
    """
    # read grid points on the range [-1,1] and weights
    grid_file = os.path.join(
        AARONTOOLS, "utils", "quad_grids", "Leg" + str(n) + ".grid"
    )
    grid_data = np.loadtxt(grid_file)

    # shift grid range to [a,b]
    grid = grid_data[:, 0] * (b - a) / 2 + a + (b - a) / 2

    # adjust weights for new range
    weights = grid_data[:, 1] * (b - a) / 2

    return grid, weights
