"""
This module defines common operations
"""
from enum import Enum

from mpmath import mp


class PropertySatisfied(Enum):
    No = 0
    Yes = 1
    Maybe = 2


def get_other_ubs(m: mp.matrix, index: int) -> mp.matrix:
    """Procedure to remove an element from a mp.matrix"""
    result = mp.matrix(m.rows - 1, 1)
    cnt = 0
    for i in range(m.rows):
        if i != index:
            result[cnt] = m[i]
            cnt += 1

    return result


def max_upper(l: mp.matrix) -> mp.mpf:
    m = mp.mpf(-10000)

    for i in range(l.rows):
        if l[i] > m:
            m = l[i]
    return m


def get_positive(a: mp.matrix) -> mp.matrix:
    """Procedure to extract the positive part of a matrix"""
    result = mp.matrix(a.rows, a.cols)

    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] if a[i, j] > 0 else mp.mpf(0)

    return result


def get_negative(a: mp.matrix) -> mp.matrix:
    """Procedure to extract the negative part of a matrix"""
    result = mp.matrix(a.rows, a.cols)

    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] if a[i, j] < 0 else mp.mpf(0)

    return result


def add(a: list, b: list, *args):
    """Procedure to sum lists of intervals"""
    result = []

    for i in range(len(a)):
        result.append(a[i][0] + b[i][0])
        for x in args:
            result[i] += x[i][0]

    return result


def matmul(a: list, b: list):
    """Procedure to multiply two 2-dimensional matrices"""

    assert len(a[0]) == len(b)

    n = len(a)
    m = len(b)
    q = len(b[0])

    result = [[0 for _ in range(q)] for _ in range(n)]

    for i in range(n):
        for j in range(q):
            s = 0
            for k in range(m):
                s += a[i][k] * b[k][0]
            result[i][j] = s

    return result


def create_disjunction_matrix(n_outs: int, label: int) -> list[mp.matrix]:
    """Procedure to create the matrix of the output property"""
    matrix = []
    c = 0
    for i in range(n_outs):
        if i != label:
            matrix.append(mp.matrix(n_outs, 1))
            matrix[c][label] = 1
            matrix[c][i] = -1
            c += 1

    return matrix


def check_unsafe(bounds: dict, matrix: mp.matrix) -> int:
    """Procedure to check whether the output bounds are unsafe"""

    possible_counter_example = False

    disj_res = check_satisfied(bounds, matrix)

    if disj_res == PropertySatisfied.Yes:
        # We are 100% sure there is a counter-example.
        # It can be any point from the input space.
        # Return anything from the input bounds
        # input_bounds = nn_bounds.numeric_pre_bounds[nn.get_first_node().identifier]
        return 1  # , list(input_bounds.get_lower())
    elif disj_res == PropertySatisfied.Maybe:
        # We are not 100% sure there is a counter-example.
        # Call an LP solver when we need a counter-example
        possible_counter_example = True
        print('Maybe unsafe')
    else:  # disj_res == PropertySatisfied.No
        # nothing to be done. Maybe other disjuncts will be satisfied
        pass

    # At least for one disjunct there is a possibility of a counter-example.
    # Do a more powerful check with an LP solver
    if possible_counter_example:
        return 0  # intersect_abstract_milp(star, nn, nn_bounds, prop)

        # Every disjunction is definitely not satisfied.
        # So we return False.
    return -1  # , []


def check_satisfied(bounds: dict, matrix: mp.matrix):
    """
    Checks if the bounds satisfy the conjunction of constraints given by

        matrix * x <= 0

    Returns
    -------
    Yes if definitely satisfied
    No if definitely not satisfied
    Maybe when unsure
    """

    max_value = get_positive(matrix).T * bounds['upper'] + get_negative(matrix).T * bounds['lower']
    min_value = get_positive(matrix).T * bounds['lower'] + get_negative(matrix).T * bounds['upper']

    if min_value[0] > 1e-6:
        # the constraint j is definitely not satisfied, as it should be <= 0
        return PropertySatisfied.No
    elif max_value[0] > 1e-6:
        # the constraint j might not be satisfied, but we are not sure
        return PropertySatisfied.Maybe
    else:
        # if we reached here, means that all max values were below 0
        # so we now for sure that the property was satisfied
        # and there is a counter-example (any point from the input bounds)
        return PropertySatisfied.Yes
