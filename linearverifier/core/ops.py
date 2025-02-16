"""
This module defines common operations for the verification algorithm
"""
import copy


def get_positive(a: list[list[float]]) -> list[list[float]]:
    """Procedure to extract the positive part of a matrix"""
    result = copy.deepcopy(a)

    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] < 0:
                result[i][j] = 0

    return result


def get_negative(a: list[list[float]]) -> list[list[float]]:
    """Procedure to extract the negative part of a matrix"""
    result = copy.deepcopy(a)

    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] > 0:
                result[i][j] = 0

    return result


def matmul_mixed(a: list[list[float]], b: list[float]) -> list[float]:
    """Procedure to compute the matrix product of a 2-d matrix and a vector"""

    assert len(a[0]) == len(b)

    n = len(a)
    m = len(b)

    result = [0 for _ in range(len(a))]

    for i in range(n):
        for j in range(m):
            result[i] += a[i][j] * b[j]

    return result


def matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
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


def create_disjunction_matrix(n_outs: int, label: int) -> list[list[float]]:
    """Procedure to create the matrix of the output property"""
    matrix = []
    c = 0
    for i in range(n_outs):
        if i != label:
            matrix.append([0 for _ in range(n_outs)])
            matrix[c][label] = 1
            matrix[c][i] = -1
            c += 1

    return matrix


def check_unsafe(bounds: dict, matrix: list[float]) -> bool:
    """Procedure to check whether the output bounds are unsafe"""

    m_plus = [m if m > 0 else 0 for m in matrix]
    m_minus = [m if m < 0 else 0 for m in matrix]
    lbs = bounds['lower']
    ubs = bounds['upper']

    min_value = (sum([m_plus[i] * lbs[i] for i in range(len(lbs))]) +
                 sum([m_minus[i] * ubs[i] for i in range(len(ubs))]))

    # Since we're linear we know for sure this is enough
    if min_value > 1e-6:
        return False
    else:
        return True
