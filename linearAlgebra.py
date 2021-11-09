import numpy as np
import copy
import math
from numpy.core.fromnumeric import shape

wiki = np.array(
    [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])


def applyPermutation(matrix, permutations):
    matrix_size = len(permutations)
    ret = np.zeros(shape=(matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            ret[i][j] = matrix[permutations[i]][permutations[j]]
    return ret


def myCholeskyImplementation(m):
    ret = np.zeros(shape=m.shape)

    for row in range(len(m)):
        for col in range(row, len(m)):
            sum_temp = np.float32()
            for ind in range(row):
                sum_temp += ret[ind][row] * ret[ind][col]
            temp = m[row][col] - sum_temp

            if row == col:
                ret[row][col] = math.sqrt(temp)
            else:
                ret[row][col] = temp / ret[row][row]
    return ret


print('My implementation of Cholesky decomposition = ')
wikiDecomposed = myCholeskyImplementation(wiki)
print(wikiDecomposed)
print('\n')

"""
Simple way of swapping using numpy arrays

wikiDecomposed[:, [0, 2]] = wikiDecomposed[:, [2, 0]]
print(wikiDecomposed)
print('\n')

wikiDecomposed[[0, 1], :] = wikiDecomposed[[1, 0], :]
print(wikiDecomposed)
print('\n')
"""


def myPivotedCholeskyImplementation(m):
    mat = copy.deepcopy(m)
    ret = np.zeros(shape=mat.shape)
    permutations = list(range(len(mat)))

    for row in range(len(mat)):
        diag = np.array(mat.diagonal())
        maxVal = diag[row]
        pivot = row
        if row != len(mat) - 1:
            for idx, val in enumerate(diag[row + 1:]):
                if val > maxVal:
                    pivot = idx + row + 1
                    val = maxVal

        if pivot != row:
            # swap permutations
            permutations[pivot] = permutations[row]
            permutations[row] = pivot

            # swap matrix m
            mat[:, [row, pivot]] = mat[:, [pivot, row]]
            mat[[row, pivot], :] = mat[[pivot, row], :]

            # swap columns of result
            ret[:, [row, pivot]] = ret[:, [pivot, row]]

        for col in range(row, len(mat)):
            sum_temp = np.float32()
            for ind in range(row):
                sum_temp += ret[ind][row] * ret[ind][col]
            temp = mat[row][col] - sum_temp

            if row == col:
                if temp >= 0:
                    ret[row][col] = math.sqrt(temp)
                else:
                    ret[row][col] = temp
            else:
                ret[row][col] = temp / ret[row][row]

    return (ret, permutations)


def revertPivotedCholeskyImplementation(L, P):
    matrix = np.dot(np.transpose(L), L)
    return applyPermutation(matrix, P)


print('My implementation of pivoted Cholesky decomposition = ')
wikiPivotedDecomposed, wikiPermutations = myPivotedCholeskyImplementation(wiki)
print(wikiPermutations)
print(wikiPivotedDecomposed)
print('\n')

print('Test matrix = ')
print(wiki)
print('L * LT permutated = ')
print(revertPivotedCholeskyImplementation(
    wikiPivotedDecomposed, wikiPermutations))
# print(np.dot(wikiPivotedDecomposed, np.transpose(wikiPivotedDecomposed)))
