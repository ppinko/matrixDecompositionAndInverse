import numpy as np
import copy
import math
from numpy.core.fromnumeric import shape

wiki = np.array(
    [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])

mat4x4 = np.array([[2.0, 4.0, -2.0, 2.0],
                   [4.0, 9.0, -1.0, 6.0],
                   [-2.0, -1.0, 14.0, 13.0],
                   [2.0, 6.0, 13.0, 35.0]])


def applyPermutation(matrix, permutations):
    matrix_size = len(permutations)
    ret = np.zeros(shape=(matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            ret[i][j] = matrix[permutations[i]][permutations[j]]
    return ret


def myCholeskyImplementation(m):
    L = np.zeros(shape=m.shape)

    for row in range(len(m)):
        for col in range(row, len(m)):
            sum_temp = np.float32()
            for ind in range(row):
                sum_temp += L[ind][row] * L[ind][col]
            temp = m[row][col] - sum_temp

            if row == col:
                if temp >= 0.0:
                    L[row][col] = math.sqrt(temp)
                else:
                    return 0
            else:
                if L[row][row] != 0:
                    L[row][col] = temp / L[row][row]
                else:
                    return 0
    return L


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
    L = np.zeros(shape=mat.shape)
    P = list(range(len(mat)))

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
            # swap P
            P[pivot] = P[row]
            P[row] = pivot

            # swap matrix m
            mat[:, [row, pivot]] = mat[:, [pivot, row]]
            mat[[row, pivot], :] = mat[[pivot, row], :]

            # swap columns of result
            L[:, [row, pivot]] = L[:, [pivot, row]]

        for col in range(row, len(mat)):
            sum_temp = np.float32()
            for ind in range(row):
                sum_temp += L[ind][row] * L[ind][col]
            temp = mat[row][col] - sum_temp

            if row == col:
                if temp >= 0:
                    L[row][col] = math.sqrt(temp)
                else:
                    return (0, 0)
            else:
                if L[row][row] != 0:
                    L[row][col] = temp / L[row][row]
                else:
                    return (0, 0)
    return (L, P)


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
print('\n')

print('My implementation of pivoted Cholesky decomposition = ')
mat4x4PivotedDecomposed, mat4x4Permutations = myPivotedCholeskyImplementation(
    mat4x4)
print(mat4x4Permutations)
print(mat4x4PivotedDecomposed)
print('\n')

print('Test matrix = ')
print(mat4x4)
print('L * LT permutated = ')
print(revertPivotedCholeskyImplementation(
    mat4x4PivotedDecomposed, mat4x4Permutations))
print('\n')


def myLDLTdecomposition(mat):
    D = np.eye(len(mat), dtype=np.float32)
    L = np.zeros(shape=mat.shape)

    for row in range(len(mat)):
        for col in range(row, len(mat)):
            sum_temp = np.float32()
            for ind in range(row):
                sum_temp += L[ind][row] * D[ind][ind] * L[ind][col]
            temp = mat[row][col] - sum_temp

            if row == col:
                D[row][row] = temp
                L[row][row] = 1.0
            else:
                if D[row][row] != 0:
                    L[row][col] = temp / D[row][row]
                else:
                    return (0, 0)
    return (L, D)


print('My implementation of pivoted LDLT decomposition: ')
L, D = myLDLTdecomposition(mat4x4)
print('\nL= ')
print(L)
print('\nD= ')
print(D)


def myPivotedLDLTdecomposition(m):
    mat = copy.deepcopy(m)
    P = list(range(len(mat)))
    D = np.eye(len(mat), dtype=np.float32)
    L = np.zeros(shape=mat.shape)

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
            # swap P
            P[pivot] = P[row]
            P[row] = pivot

            # swap matrix mat
            mat[:, [row, pivot]] = mat[:, [pivot, row]]
            mat[[row, pivot], :] = mat[[pivot, row], :]

            # swap columns of result
            L[:, [row, pivot]] = L[:, [pivot, row]]

        for col in range(row, len(mat)):
            sum_temp = np.float32()
            for ind in range(row):
                sum_temp += L[ind][row] * D[ind][ind] * L[ind][col]
            temp = mat[row][col] - sum_temp

            if row == col:
                D[row][row] = temp
                L[row][row] = 1.0
            else:
                if D[row][row] != 0:
                    L[row][col] = temp / D[row][row]
                else:
                    return (0, 0)
    return (L, D, P)


def revertPivotedLDLTDecomposition(L, D, P):
    matrix = np.dot(np.dot(np.transpose(L), D), L)
    return applyPermutation(matrix, P)


print('My implementation of pivoted LDLT decomposition = ')
L, D, P = myPivotedLDLTdecomposition(mat4x4)
print(P)
print(L)
print(D)
print('\n')

print('Test matrix = ')
print(mat4x4)
print('L * LT permutated = ')
print(revertPivotedLDLTDecomposition(
    L, D, P))
print('\n')
