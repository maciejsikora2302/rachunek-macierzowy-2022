from math import ceil, log
from common import *


def mat_mul_normal(A, B):
    global test
    out = [[0 for _ in range(len(A))] for _ in range(len(A))]
    for row in range(len(A)):
        for col in range(len(B)):
            sum = 0
            for i in range(len(A)):
                sum += A[row][i] * B[i][col]
                counter.add(2)
            out[row][col] = sum
    return out


def add(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] + B[i][j]
            counter.add(1)
    return C


def subtract(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] - B[i][j]
            counter.add(1)
    return C


def mat_mul_strassenR(A, B, treshold = 4):
    """
        Implementation of the strassen algorithm.
    """
    n = len(A)

    if n <= treshold:
        return mat_mul_normal(A, B)
    else:
        # initializing the new sub-matrices
        newSize = int(n / 2)
        a11 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        a12 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        a21 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        a22 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]

        b11 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        b12 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        b21 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        b22 = [[0 for j in range(0, newSize)] for i in range(0, newSize)]

        aResult = [[0 for j in range(0, newSize)] for i in range(0, newSize)]
        bResult = [[0 for j in range(0, newSize)] for i in range(0, newSize)]

        # dividing the matrices in 4 sub-matrices:
        for i in range(0, newSize):
            for j in range(0, newSize):
                a11[i][j] = A[i][j]  # top left
                a12[i][j] = A[i][j + newSize]  # top right
                a21[i][j] = A[i + newSize][j]  # bottom left
                a22[i][j] = A[i + newSize][j + newSize]  # bottom right

                b11[i][j] = B[i][j]  # top left
                b12[i][j] = B[i][j + newSize]  # top right
                b21[i][j] = B[i + newSize][j]  # bottom left
                b22[i][j] = B[i + newSize][j + newSize]  # bottom right

        # Calculating p1 to p7:
        aResult = add(a11, a22)
        bResult = add(b11, b22)
        p1 = mat_mul_strassenR(aResult, bResult)  # p1 = (a11+a22) * (b11+b22)

        aResult = add(a21, a22)  # a21 + a22
        p2 = mat_mul_strassenR(aResult, b11)  # p2 = (a21+a22) * (b11)

        bResult = subtract(b12, b22)  # b12 - b22
        p3 = mat_mul_strassenR(a11, bResult)  # p3 = (a11) * (b12 - b22)

        bResult = subtract(b21, b11)  # b21 - b11
        p4 = mat_mul_strassenR(a22, bResult)  # p4 = (a22) * (b21 - b11)

        aResult = add(a11, a12)  # a11 + a12
        p5 = mat_mul_strassenR(aResult, b22)  # p5 = (a11+a12) * (b22)

        aResult = subtract(a21, a11)  # a21 - a11
        bResult = add(b11, b12)  # b11 + b12
        p6 = mat_mul_strassenR(aResult, bResult)  # p6 = (a21-a11) * (b11+b12)

        aResult = subtract(a12, a22)  # a12 - a22
        bResult = add(b21, b22)  # b21 + b22
        p7 = mat_mul_strassenR(aResult, bResult)  # p7 = (a12-a22) * (b21+b22)

        # calculating c21, c21, c11 e c22:
        c12 = add(p3, p5)  # c12 = p3 + p5
        c21 = add(p2, p4)  # c21 = p2 + p4

        aResult = add(p1, p4)  # p1 + p4
        bResult = add(aResult, p7)  # p1 + p4 + p7
        c11 = subtract(bResult, p5)  # c11 = p1 + p4 - p5 + p7

        aResult = add(p1, p3)  # p1 + p3
        bResult = add(aResult, p6)  # p1 + p3 + p6
        c22 = subtract(bResult, p2)  # c22 = p1 + p3 - p2 + p6

        # Grouping the results obtained in a single matrix:
        C = [[0 for j in range(0, n)] for i in range(0, n)]
        for i in range(0, newSize):
            for j in range(0, newSize):
                C[i][j] = c11[i][j]
                C[i][j + newSize] = c12[i][j]
                C[i + newSize][j] = c21[i][j]
                C[i + newSize][j + newSize] = c22[i][j]
        return C


def mat_mul_strassen(A, B):
    # print(A, B)
    # print(f"len(A) {len(A)} , len(A[0]) {len(A[0])},  len(B) {len(B)}, len(B[0]) {len(B[0])}")
    assert len(A) == len(A[0]) == len(B) == len(B[0])

    # Make the matrices bigger so that you can apply the strassen
    # algorithm recursively without having to deal with odd
    # matrix sizes
    nextPowerOfTwo = lambda n: 2 ** int(ceil(log(n, 2)))
    n = len(A)
    m = nextPowerOfTwo(n)
    APrep = [[0 for i in range(m)] for j in range(m)]
    BPrep = [[0 for i in range(m)] for j in range(m)]
    for i in range(n):
        for j in range(n):
            APrep[i][j] = A[i][j]
            BPrep[i][j] = B[i][j]
    CPrep = mat_mul_strassenR(APrep, BPrep)
    C = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = CPrep[i][j]
    return C