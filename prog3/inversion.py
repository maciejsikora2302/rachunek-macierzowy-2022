import numpy as np
from multiplication_functions import *

def inv(A, mult_function):
    if len(A) == 1:
        if A[0][0] == 0: raise Exception('Matrix contains 0, can\'t invert')
        A[0][0] = 1/A[0][0]
        return A

    n = int(len(A)/2)
    A = np.array(A)
    a11 = A[:n, :n]
    a12 = A[n:, :n]
    a21 = A[:n, n:]
    a22 = A[n:, n:]

    a11_inv = inv(a11, mult_function)

    s22 = subtract(a22, mult_function(mult_function(a21, a11_inv), a12)) #S22 = A22 − A21 ∗ A11^(−1) ∗ A12

    s22_inv = inv(s22, mult_function)

    top_right = mult_function(mult_function(a11_inv, a12), s22_inv)


    A[:n, :n] = add(a11_inv, mult_function(mult_function(top_right, a21), a11_inv))
    A[n:, :n] = -1*np.array(mult_function(mult_function(s22_inv, a21), a11_inv))
    A[:n, n:] = -1*np.array(top_right)
    A[n:, n:] = s22_inv

    return A