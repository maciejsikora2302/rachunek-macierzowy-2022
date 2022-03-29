import numpy as np
from pprint import pprint as pp
from multiplication_functions import *
from time import time_ns
from copy import copy as cp
from common import *
from matplotlib import pyplot as plt


def inv(A, mult_function):
    if len(A) == 1:
        if A[0][0] == 0: return A
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



# two_power = 2
# percision = 10**(-10)


# A = np.random.rand(2**two_power,2**two_power)
# B = np.random.rand(2**two_power,2**two_power)

# pp(A@B - mat_mul_normal(A,B) < percision)
# pp(A@B - mat_mul_normal(A,B) < percision)

# pp(np.allclose(A@B, mat_mul_normal(A, B), atol = percision, rtol = 0))
# pp(np.allclose(A@B, mat_mul_normal(A, B), atol = percision, rtol = 0))

# import sys

# sys.exit(0)

class StatGatherer():
    def __init__(self, name):
        self.name = name
        self.size = []
        self.time = []
        self.opcount = []

stat_strassen = StatGatherer('STRASSEN')
stat_normal = StatGatherer('NORMAL')


functions = [mat_mul_normal, mat_mul_strassen]

for two_power in range(2,5):
    for f in functions:
        A = np.random.rand(2**two_power,2**two_power)

        start = time_ns()
        C = inv(A, f)
        mesured_time = time_ns() - start
        
        print(f"Size: {2**two_power}, name: {f.__name__}, time: {mesured_time}, op_count: {counter.get()}")
        if f.__name__=='mat_mul_normal':
            stat_normal.size.append(2**two_power)
            stat_normal.time.append(mesured_time)
            stat_normal.opcount.append(counter.get())
        else:
            stat_strassen.size.append(2**two_power)
            stat_strassen.time.append(mesured_time)
            stat_strassen.opcount.append(counter.get())

with open('./saved_results.txt', 'w') as f:
    f.write("NORMAL\n")
    f.write(f"{','.join(list(map(str, stat_normal.size)))}\n")
    f.write(f"{','.join(list(map(str, stat_normal.time)))}\n")
    f.write(f"{','.join(list(map(str, stat_normal.opcount)))}\n")
    f.write("\nSTRASSEM\n")
    f.write(f"{','.join(list(map(str, stat_strassen.size)))}\n")
    f.write(f"{','.join(list(map(str, stat_strassen.time)))}\n")
    f.write(f"{','.join(list(map(str, stat_strassen.opcount)))}\n")

fig, axs = plt.subplots(2, 2)
fig.suptitle('First row normal, second row strassen')
axs[0, 0].plot(stat_normal.size, stat_normal.time)
axs[0, 0].set_title('Normal - time count')
axs[0, 0].set_xlabel("Size")
axs[0, 0].set_ylabel("Time")
axs[0, 0].grid(axis='y')
axs[0, 1].plot(stat_normal.size, stat_normal.opcount)
axs[0, 1].set_title('Normal - operation count')
axs[0, 1].set_xlabel("Size")
axs[0, 1].set_ylabel("Operations")
axs[0, 1].grid(axis='y')
axs[1, 0].plot(stat_strassen.size, stat_strassen.time, "r-")
axs[1, 0].set_title('Strassen - time count')
axs[1, 0].set_xlabel("Size")
axs[1, 0].set_ylabel("Time")
axs[1, 0].grid(axis='y')
axs[1, 1].plot(stat_strassen.size, stat_strassen.opcount, "r-")
axs[1, 1].set_title('Strassen - operation count')
axs[1, 1].set_xlabel("Size")
axs[1, 1].set_ylabel("Operations")
axs[1, 1].grid(axis='y')

plt.show()