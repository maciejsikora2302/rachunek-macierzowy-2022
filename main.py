import numpy as np
from pprint import pprint as pp
from multiplication_functions import *
from time import time_ns
from copy import copy as cp
from common import *
from matplotlib import pyplot as plt

two_power = 3
percision = 10**(-10)

# A = np.random.rand(2**two_power,2**two_power)
# B = np.random.rand(2**two_power,2**two_power)

# pp(A@B - mat_mul_normal(A,B) < percision)
# pp(A@B - mat_mul_normal(A,B) < percision)

# pp(np.allclose(A@B, mat_mul_normal(A, B), atol = percision, rtol = 0))
# pp(np.allclose(A@B, mat_mul_normal(A, B), atol = percision, rtol = 0))

SIZE_NORMAL = []
SIZE_STRASSEN = []

TIME_NORMAL = []
TIME_STRASSEN = []

OPCOUNT_NORMAL = []
OPCOUNT_STRASSEN = []


functions = [mat_mul_normal, mat_mul_strassen]

for two_power in range(2,8):
    for f in functions:
        A = np.random.rand(2**two_power,2**two_power)
        B = np.random.rand(2**two_power,2**two_power)

        start = time_ns()
        C, operation_count = f(A,B)
        mesured_time = time_ns() - start
        correct = np.allclose(A@B, np.array(C), atol = percision, rtol = 0)
        
        print(f"Size: {2**two_power}, name: {f.__name__}, correct: {correct}, time: {mesured_time}, op_count: {operation_count}")
        if f.__name__=='mat_mul_normal':
            SIZE_NORMAL.append(2**two_power)
            TIME_NORMAL.append(mesured_time)
            OPCOUNT_NORMAL.append(operation_count)
        else:
            SIZE_STRASSEN.append(2**two_power)
            TIME_STRASSEN.append(mesured_time)
            OPCOUNT_STRASSEN.append(operation_count)

with open('./saved_results.txt', 'w') as f:
    f.write("NORMAL\n")
    f.write(f"{','.join(list(map(str, SIZE_NORMAL)))}\n")
    f.write(f"{','.join(list(map(str, TIME_NORMAL)))}\n")
    f.write(f"{','.join(list(map(str, OPCOUNT_NORMAL)))}\n")
    f.write("\nSTRASSEM\n")
    f.write(f"{','.join(list(map(str, SIZE_STRASSEN)))}\n")
    f.write(f"{','.join(list(map(str, TIME_STRASSEN)))}\n")
    f.write(f"{','.join(list(map(str, OPCOUNT_STRASSEN)))}\n")

fig, axs = plt.subplots(2, 2)
fig.suptitle('First row normal, second row strassen')
axs[0, 0].plot(SIZE_NORMAL, TIME_NORMAL)
axs[0, 0].set_title('Normal - time count')
axs[0, 0].set_xlabel("Size")
axs[0, 0].set_ylabel("Time")
axs[0, 0].grid(axis='y')
axs[0, 1].plot(SIZE_NORMAL, OPCOUNT_NORMAL)
axs[0, 1].set_title('Normal - operation count')
axs[0, 1].set_xlabel("Size")
axs[0, 1].set_ylabel("Operations")
axs[0, 1].grid(axis='y')
axs[1, 0].plot(SIZE_STRASSEN, TIME_STRASSEN, "r-")
axs[1, 0].set_title('Strassen - time count')
axs[1, 0].set_xlabel("Size")
axs[1, 0].set_ylabel("Time")
axs[1, 0].grid(axis='y')
axs[1, 1].plot(SIZE_STRASSEN, OPCOUNT_STRASSEN, "r-")
axs[1, 1].set_title('Strassen - operation count')
axs[1, 1].set_xlabel("Size")
axs[1, 1].set_ylabel("Operations")
axs[1, 1].grid(axis='y')

plt.show()