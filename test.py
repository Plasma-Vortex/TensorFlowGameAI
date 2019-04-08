import numpy as np
from Game import *

def foo(s):
    if __debug__:
        print('hi')
    return s

print(1/2)

t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a = np.array([t, u, 0])
b = AddSymmetriesTTT(a)
for i in b:
    print(i)