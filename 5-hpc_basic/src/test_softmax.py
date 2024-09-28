import numpy as np

L = 64
Dim = 16
BLK = 4

# Q[L, Dim]
# K[L, Dim]
# V[L, Dim]

MIN_M = -10000


def np_softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x


def normal_softmax(x):
    out = np.array(x)
    for r in range(0, L):
        maxi = 0
        for i in range(0, L):
            maxi = max(maxi, x[r, i])
        e_sum = 0
        for i in range(0, L):
            e_sum += np.exp(x[r, i] - maxi)
        for i in range(0, L):
            out[r, i] = np.exp(x[r, i] - maxi) / e_sum
    return out

a = np.random.uniform(-1, 1, [L, L]).astype("float32")

np_res = np_softmax(a)
# print(np_res)

normal_res = normal_softmax(a)
# print(normal_res)
np.testing.assert_allclose(np_res, normal_res, 1e-6, 1e-6)