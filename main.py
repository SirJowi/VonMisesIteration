""" vonMisesIteration, 12/18/2023 """

import numpy as np

def vonMisesIteration(A, B):
    # check if A and B are symmetric and if they have the same size
    dimA = np.shape(A)
    dimB = np.shape(B)
    # Error if A is not symmetric
    if dimA[0] != dimA[1]:
        print("A is not symmetric.")
        return
    # Error if B is not symmetric
    if dimB[0] != dimB[1]:
        print("B is not symmetric.")
        return
    # Error if A and B do not have the same size
    if dimA != dimB:
        print("A and B do not have the same shape")
        return

    # STEP ZERO -------------------------
    # set dimension of the starting vector
    dim = dimA
    # starting vector filled with ones, shape equals dimension of the problem
    v0 = np.ones(shape=dim[0])
    # set solution vector to v0
    v = v0
    # numerator of the Rayleigh quotient
    Z = np.matmul(np.matmul(v.transpose(), A), v)
    # denominator of the Rayleigh quotient
    h = np.matmul(B, v)
    N = np.matmul(v.transpose(), h)
    # Rayleigh quotient
    R = Z / N
    # Normalization
    h = h / np.sqrt(N)

    # STEP K
    epsilon = 10
    n = 0
    while(abs(epsilon) > 1E-10):
        # number of iterations counter
        n = n + 1
        # new vector
        v = np.matmul(np.linalg.inv(A), h)
        # new numerator
        Z = np.matmul(np.matmul(v.transpose(), A), v)
        # new denominator
        h = np.matmul(B, v)
        N = np.matmul(v.transpose(), h)
        # Remember the old R to calculate an error epsilon
        R_old = R
        # new Rayleigh quotient
        R = Z / N
        # normalize
        h = h / np.sqrt(N)
        # Error
        epsilon = (R-R_old)/(R+R_old)

    print("#############################")
    print("-----------------------------")
    print("Problem: (A - λ * B) * x = 0")
    print("A:")
    print(A)
    print("B:")
    print(B)
    print("-----------------------------")
    print("Solution:")
    print("Anzahl Iterationen\t:", n)
    print("Rayleigh-Quotient\t:", R)
    print("Vektor\t\t\t\t:", h)
    print("#############################")


# Exercise Problem
A1 = np.array([[1, 2, 1],
              [2, 7, 8],
              [1, 8, 17]])

B1 = np.array([[1, 2, 1],
              [2, 5, 4],
              [1, 4, 6]])

# vonMisesIteration(A1, B1)

# Project Task
A2 = np.array([[1, -2],
              [-2, 5]])

B2 = np.array([[1, 1],
              [1, 2]])

vonMisesIteration(A2, B2)
