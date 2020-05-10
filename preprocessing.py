import numpy as np

# Define universal constants for the problem
modulus = 2e8
area = 0.001
length = 6
t_l = -4e6
b = 2e3

def global_stiffness(n_e, order=1):
    l_e = t_l / n_e
    K_e = local_stiffness(order)


def local_stiffness(l_e, order=1):
    if order == 1:
        return area * modulus / l_e * np.array([[1, -1], [-1, 1]])
