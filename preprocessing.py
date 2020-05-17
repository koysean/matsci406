import numpy as np

# NOTE: 0-based numbering is used for counting elements and nodes. i.e. for a 2
# element system, the elements are numbered (0,1) and the nodes are numbered
# (0,1,2), assuming linear shape function.

# Define universal constants for the problem
modulus = 2e8
area = 0.001
length = 6
t_l = -4e6
b = 2e3

## Example parameters
#modulus = 4e5
#area = 0.1
#length = 4
#t_l = 5
#b = 5

def main():
    ''' For testing purposes '''
    print(global_matrices(2, 2))

def global_matrices(n_e, order=1):
    ''' Generates global stiffness matrix and force vector. n_e is the number of
    elements. Order determines the shape function (1 = linear, 2 = quadratic).'''
    # Define local parameters and matrices
    l_e = length / n_e
    K_e = local_stiffness(l_e, order)
    f_e_b = local_body_force(l_e, order)

    # Number of nodes in the mesh
    n_n = order * n_e + 1

    # generate the elemental traction force vectors (zero for all elements
    # except for the one at x=6)
    f_e_t = np.zeros((n_e, n_n))
    f_e_t[-1][-1] = area * t_l

    # Initiate global matrices
    K_g = np.zeros((n_n, n_n))
    f_g = np.zeros((n_n))

    for e in range(n_e):
        # For each element, the gather matrix is generated and then used to
        # calculate the summation term for the global matrices associated with
        # that element.
        L_e = gather_matrix(n_e, e, order)
        print(e, L_e)
        np.matmul(np.transpose(L_e), f_e_b)

        K_g += np.matmul(np.transpose(L_e), np.matmul(K_e, L_e))
        f_g += np.matmul(np.transpose(L_e), f_e_b) + f_e_t[e]

    return K_g, f_g

def gather_matrix(n_e, e, order=1):
    ''' Generates the gather matrix for element e. The matrix is an (order+1) x
    (number of nodes) matrix. '''
    L_e = np.zeros((order+1, order*n_e + 1))
    for i, row in enumerate(L_e):
        row[order*e+i] = 1
    return L_e

def local_stiffness(l_e, order=1):
    ''' Returns the appropriate local stiffness matrix given the element length
    and order '''
    if order == 1:
        return area * modulus / l_e * np.array([[1, -1], [-1, 1]])
    elif order == 2:
        return area * modulus / (3 * l_e) * np.array([
            [7, -8, 1], [-8, 16, -8], [1, -8, 7] ])

def local_body_force(l_e, order=1):
    ''' Returns the appropriate local body force vectors given the element
    length and order '''
    if order == 1:
        return b * l_e / 2 * np.array([1, 1])
    elif order==2:
        return b * l_e / 6 * np.array([1,4,1])

if __name__ == '__main__':
    main()
