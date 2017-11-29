import numpy as np
import cvxpy as cvx

def euclidean_proj_simplex(x, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    v = np.squeeze(x)
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]

    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    w = np.expand_dims(w, axis = 1)
    return w

def update_x1(M_inv, A, y, p, z, u1):
    tmp_x = 2 * np.dot(A.T, y) + p * z - p * u1
    x = np.dot(M_inv, tmp_x)
    return x

def update_x2G(z, u2, p, lama = 1.0):
    t = lama / p
    x_tmp = z - u2
    x_l2 = np.linalg.norm(x_tmp, 2)

    print "x_l2 : ", x_l2
    if x_l2 >= t:
        x2 = (1 - t / x_l2) * x_tmp
    else:
        x2 = np.zeros(z.shape)
    
    return x2

def update_x2full(z, u2, p, lama = 1.0):
    x2g1 = update_x2G(z[0:33, :], u2[0:33, :], p)
    x2g2 = update_x2G(z[33:66, :], u2[33:66, :], p)
    x2g3 = update_x2G(z[66:100, :], u2[66:100, :], p)
    x2 = np.vstack((x2g1, x2g2, x2g3))
    #print x2.shape
    return x2

def update_z(x1, x2, u1, u2):
    z_tmp = (x1 + x2 + u1 + u2) / 2.0 
    z = euclidean_proj_simplex(z_tmp)
    #print z.T
    return z

def update_u(u1, u2, x1, x2, z):
    u1 = u1 + (x1 - z)
    u2 = u2 + (x2 - z)
    return u1, u2

def update_x1cvx(y, A, z, u1, p):
    x1 = cvx.Variable(n)

    objective = cvx.Minimize(cvx.sum_squares(A * x1 - y) + p / 2.0 * cvx.sum_squares(x1 - z + u1))

    prob = cvx.Problem(objective)

    result = prob.solve()

    return x1.value

def update_x2gcvx(zg, u2g, p, lama = 1.0):
    num, _ = zg.shape

    x2g = cvx.Variable(num)

    objective = cvx.Minimize(lama * cvx.norm(x2g) + p / 2.0 * cvx.sum_squares(x2g - zg + u2g))

    prob = cvx.Problem(objective)

    result = prob.solve()

    return x2g.value

def update_x2cvx(z, u2, p, lama = 1.0):
    x2g1 = update_x2gcvx(z[0:33, :], u2[0:33, :], p)
    x2g2 = update_x2gcvx(z[33:66, :], u2[33:66, :], p)
    x2g3 = update_x2gcvx(z[66:100, :], u2[66:100, :], p)
    x2 = np.vstack((x2g1, x2g2, x2g3))
    #print x2.shape
    return x2

def update_zcvx(x1, x2, u1, u2):
    
    z = cvx.Variable(n)

    objective = cvx.Minimize(cvx.sum_squares(0.2 * (x1 + x2 + u1 + u2) - z))

    constraint = [z >= 0, cvx.sum_entries(z) == 1]

    prob = cvx.Problem(objective, constraint)

    result = prob.solve()

    return z.value


n = 100
m = 30
A = np.random.randn(m, n)
x = np.vstack((np.random.rand(3, 1), np.zeros((30, 1)),
               np.random.rand(3, 1), np.zeros((30, 1)),
               np.random.rand(3, 1), np.zeros((31, 1))))

x = np.dot(np.eye(n)/np.sum(x), x)

v = 0.1 * np.random.rand(m, 1)

y = np.dot(A, x)

#print x

p = 2.0
M = 2 * np.dot(A.T, A) + p * np.eye(n, n)
M_inv = np.eye(n) / M

print M_inv * M

x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
#X2 = x1

z = np.random.randn(n, 1)

#u1 = np.random.randn(n, 1)
#u2 = np.random.randn(n, 1)
u1 = x1 - z
u2 = x2 - z


for i in range(100):
    '''
    x1 = update_x1(M_inv, A, y, p, z, u1)
    x2 = update_x2full(z, u2, p)
    z = update_z(x1, x2, u1, u2)
    u1, u2 = update_u(u1, u2, x1, x2, z)
    '''

    x1 = update_x1cvx(y, A, z, u1, p)
    x2 = update_x2cvx(z, u2, p)
    z = update_zcvx(x1, x2, u1, u2)
    u1, u2 = update_u(u1, u2, x1, x2, z)
    
    print np.linalg.norm(z - x, 2)
    print "u1: ", np.linalg.norm(u1, 2)



