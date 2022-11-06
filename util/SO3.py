import numpy as np
from scipy.spatial.transform import Rotation as sciR
import einops
import scipy

DATA_PATH = 'data'
path = DATA_PATH + '/J_dense.npy'
Jd_np = np.load(str(path), allow_pickle = True)
Jd = list(Jd_np)

def wigner_d_matrix(degree, alpha, beta, gamma):
    '''
    here, alpha, beta, gamma: alpha, beta, gamma = sciR.from_quat(quat).as_euler('ZYZ')
    ZYZ euler with relative Rz@Ry@Rz
    Note that when degree is 1 wigner_d matrix is not equal to rotation matrix
    The equality comes from sciR.from_quat(q).as_matrix() = wigner_d_matrix(1, *sciR.from_quat(q).as_euler('YXY'))
    '''
    """Create wigner D matrices for batch of ZYZ Euler anglers for degree l."""
    if degree==0:
        return np.array([[1.0]])
    if degree==1:
        return sciR.from_euler('YXY',[alpha,beta,gamma]).as_matrix()
    J = Jd[degree]
    order = 2*degree + 1
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta, degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    return res.reshape(order, order)

def wigner_d_from_quat(degree, quat):
    if isinstance(degree, list) or isinstance(degree, tuple):
        return scipy.linalg.block_diag(*[wigner_d_matrix(oo, *list(sciR.from_quat(quat).as_euler('ZYZ'))) for oo in degree])
    else:
        return wigner_d_matrix(degree, *list(sciR.from_quat(quat).as_euler('ZYZ')))

def z_rot_mat(angle, l):
    order = 2*l+1
    m = np.zeros((order, order))
    inds = np.arange(0, order)
    reversed_inds = np.arange(2*l, -1, -1)
    frequencies = np.arange(l, -l -1, -1)[None]

    m[inds, reversed_inds] = np.sin(frequencies * np.array(angle)[None])
    m[inds, inds] = np.cos(frequencies * np.array(angle)[None])
    return m

def x_to_alpha_beta(x):
    '''
    Convert point (x, y, z) on the sphere into (alpha, beta)
    '''
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    beta = np.arccos(x[2])
    alpha = np.arctan2(x[1], x[0])
    return alpha, beta

def sh_via_wigner_d(l, pnt):
    a, b = x_to_alpha_beta(pnt)
    return wigner_d_matrix(l, np.array(a), np.array(b), np.array(0))[...,:,l]


def get_matrix_kernel(A, eps=1e-10):
    '''
    Compute an orthonormal basis of the kernel (x_1, x_2, ...)
    A x_i = 0
    scalar_product(x_i, x_j) = delta_ij

    :param A: matrix
    :return: matrix where each row is a basis vector of the kernel of A
    '''
    _u, s, vh = np.linalg.svd(A)

    # A = u @ torch.diag(s) @ v.t()
    kernel = vh[s < eps]
    return kernel


def get_matrices_kernel(As, eps=1e-10):
    '''
    Computes the commun kernel of all the As matrices
    '''
    return get_matrix_kernel(np.concatenate(As, axis=0), eps)

# @cached_dirpklgz("cache/trans_Q")
def _basis_transformation_Q_J(J, order_in, order_out, test=False):
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    np.kron(wigner_d_matrix(order_out, a, b, c),wigner_d_matrix(order_in, a, b, c)) @ Q_J =
        Q_J @ wigner_d_matrix(J, a, b, c)
    """
    def _R_tensor(a, b, c): return np.kron(wigner_d_matrix(order_out, a, b, c), wigner_d_matrix(order_in, a, b, c))

    def _sylvester_submatrix(J, a, b, c):
        ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
        R_tensor = _R_tensor(a, b, c)  # [m_out * m_in, m_out * m_in]
        R_irrep_J = wigner_d_matrix(J, a, b, c)  # [m, m]
        return np.kron(R_tensor, np.eye(R_irrep_J.shape[0])) - \
            np.kron(np.eye(R_tensor.shape[0]), R_irrep_J.T)  # [(m_out * m_in) * m, (m_out * m_in) * m]

    random_angles = [
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.2908958, 3.90040975]
    ]
    null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
    Q_J = null_space[0]  # [(m_out * m_in) * m]
    Q_J = Q_J.reshape((2 * order_out + 1) * (2 * order_in + 1), 2 * J + 1)  # [m_out * m_in, m]
    Q_J = np.where(np.abs(Q_J)<1e-10,0.0,Q_J)
    if test:
        for _ in range(3):
            a,b,c = np.random.normal(size=(3,))
            DQJ = np.einsum('...ij,...jk', _R_tensor(a, b, c), Q_J)
            QJD = np.einsum('...ij,...jk', Q_J, wigner_d_matrix(J, a, b, c))
            print(np.sum(np.abs(DQJ - QJD)), np.sum(np.abs(np.einsum('...ij,...ik',Q_J,Q_J) - np.eye(2*J+1))) )
    return Q_J  # [m_out * m_in, m]

def _basis_transformation_Q_J_list(J, order_in_list, order_out_list, test=False):
    """
    :param J: order of the spherical harmonics
    :param order_in: order of the input representation
    :param order_out: order of the output representation
    :return: one part of the Q^-1 matrix of the article
    np.kron(wigner_d_matrix(order_out, a, b, c),wigner_d_matrix(order_in, a, b, c)) @ Q_J =
        Q_J @ wigner_d_matrix(J, a, b, c)
    """
    import scipy
    def _R_tensor(a, b, c):
        ao = scipy.linalg.block_diag(*[wigner_d_matrix(oo, a, b, c) for oo in order_out_list])
        ai = scipy.linalg.block_diag(*[wigner_d_matrix(oi, a, b, c) for oi in order_in_list])
        return np.kron(ao, ai)

    def _sylvester_submatrix(J, a, b, c):
        ''' generate Kronecker product matrix for solving the Sylvester equation in subspace J '''
        R_tensor = _R_tensor(a, b, c)  # [m_out * m_in, m_out * m_in]
        R_irrep_J = wigner_d_matrix(J, a, b, c)  # [m, m]
        return np.kron(R_tensor, np.eye(R_irrep_J.shape[0])) - \
            np.kron(np.eye(R_tensor.shape[0]), R_irrep_J.T)  # [(m_out * m_in) * m, (m_out * m_in) * m]

    random_angles = [
        [4.41301023, 5.56684102, 4.59384642],
        [4.93325116, 6.12697327, 4.14574096],
        [0.53878964, 4.09050444, 5.36539036],
        [2.16017393, 3.48835314, 5.55174441],
        [2.52385107, 0.2908958, 3.90040975],
    ]
    # random_angles = random_angles + list(np.random.uniform(0, 2*np.pi, size=(5,3)))
    null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a, b, c in random_angles])
    # null_space = np.sum(null_space, axis=0, keepdims=True)
    Q_J = null_space.reshape(null_space.shape[0], -1, 2 * J + 1)  # [m_out * m_in, m]
    Q_J = np.where(np.abs(Q_J)<1e-10,0.0,Q_J)
    # Q_J = Q_J * np.sqrt(2*J+1) / np.sqrt(np.sum(np.diag(Q_J.T@Q_J))) ## normalize
    if test:
        ref_idx = np.array([1,11,12,13,21,22,23,24,25])
        ref_idx1 = np.kron(ref_idx,np.ones_like(ref_idx))
        ref_idx2 = np.kron(np.ones_like(ref_idx),ref_idx)
        Q_J_ref = np.sum(np.abs(Q_J), axis=-1)
        for j in range(Q_J.shape[0]):
            print(ref_idx1[np.where(Q_J_ref[j]>0)])
            print(ref_idx2[np.where(Q_J_ref[j]>0)])
        for _ in range(3):
            a,b,c = np.random.normal(size=(3,))
            DQJ = np.einsum('...ij,...jk', _R_tensor(a, b, c), Q_J)
            QJD = np.einsum('...ij,...jk', Q_J, wigner_d_matrix(J, a, b, c))
            print(np.sum(np.abs(DQJ - QJD)), np.sum(np.abs(np.einsum('...ij,...ik',Q_J,Q_J) - np.eye(2*J+1))) )
    return Q_J  # [m_out * m_in, m]


def cg_product(J, lin, lout, f1, f2):
    fv = np.kron(f1, f2)
    # fv = einops.rearrange(np.kron(f1, f2), '... i j -> ... (i j)')
    return np.einsum('...ij,...j->...i', _basis_transformation_Q_J(J, lin, lout).T, fv)

def cg_product_list(J, lin_list, lout_list, f1, f2):
    '''
    f1: nc, nf
    f2: nc, nf
    '''
    return np.einsum('...ij,...j->...i', _basis_transformation_Q_J_list(J, lin_list, lout_list).T, np.kron(f1, f2))


def cg_product_weights(weight, nlout, f1, f2, Q=None):
    '''
    Q: list[nf1*nf2 x nfo0, nf1*nf2 x nfo1, ...]
    f1: nc1 x nf1(1+3+5)
    f2: nc2 x nf2(1+3+5)
    weight: nc1*nc2*nlout*nl1*nl2
    return: nc1*nc2 x nfo
    '''
    def nf_to_nl(nf):
        if nf==9:
            return 3
        elif nf==4:
            return 2
        elif nf==1:
            return 1
        else:
            raise ValueError
    nf1 = f1.shape[-1]
    nf2 = f2.shape[-1]
    nc1 = f1.shape[-2]
    nc2 = f2.shape[-2]
    nl1 = nf_to_nl(nf1)
    nl2 = nf_to_nl(nf2)
    if Q is None:
        Q_list = [_basis_transformation_Q_J_list(i, np.arange(nl1), np.arange(nl2)) for i in range(nlout)]
    nc_w_list = [q.shape[0] for q in Q_list]
    nc_w_idx_list = np.cumsum(nc_w_list)
    nc_w_idx_list = np.concatenate([[0],nc_w_idx_list], axis=-1)
    nlpair = np.sum(nc_w_list)
    assert weight.shape[-1] == nc1*nc2*nlpair
    
    if nlout == 3:
        weight = einops.rearrange(weight, '... (nc1 nc2 nlpair) -> ... nc1 nc2 nlpair', nc1=nc1, nc2=nc2, nlpair=nlpair)
        f1_ext = f1[...,:,None,None,:]
        f2_ext = f2[...,None,:,None,:]

        res_list = []
        for i in range(3):
            w = weight[...,nc_w_idx_list[i]:nc_w_idx_list[i+1],None,None] # nc1 nc2 npi 1 1
            Q_ext = Q_list[i] # npi (nf1 nf2) nfoi
            Q_ext = Q_ext[...,None,None,:,:,:] # 1 1 npi (nf1 nf2) nfoi
            Q_extT = einops.rearrange(Q_ext, '... i j -> ... j i') # 1 1 npi nfoi (nf1 nf2)
            fout = np.einsum('...ij,...j', w * Q_extT, np.kron(f1_ext, f2_ext)) # nc1 nc2 npi nfoi
            fout = einops.rearrange(fout, '... i j k t -> ... (i j k) t')
            res_list.append(fout)

        return res_list
    else:
        raise ValueError




## test code ###
def _test_sh_wigner_D(order):
    pnt = np.random.normal(size=(3,))
    quat = np.random.normal(size=(4,))
    quat = quat / np.linalg.norm(quat)

    rpnt = sciR.from_quat(quat).apply(pnt)
    Yrx = sh_via_wigner_d(order, rpnt)
    # # Y = get_spherical_harmonics_from_pnts(order, pnt)
    Y = sh_via_wigner_d(order, pnt)
    DrY = wigner_d_from_quat(order, quat)@ Y

    print(np.sum(np.abs(Yrx - DrY)))


def _test_cg_product():
    quat = np.random.normal(size=(4,))
    quat = quat / np.linalg.norm(quat)

    f1 = np.random.normal(size=(3,))
    f2 = np.random.normal(size=(3,))

    D0 = wigner_d_from_quat(1, quat)
    D1 = wigner_d_from_quat(1, quat)
    D2 = wigner_d_from_quat(2, quat)
    
    rfvn = cg_product(2,1,1, D1@f1, D1@f2)
    fvn = D2@cg_product(2,1,1, f1, f2)
    print(np.sum(np.abs(rfvn-fvn)))


def _test_cg_product_list():
    import scipy
    quat = np.random.normal(size=(4,))
    quat = quat / np.linalg.norm(quat)

    order1_list = [0,1,2]
    order2_list = [0,1,2]
    
    f1 = np.random.normal(size=np.sum([2*f+1 for f in order1_list], keepdims=True))
    f2 = np.random.normal(size=np.sum([2*f+1 for f in order2_list], keepdims=True))

    Din = scipy.linalg.block_diag(*[wigner_d_from_quat(oo, quat) for oo in order1_list])
    Dout = scipy.linalg.block_diag(*[wigner_d_from_quat(oo, quat) for oo in order2_list])

    D0 = wigner_d_from_quat(0, quat)
    D1 = wigner_d_from_quat(1, quat)
    D2 = wigner_d_from_quat(2, quat)
    
    rfvn = cg_product_list(2, order1_list, order2_list, Din@f1, Dout@f2)
    fvn = D2@cg_product_list(2, order1_list, order2_list, f1, f2)
    print(np.sum(np.abs(rfvn-fvn)))


def _test_wiger_D_Q_compatibility():
    quat = np.random.normal(size=(4,))
    quat = quat / np.linalg.norm(quat)

    res_list = [_basis_transformation_Q_J(i, 1, 1) for i in range(3)]
    Q = np.concatenate(res_list, axis=1)

    A = np.kron(wigner_d_from_quat(1, quat), wigner_d_from_quat(1, quat))
    B = np.zeros_like(A)
    B[0:1,0:1] = wigner_d_from_quat(0, quat)
    B[1:4,1:4] = wigner_d_from_quat(1, quat)
    B[4:,4:] = wigner_d_from_quat(2, quat)

    print(np.sum(np.abs(A@Q - Q@B)))

def _test_sh_with_lib(order):
    import spherical_harmonic
    pnt = np.random.normal(size=(3,))
    pnt = pnt/np.linalg.norm(pnt)
    Y1 = spherical_harmonic.get_spherical_harmonics_from_pnts(order, pnt)
    Y2 = sh_via_wigner_d(order, pnt)
    ratio = Y1 / Y2
    print(np.sum(np.abs(ratio - ratio[...,0:1])))

def _test_Q_J_list():
    order_in_list = [0,1,2]
    order_out_list = [0,1,2]
    Q = _basis_transformation_Q_J_list(2, order_in_list, order_out_list, test=True)




if __name__ == '__main__':
    # res = _basis_transformation_Q_J(2,1,1,test=True)
    # _basis_transformation_Q_J_list(1, [0,1,2], [0,1,2], test=True)
    # for i in range(3):
    #     # _basis_transformation_Q_J(i,2,2,test=True)
    #     _basis_transformation_Q_J_list(i, [0,1,2], [0,1,2], test=True)

    nl1 = 3
    nl2 = 3
    nlout = 3
    nc1 = 2
    nc2 = 3
    f1 = np.random.normal(size=(nc1,9,))
    f2 = np.random.normal(size=(nc2,9,))
    npair = 3+6+6
    weight = np.random.normal(size=(nc1*nc2*npair,))

    quat = np.random.normal(size=(4,))
    quat = quat / np.linalg.norm(quat)

    D012 = wigner_d_from_quat([0,1,2], quat)
    D0 = wigner_d_from_quat(0, quat)
    D1 = wigner_d_from_quat(1, quat)
    D2 = wigner_d_from_quat(2, quat)
    def matmul(a,b):
        return np.einsum('...ij,...j',a,b)

    foutD = cg_product_weights(weight, nlout, matmul(D012,f1), matmul(D012,f2))
    Dfout = [matmul(d,f) for f, d in zip(cg_product_weights(weight, nlout, f1, f2), [D0,D1,D2])]

    for i in range(6):
        _test_cg_product_list()
        _test_sh_wigner_D(i)
        _test_cg_product()
        _test_wiger_D_Q_compatibility()
        _test_sh_with_lib(i)
        _test_Q_J_list()
    
