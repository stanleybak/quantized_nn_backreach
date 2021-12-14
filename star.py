"""
Linear Star State for use in quantized backreach

Stanley Bak
Dec 2021
"""

import numpy as np

import kamenev
from lpinstance import LpInstance

class Star:
    """linear star set

    a linear star is a state represented as an affine transformation of an h-polytope
    the h-polytope is encoded in an lp using glpk
    """

    X_OWN = 0
    Y_OWN = 1
    VX_OWN = 2
    VY_OWN = 3
    X_INT = 4
    VX_INT = 5
    
    NUM_VARS = 6

    def __init__(self, box, constraints_a_mat=None, constraints_b_vec=None):
        """create star from a list of box constraints
        and inequality constrainta a_mat * x <= b_vec

        """

        names = ("x_o", "y_o", "vx_o", "vy_o", "x_i", "vx_i")
        self.dims = len(names)

        assert len(box) == len(names)

        self.hpoly = LpInstance()
        self.a_mat = np.identity(self.dims)
        self.b_vec = np.zeros(self.dims)

        for name, (lb, ub) in zip(names, box):
            self.hpoly.add_double_bounded_cols([name], lb, ub)
        
        # add constraints
        if constraints_a_mat is not None:
            assert len(constraints_a_mat) == len(constraints_b_vec)

            for row, rhs in zip(constraints_a_mat, constraints_b_vec):
                self.hpoly.add_dense_row(np.array(row, dtype=float), rhs)

    def verts(self, xdim=0, ydim=1, epsilon=1e-7):
        'get a 2-d projection of this lp_star'

        dims = self.a_mat.shape[0]

        if isinstance(xdim, int):
            assert 0 <= xdim < dims, f"xdim {xdim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[xdim] = 1
            xdim = vec
        else:
            assert xdim.size == dims

        if isinstance(ydim, int):
            assert 0 <= ydim < dims, f"ydim {ydim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[ydim] = 1
            ydim = vec
        else:
            assert ydim.size == dims

        def supp_point_func(vec2d):
            'maximize a support function direction'

            # use negative to maximize
            lpdir = -vec2d[0] * xdim + -vec2d[1] * ydim

            res = self.minimize_vec(lpdir)

            # project onto x and y
            resx = np.dot(xdim, res)
            resy = np.dot(ydim, res)

            return np.array([resx, resy], dtype=float)

        verts = kamenev.get_verts(2, supp_point_func, epsilon=epsilon)

        #assert np.allclose(verts[0], verts[-1])

        return verts

    def add_dense_row(self, vec, rhs):
        """intersect the domain with a linear constraint"""

        # vec * (A*alpha + b) <= rhs

        lp_vec = vec @ self.a_mat
        lp_rhs = rhs - vec @ self.b_vec

        self.hpoly.add_dense_row(lp_vec, lp_rhs)

    def is_feasible(self):
        """is the star feasible?"""

        return self.hpoly.is_feasible()

    def get_witness(self):
        """get a witness point of the star, trying to be close to the center"""

        assert self.hpoly.is_feasible()
        dims = self.hpoly.get_num_cols()
        lp_vec = np.zeros(dims)

        domain_pt = lp_vec.copy()
        num_pts = 2*dims

        for dim in range(dims):
            lp_vec[dim] = 1.0
            domain_pt += self.hpoly.minimize(lp_vec) / num_pts

            lp_vec[dim] = -1.0
            domain_pt += self.hpoly.minimize(lp_vec) / num_pts

            lp_vec[dim] = 0.0
        
        range_pt = self.a_mat @ domain_pt + self.b_vec

        return domain_pt, range_pt

    def minimize_vec(self, vec, return_io=False, fail_on_unsat=True):
        """optimize over this set

        vec is the vector of outputs we're optimizing over, None means use zero vector

        if return_io is true, returns a tuple (input, output); otherwise just output
        note that the cinput will be the compressed input if input space is not full dimensional

        returns all the outputs (coutput) if return_io=False, else (cinput, coutput)
        """

        if vec is None:
            vec = np.zeros((self.a_mat.shape[0],), dtype=float)

        assert len(vec) == self.a_mat.shape[0], f"minimize called with vector with {len(vec)} elements, " + \
            f"but set has {self.a_mat.shape[0]} outputs"

        assert isinstance(vec, np.ndarray)

        lp_vec = vec @ self.a_mat
        assert lp_vec.shape == (len(lp_vec),)

        num_init_vars = self.a_mat.shape[1]

        lp_result = self.hpoly.minimize(lp_vec, fail_on_unsat=fail_on_unsat)

        if lp_result is None:
            rv = None
        else:
            assert len(lp_result) == num_init_vars

            # convert optimization result back to output space
            rv = self.a_mat @ lp_result + self.b_vec

        # return input as well
        if rv is not None and return_io:
            rv = (lp_result, rv)

        return rv
