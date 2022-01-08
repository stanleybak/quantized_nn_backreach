"""
Linear Star State for use in quantized backreach

Stanley Bak
Dec 2021
"""

import numpy as np

import kamenev
from lpinstance import LpInstance
from timerutil import timed

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

    @timed
    def limit_dx_dy(self, dx_range, dy_range):
        """limit dx and dy range of the star"""

        for b in self.b_vec:
            assert b == 0

        xint_row = self.a_mat[Star.X_INT]
        xown_row = self.a_mat[Star.X_OWN]
        yown_row = self.a_mat[Star.Y_OWN]

        dx_vec = xint_row - xown_row

        self.hpoly.add_dense_row(dx_vec, dx_range[1])
        self.hpoly.add_dense_row(-dx_vec, -dx_range[0])

        dy_vec = -yown_row

        self.hpoly.add_dense_row(dy_vec, dy_range[1])
        self.hpoly.add_dense_row(-dy_vec, -dy_range[0])

    @timed
    def add_dense_row(self, vec, rhs):
        """intersect the domain with a linear constraint"""

        # vec * (A*alpha + b) <= rhs

        lp_vec = vec @ self.a_mat
        lp_rhs = rhs - vec @ self.b_vec

        self.hpoly.add_dense_row(lp_vec, lp_rhs)

    @timed
    def is_feasible(self):
        """is the star feasible?

        returns None or a feasible point in the domain
        """

        return self.hpoly.is_feasible()

    @timed
    def get_witness(self, get_radius=False):
        """get a witness point of the star, using the Chebeshev center"""

        #assert self.hpoly.is_feasible()

        constraints = self.hpoly.get_constraints_csr().toarray()        
        col_bounds = self.hpoly.get_col_bounds()
        rhs_list = self.hpoly.get_rhs() # this fails if exist constriants other than '<='

        lpi = LpInstance()

        lpi.add_cols(self.hpoly.names) # note: added as free (unbounded) variables
        
        #for name, lb, ub in zip(self.hpoly.names, lbs, ubs):
        #    lpi.add_double_bounded_cols([name], lb, ub)
            
        lpi.add_positive_cols(['r']) # radius of circle
        cols = lpi.get_num_cols()

        for row, rhs in zip(constraints, rhs_list):
            norm = np.linalg.norm(row)

            v = np.zeros(cols)
            v[:-1] = row
            v[-1] = norm

            lpi.add_dense_row(v, rhs)

        # also add radius constraints related to each of the lower and upper bounds
        for i, (lb, ub) in enumerate(col_bounds):
            # example: x in [50, 100]

            if ub != np.inf:
                v = np.zeros(cols)
                v[i] = 1

                if abs(lb - ub) > 1e-6:
                    v[-1] = 1

                # x + r <= 100
                lpi.add_dense_row(v, ub)

            if lb != -np.inf:
                v = np.zeros(cols)
                v[i] = -1

                if abs(lb - ub) > 1e-6:
                    v[-1] = 1

                # x - r >= 50
                # -x + r <= -50
                lpi.add_dense_row(v, -lb)

        max_r = np.zeros(cols)
        max_r[-1] = -1

        res = lpi.minimize(max_r)
        
        domain_pt = res[:-1]
        range_pt = self.domain_to_range(domain_pt)

        if get_radius:
            radius = res[-1]
            return domain_pt, range_pt, radius
        
        return domain_pt, range_pt

    def domain_to_range(self, domain_pt):
        """convert a domain pt to a range pt"""

        return self.a_mat @ domain_pt + self.b_vec

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
