"""for handling the change in structure of a series of geometries"""
import numpy as np
from scipy.integrate import quad as integrate


class Pathway:
    """
    interpolating between multiple Geometries

    Attributes:
        geometry - structure for which coordinates are interpolated
        num_geom - number of geometries
        n_cart - number of Cartesian coordinates
        basis -  matrix representation of basis used to interpolate between geometries
        basis_inverse - inverse of basis
        region_length - length of each spline subregion
    """

    def __init__(
            self,
            geometry,
            coordinate_array,
            other_vars=None,
            basis=None,
            mass_weighted=False,
    ):
        """
        geometry - Geometry()
        coordinate_array - np.array(float, shape=(N, n_atoms, 3))
                           coordinates for the geometry at N different points
        other_vars - dict('variable name':[float])
                     dictionary of other variables (e.g. energy)
        basis - list(np.array(float, shape=(n_atoms,3)))
                coordinate displacement matrices (shape n_atoms x 3)
        mass_weighted - bool, True if supplied modes are mass-weighted
        """
        self.geometry = geometry
        self.coordsets = coordinate_array
        if other_vars:
            self.other_vars = other_vars
        else:
            self.other_vars = {}

        if basis is None:
            # if no modes are given, Cartesian coordinates are used
            self.gen_basis([], mass_weighted)
        else:
            self.gen_basis(basis, mass_weighted)

        # set up coordinate and energy functions
        self.gen_func()

    @property
    def num_geom(self):
        """returns number of input coordinate sets"""
        return self.coordsets.shape[0]

    def gen_basis(self, modes, remove_mass_weight=False):
        """
        modes - list of coordinate displacement matrices
        remove_mass_weight  bool, True to remove mass-weighting from displacement coordinates
        """

        basis = []
        n_cart = 3 * len(self.geometry.atoms)

        # go through each of the supplied coordinate displacement matrices
        # remove mass-weighting if needed, and reshape
        for mode in modes:
            no_mass_mode = []
            for i, atom in enumerate(self.geometry.atoms):
                if remove_mass_weight:
                    no_mass_mode.append(mode[i] * atom.mass())
                else:
                    no_mass_mode.append(mode[i])

            basis.append(np.array(np.reshape(no_mass_mode, n_cart)))

        # we need n_cart basis vectors
        if len(basis) < n_cart:
            if basis:
                raise RuntimeError(
                    "number of basis vectors (%i) is less than 3N (%i)"
                    % (len(basis), n_cart)
                )

            # if we don't have any, this is equivalent to using each atom's
            # x, y, and z coordinates as our basis
            basis = np.identity(n_cart)

        basis = np.transpose(basis)

        self.basis = basis
        self.basis_inverse = np.linalg.inv(basis)

    def gen_func(self):
        """
        generate Cartesian-displacement-representation coordinate and
        miscellaneous variable interpolation functions
        sets self.coord_func, self.dcoord_func_dt, self.E_func, and self.dE_func_dt
        """

        basis_rep = []
        n_cart = 3 * len(self.geometry.atoms)
        for xyz in self.coordsets:
            d_xyz = xyz - self.coordsets[0]
            a = Pathway.dxyz_to_q(d_xyz, self.basis_inverse)
            basis_rep.append(np.reshape(a, (n_cart, 1)))

        basis_rep = np.reshape(np.array(basis_rep), (n_cart * self.num_geom, 1))

        # get cubic spline coefficients for the subregions
        # solved by mat * coord = basis -> coord = mat^-1 * basis
        mat = Pathway.get_splines_mat(self.num_geom)
        # basis is a matrix with basis rep. coefficients, or zeros for derivative rows in M
        basis = np.zeros((4 * (self.num_geom - 1), n_cart))

        for i in range(0, self.num_geom - 1):
            for j in range(0, n_cart):
                basis[2 * i][j] = basis_rep[i * n_cart + j][0]
                basis[2 * i + 1][j] = basis_rep[(i + 1) * n_cart + j][0]

        mat_i = np.linalg.inv(mat)
        coord = np.dot(mat_i, basis)

        # get arc length for each region
        arc_length = Pathway.get_arc_length(coord)

        # region_length = [simpson(arc_length, m, m+1) for m in range(0, self.num_geom-1)]
        region_length = [
            integrate(arc_length, m, m + 1)[0] for m in range(0, self.num_geom - 1)
        ]
        self.region_length = region_length

        # set self's coordinate function
        # coordinates are coefficients for Cartesian displacement representation
        self.coord_func, self.dcoord_func_dt = self.get_coord_func(
            coord, region_length
        )
        self.var_func = {}
        self.dvar_func_dt = {}
        for var in self.other_vars:
            c_var = np.dot(mat_i, Pathway.get_splines_vector(self.other_vars[var]))
            self.var_func[var], self.dvar_func_dt[var] = Pathway.get_var_func(
                c_var, region_length
            )

    def geom_func(self, t):
        """returns a Geometry from the interpolated pathway at point t
        t       float       point on pathway {t|0 <= t <= 1}"""
        geom = self.geometry.copy()
        geom.update_geometry(self.coords_func(t))
        return geom

    def coords_func(self, t):
        """returns Cartesian coordinates for the geometry at point t"""
        Q = self.coord_func(t)
        return self.q_to_xyz(Q)

    def get_coord_func(self, coord, region_length):
        """
        returns function for Cartesian displacement coordinate as a function of t (t [0, 1])
        and a derivative of this function
        coord - array-like(float, shape = (4*n_subregions, n_cart))
                matrix of cubic polynomial coefficients
        region_length - array-like(float)
                        arc length of each subregion
        """
        n_cart = 3 * len(self.geometry.atoms)

        def coord_fun(t):
            # map input t to s and region number
            s, r = Pathway.t_to_s(t, region_length)

            # evaluate polynomial
            q = np.array(
                [
                    coord[4 * r][i] * (s - r) ** 3
                    + coord[4 * r + 1][i] * (s - r) ** 2
                    + coord[4 * r + 2][i] * (s - r)
                    + coord[4 * r + 3][i]
                    for i in range(0, n_cart)
                ]
            )

            return q

        def dcoord_dt(t):
            s, r = Pathway.t_to_s(t, region_length)

            q = np.array(
                [
                    3 * coord[4 * r][i] * (s - r) ** 2
                    + 2 * coord[4 * r + 1][i] * (s - r)
                    + coord[4 * r + 2][i]
                    for i in range(0, n_cart)
                ]
            )

            return q

        return coord_fun, dcoord_dt

    @staticmethod
    def get_var_func(c_var, region_length):
        """just like get_coord_func, but for other variables"""

        def var_func(t):
            s, r = Pathway.t_to_s(t, region_length)

            var = (
                c_var[4 * r] * (s - r) ** 3
                + c_var[4 * r + 1] * (s - r) ** 2
                + c_var[4 * r + 2] * (s - r)
                + c_var[4 * r + 3]
            )

            return var

        def dvardt_func(t):
            s, r = Pathway.t_to_s(t, region_length)

            dvar = (
                3 * c_var[4 * r] * (s - r) ** 2
                + 2 * c_var[4 * r + 1] * (s - r)
                + c_var[4 * r + 2]
            )

            return dvar

        return var_func, dvardt_func

    @staticmethod
    def dxyz_to_q(dxyz, basis_inverse):
        """
        converts Cartesian changes (dxyz) to whatever basis set basis_inverse
        is (normal mode displacements/Cartesian)
        returns a vector containing the coefficients of each basis matrix
        """
        q = np.reshape(dxyz, 3 * len(dxyz))
        a = np.dot(basis_inverse, q)
        return a

    def q_to_xyz(self, current_q):
        """converts coordinates for self.basis to Cartesian"""
        coords = self.coordsets[0].copy()
        for i, mode in enumerate(np.transpose(self.basis)):
            coords += current_q[i] * np.reshape(
                mode, (len(self.geometry.atoms), 3)
            )

        return coords

    @staticmethod
    def get_splines_mat(n_nodes):
        """generate matrix for fitting cubic splines to data
        matrix is 4*n_regions x 4*n_regions (n_regions = n_nodes-1)
        additional contraints (that might not be valid) as that
        the first derivatives are 0 at both ends of the interpolation
        region (e.g. f'(0) = 0 and f'(1) = 0)"""
        mat = np.zeros((4 * (n_nodes - 1), 4 * (n_nodes - 1)))

        # function values are equal where regions meet
        for i in range(0, n_nodes - 1):
            mat[2 * i][4 * (i + 1) - 1] = 1
            for k in range(4 * i, 4 * (i + 1)):
                mat[2 * i + 1][k] = 1

        # 1st derivatives are equal where regions meet
        for i in range(0, n_nodes - 2):
            j = 2 * (n_nodes - 1) + i
            mat[j][4 * i] = 3
            mat[j][4 * i + 1] = 2
            mat[j][4 * i + 2] = 1
            mat[j][4 * i + 6] = -1

        # 2nd derivatives are equal where regions meet
        for i in range(0, n_nodes - 2):
            j = 3 * (n_nodes - 1) - 1 + i
            mat[j][4 * i] = 6
            mat[j][4 * i + 1] = 2
            mat[j][4 * i + 5] = -2

        # 1st derivatives are 0 at the ends
        mat[-2][2] = 1
        mat[-1][-2] = 1
        mat[-1][-3] = 2
        mat[-1][-4] = 3

        return mat

    @staticmethod
    def get_splines_vector(data):
        """organize data into a vector that can be used with cubic splines matrix"""
        n_regions = len(data) - 1

        v = np.zeros(4 * (n_regions))

        for i in range(0, n_regions):
            v[2 * i] = data[i]
            v[2 * i + 1] = data[i + 1]

        return v

    @staticmethod
    def t_to_s(t, region_length):
        """
        maps t ([0, 1]) to s (changes linearly with displacement coordinates
        need to map b/c cubic splines polynomials generated for interpolation subregions
        should be given an input between 0 and 1, no matter where they are on the
        whole interpolation
        s               float           point on interpolation arc
        region_length   list(float)     arc length of each region

        returns s, r
        r is the region number
        s is the point in that region
        """
        n_regions = len(region_length)
        path_length = sum(region_length)
        region_start = [
            sum(region_length[:i]) for i in range(0, len(region_length))
        ]
        u = t * path_length
        r = 0
        for l in range(0, n_regions):
            if u > region_start[l]:
                r = l

        s = r + (u - region_start[r]) / region_length[r]
        return s, r

    @staticmethod
    def s_to_t(s, region_length):
        """
        map s (changes linearly with displacement coordinate) to t (ranges from 0 to 1)
        s               float           point on interpolation arc
        region_length   list(float)     arc length of each region

        returns t   float
        """
        n_regions = len(region_length)
        path_length = sum(region_length)
        region_start = [
            sum(region_length[:i]) for i in range(0, len(region_length))
        ]
        r = int(s)
        while r >= (n_regions):
            r -= 1
        u = (s - r) * region_length[r] + region_start[r]
        t = u / path_length
        return t

    @staticmethod
    def get_arc_length(coord):
        """
        returns a function that can be integrated to determine the arc length
        of interpolation splines before normalization
        coord - array-like(float, shape = (4*n_subregions, n_cart))
                matrix of cubic polynomial coefficients

        returns function(s)
        """
        n_cart = len(coord[0])

        def unnormalized_func(s):
            r = int(s)
            if r == s:
                r = int(s - 1)

            f = 0
            for i in range(0, n_cart):
                f += (
                    3 * coord[4 * r][i] * (s - r) ** 2
                    + 2 * coord[4 * r + 1][i] * (s - r)
                    + coord[4 * r + 2][i]
                ) ** 2

            return np.sqrt(f)

        return unnormalized_func
