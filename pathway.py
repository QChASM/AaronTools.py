"""for handling the change in structure of a series of geometries"""
import numpy as np
from scipy.integrate import quad as integrate


class Pathway:
    """
    interpolating between multiple Geometries

    Attributes:
        G                       list(Geometry)          input Geometry objects
        nG                      int                     number of geometries in G
        N_Cart                  int                     number of cartesean coordinates
        B                       list(np.array(float, shape=(N_Cart x N_Cart)))
                                                    matrix representation of basis used to interpolate between geometries
        Bi                      list(np.array(float))   inverse of B
        region_length           list(float)             length of each subregion
        Geom_func(t)            function(x=float)       returns interpolated Geometry ({t | 0 <= t <= 1})
        coord_func(t)           function(x=float)       returns interpolated basis vector coefficients ({t | 0 <= t <= 1})
        dcoord_func_dt(t)       function(x=float)       derivative of coord_func(t)
        var_func(var, t)        function(x=float)       returns interpolated variable 'var' at t
        dvar_func_dt(var, t)    function(x=float)       derivative of var_func
    """

    def __init__(
        self,
        geometry,
        coordinate_array,
        other_vars={},
        basis=None,
        mass_weighted=False,
    ):
        """initialize Pathway

        geometry            Geometry
        coordinate_array    np.array(float, shape=(N, n_atoms, 3))      coordinates for the geometry at N different points
        other_vars          dict('variable name':[float])               dictionary of other variables (e.g. energy)
        basis               list(np.array(float, shape=(n_atoms,3)))    coordinate displacement matrices (shape n_atoms x 3)
        mass_weighted       bool                                        True if modes are mass-weighted

        raises RuntimeError if geometries in G_list don't have the same number of atoms
        or the atoms are in a different order"""
        self.geometry = geometry
        self.coordsets = coordinate_array
        self.other_vars = other_vars

        if basis is None:
            # if no modes are given, cartesean coordinates are used
            self.gen_basis([], mass_weighted)
        else:
            self.gen_basis(basis, mass_weighted)

        # set up coordinate and energy functions
        self.gen_func()

    @property
    def nG(self):
        """returns number of input coordinate sets"""
        return self.coordsets.shape[0]

    def gen_basis(self, modes, mass_depen=False):
        """expand number of basis vectors for coordinate displacement matrices to N_Cart
        remove mass-dependence in displacements if mass_depen
        sets self.B and self.Bi

        modes       list(array-like(float, shape=(n_atoms x 3)))    list of coordinate displacement matrices
        mass_depen  bool                                            True to remove mass-weighting from displacement coordinates
        """

        B = []
        N_Cart = 3 * len(self.geometry.atoms)

        # go through each of the supplied coordinate displacement matrices
        # remove mass-weighting if needed, and reshape
        for v, mode in enumerate(modes):
            no_mass_mode = []
            for i, atom in enumerate(self.geometry.atoms):
                if mass_depen:
                    no_mass_mode.append(mode[i] * atom.mass())
                else:
                    no_mass_mode.append(mode[i])

            B.append(np.array(np.reshape(no_mass_mode, N_Cart)))

        # we need N_Cart basis vectors
        if len(B) < N_Cart:
            if len(B) > 0:
                raise RuntimeError(
                    "number of basis vectors (%i) is less than 3N (%i)"
                    % (len(B), N_Cart)
                )
            else:
                # if we don't have any, this is equivalent to using each atom's
                # x, y, and z coordinates as our basis
                B = np.identity(N_Cart)

        B = np.transpose(B)

        self.B = B
        self.Bi = np.linalg.inv(B)

    def gen_func(self):
        """generate cartesean-displacement-representation coordinate and energy interpolation functions
        sets self.coord_func, self.dcoord_func_dt, self.E_func, and self.dE_func_dt"""

        basis_rep = []
        N_Cart = 3 * len(self.geometry.atoms)
        for XYZ in self.coordsets:
            dXYZ = XYZ - self.coordsets[0]
            a = Pathway.dXYZ_to_Q(dXYZ, self.Bi)
            basis_rep.append(np.reshape(a, (N_Cart, 1)))

        basis_rep = np.reshape(np.array(basis_rep), (N_Cart * self.nG, 1))

        # get cubic spline coefficients for the subregions
        # solved by MC = B -> C = M^-1B
        M = Pathway.get_splines_mat(self.nG)
        # B is a matrix with basis rep. coefficients, or zeros for derivative rows in M
        B = np.zeros((4 * (self.nG - 1), N_Cart))

        for i in range(0, self.nG - 1):
            for j in range(0, N_Cart):
                B[2 * i][j] = basis_rep[i * N_Cart + j][0]
                B[2 * i + 1][j] = basis_rep[(i + 1) * N_Cart + j][0]

        Mi = np.linalg.inv(M)
        C = np.dot(Mi, B)

        # get arc length for each region
        arc_length = Pathway.get_arc_length(C)

        # region_length = [simpson(arc_length, m, m+1) for m in range(0, self.nG-1)]
        region_length = [
            integrate(arc_length, m, m + 1)[0] for m in range(0, self.nG - 1)
        ]
        self.region_length = region_length

        # set self's coordinate function (coordinates are coefficients for cartesean displacement representation)
        self.coord_func, self.dcoord_func_dt = self.get_coord_func(
            C, region_length
        )
        self.var_func = {}
        self.dvar_func_dt = {}
        for var in self.other_vars:
            cVar = np.dot(Mi, Pathway.get_splines_vector(self.other_vars[var]))
            self.var_func[var], self.dvar_func_dt[var] = Pathway.get_var_func(
                cVar, region_length
            )

    def Geom_func(self, t):
        """returns a Geometry from the interpolated pathway at point t
        t       float       point on pathway {t|0 <= t <= 1}"""
        G = self.geometry.copy()
        G.update_geometry(self.coords_func(t))
        return G

    def coords_func(self, t):
        Q = self.coord_func(t)
        return self.q_to_xyz(Q)

    def get_coord_func(self, C, region_length):
        """returns function for cartesean displacement rep. coordinate as a function of t (t [0, 1])
        and a derivative of this function
        C               array-like(float, shape = (4*n_subregions, N_Cart))     matrix of cubic polynomial coefficients
        region_length   array-like(float)                                       arc length of each subregion"""
        N_Cart = 3 * len(self.geometry.atoms)

        def coord(t):
            # map input t to s and region number
            s, r = Pathway.t_to_s(t, region_length)

            # evaluate polynomial
            q = np.array(
                [
                    C[4 * r][i] * (s - r) ** 3
                    + C[4 * r + 1][i] * (s - r) ** 2
                    + C[4 * r + 2][i] * (s - r)
                    + C[4 * r + 3][i]
                    for i in range(0, N_Cart)
                ]
            )

            return q

        def dcoorddt(t):
            s, r = Pathway.t_to_s(t, region_length)

            Q = np.array(
                [
                    2 * C[4 * r][i] * (s - r) ** 2
                    + 2 * C[4 * r + 1][i] * (s - r)
                    + C[4 * r + 2][i]
                    for i in range(0, N_Cart)
                ]
            )

            return Q

        return coord, dcoorddt

    @staticmethod
    def get_var_func(cVar, region_length):
        """just like get_coord_func, but for other variables"""

        def var_func(t):
            s, r = Pathway.t_to_s(t, region_length)

            var = (
                cVar[4 * r] * (s - r) ** 3
                + cVar[4 * r + 1] * (s - r) ** 2
                + cVar[4 * r + 2] * (s - r)
                + cVar[4 * r + 3]
            )

            return var

        def dvardt_func(t):
            s, r = Pathway.t_to_s(t, region_length)

            dvar = (
                3 * cVar[4 * r] * (s - r) ** 2
                + 2 * cVar[4 * r + 1] * (s - r)
                + cVar[4 * r + 2]
            )

            return dvar

        return var_func, dvardt_func

    @staticmethod
    def dXYZ_to_Q(dXYZ, Bi):
        """converts Cartesian changes to whatever basis set Bi is (normal mode displacements/Cartesian)
        returns a vector containing the coefficients of each basis matrix"""
        q = np.reshape(dXYZ, 3 * len(dXYZ))
        a = np.dot(Bi, q)
        return a

    def q_to_xyz(self, current_q):
        coords = self.coordsets[0].copy()
        for i, mode in enumerate(np.transpose(self.B)):
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
        M = np.zeros((4 * (n_nodes - 1), 4 * (n_nodes - 1)))

        # function values are equal where regions meet
        for i in range(0, n_nodes - 1):
            M[2 * i][4 * (i + 1) - 1] = 1
            for k in range(4 * i, 4 * (i + 1)):
                M[2 * i + 1][k] = 1

        # 1st derivatives are equal where regions meet
        for i in range(0, n_nodes - 2):
            j = 2 * (n_nodes - 1) + i
            M[j][4 * i] = 3
            M[j][4 * i + 1] = 2
            M[j][4 * i + 2] = 1
            M[j][4 * i + 6] = -1

        # 2nd derivatives are equal where regions meet
        for i in range(0, n_nodes - 2):
            j = 3 * (n_nodes - 1) - 1 + i
            M[j][4 * i] = 6
            M[j][4 * i + 1] = 2
            M[j][4 * i + 5] = -2

        # 1st derivatives are 0 at the ends
        M[-2][2] = 1
        M[-1][-2] = 1
        M[-1][-3] = 2
        M[-1][-4] = 3

        return M

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
        """maps t ([0, 1]) to s (changes linearly with displacement coordinates
        need to map b/c cubic splines polynomials generated for interpolation subregions
        should be given an input between 0 and 1, no matter where they are on the whole interpolation
        s               float           point on interpolation arc
        region_length   list(float)     arc length of each region

        returns s, r
        r is the region number
        s is the point in that region"""
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
        """map s (changes linearly with displacement coordinate) to t (ranges from 0 to 1)
        s               float           point on interpolation arc
        region_length   list(float)     arc length of each region

        returns t   float"""
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
    def get_arc_length(C):
        """returns a function that can be integrated to determine the arc length of interpolation Pathway subregions
        C       array-like(float, shape = (4*n_subregions, N_Cart))     matrix of cubic polynomial coefficients

        returns function(s)"""
        N_Cart = len(C[0])

        def unnormalized_func(s):
            r = int(s)
            if r == s:
                r = int(s - 1)

            f = 0
            for i in range(0, N_Cart):
                f += (
                    3 * C[4 * r][i] * (s - r) ** 2
                    + 2 * C[4 * r + 1][i] * (s - r)
                    + C[4 * r + 2][i]
                ) ** 2

            return np.sqrt(f)

        return unnormalized_func
