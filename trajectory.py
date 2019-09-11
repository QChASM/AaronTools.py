"""for handling the change in structure of a series of geometries"""
import numpy as np
from scipy import integrate
from scipy.linalg import null_space

class Pathway:
    """
    interpolating between multiple Geometries
    
    Attributes:
        G                   list(Geometry)          input Geometry objects
        nG                  int                     number of geometries in G
        N_Cart              int                     number of cartesean coordinates
        B                   list(np.array(float, shape=(N_Cart x N_Cart)))
                                                    matrix representation of basis used to interpolate between geometries
        Bi                  list(np.array(float))   inverse of B
        region_length       list(float)             length of each subregion
        Geom_func(t)        function(x=float)       returns interpolated Geometry ({t | 0 <= t <= 1})
        coord_func(t)       function(x=float)       returns interpolated basis vector coefficients ({t | 0 <= t <= 1})
        dcoord_func_dt(t)   function(x=float)       derivative of coord_func(t)
        E_func(t)           function(x=float)       returns interpolated energies
        dE_func_dt(t)       function(x=float)       derivative of E_func
    """

    def __init__(self, G_list, modes = None, mass_weighted = False):
        """initialize Pathway

            G_list          list(Geometry)      Geometry objects
            modes           list(np.array(float, shape=(n_atoms,3)))
                                                coordinate displacement matrices (shape n_atoms x 3)
            mass_weighted   bool                True if modes are mass-weighted

            raises RuntimeError if geometries in G_list don't have the same number of atoms
            or the atoms are in a different order"""
        self.G = G_list

        # check to make sure the atoms have a chance of being in the same order in all input geometries
        if not all([ len(G.atoms) == len(self.G[0].atoms) for G in self.G[1:] ]):
            s = "\n"
            frmt = "%" + str(max([len(G.name) for G in self.G])) + "s %s\n"
            for G in self.G:
                s += frmt % (G.name, len(G.atoms))
            raise RuntimeError("not all input geometries have the same number of atoms: %s" % s)

        if not all([ all([G.atoms[i].element == self.G[0].atoms[i].element \
            for i in range(0, len(self.G[0].atoms))]) \
            for G in self.G[1:]]):
            s = "\n"
            frmt = "%" + str(max([len(G.name) for G in self.G])) + "s %s\n"
            for G in self.G:
                s += frmt % (G.name, ' '.join([atom.element for atom in G.atoms]))
            raise RuntimeError("atoms are not in the same order in some of the input geometries: %s" % s)

        if modes is None:
            #if no modes are given, cartesean coordinates are used
            self.gen_basis([], mass_weighted)
        else:
            self.gen_basis(modes, mass_weighted)

        #set up coordinate and energy functions
        self.gen_func()

    @property
    def nG(self):
        """returns len(self.G_list)"""
        return len(self.G)

    def gen_basis(self, modes, mass_depen = False):
        """expand number of basis vectors for coordinate displacement matrices to N_Cart
        remove mass-dependence in displacements if mass_depen
        sets self.B and self.Bi

        modes       list(array-like(float, shape=(n_atoms x 3)))    list of coordinate displacement matrices
        mass_depen  bool                                            True to remove mass-weighting from displacement coordinates
        """

        B = []
        N_Cart = 3*len(self.G[0].atoms)

        #go through each of the supplied coordinate displacement matrices
        #remove mass-weighting if needed, and reshape
        for v, mode in enumerate(modes):
            no_mass_mode = []
            for i, atom in enumerate(self.G[0].atoms):
                if mass_depen:
                    no_mass_mode.append(mode[i] * atom.mass())
                else:
                    no_mass_mode.append(mode[i])

            B.append(np.array(np.reshape(no_mass_mode, N_Cart)))

        #we need N_Cart basis vectors
        if len(B) < N_Cart:
            if len(B) > 0:
                #if we already have some vectors, get the rest from the null space
                #of the current set
                B = np.array(B)
                ns = np.transpose(null_space(B))
                B = np.concatenate((B, ns))
            else:
                #if we don't have any, this is equivalent to using each atom's
                #x, y, and z coordinates as our basis
                B = np.identity(N_Cart)

        B = np.transpose(B)

        self.B = B
        self.Bi = np.linalg.inv(B)

    def gen_func(self):
        """generate cartesean-displacement-representation coordinate and energy interpolation functions
        sets self.coord_func, self.dcoord_func_dt, self.E_func, and self.dE_func_dt"""

        #determine coeffiencts for basis vectors that will transform self.G[0] into each
        #of the other Geometries in self.G
        basis_rep = []
        N_Cart = 3*len(self.G[0].atoms)
        for G in self.G:
            XYZ = G.coords()
            dXYZ = XYZ - self.G[0].coords()
            a = Pathway.dXYZ_to_Q(dXYZ, self.Bi)
            basis_rep.append(np.reshape(a, (N_Cart, 1)))

        basis_rep = np.reshape(np.array(basis_rep), (N_Cart*self.nG, 1))

        #get cubic spline coefficients for the subregions
        #solved by MC = B -> C = M^-1B
        M = Pathway.get_splines_mat(self.nG)
        #B is a matrix with basis rep. coefficients, or zeros for derivative rows in M
        B = np.zeros( (4*(self.nG-1), N_Cart) )
        if all([hasattr(G, 'other') for G in self.G]) and all(["energy" in G.other for G in self.G]):
            E = Pathway.get_splines_vector( [G.other['energy'] for G in self.G] )
        else:
            E = Pathway.get_splines_vector( [0 for G in self.G] )

        for i in range(0, self.nG-1):
            for j in range(0, N_Cart):
                B[2*i][j] = basis_rep[i*N_Cart+j][0]
                B[2*i+1][j] = basis_rep[(i+1)*N_Cart+j][0]

        Mi = np.linalg.inv(M)
        C = np.dot(Mi, B)
        cE = np.dot(Mi, E)

        #get arc length for each region
        arc_length = Pathway.get_arc_length(C)

        #region_length = [simpson(arc_length, m, m+1) for m in range(0, self.nG-1)]
        region_length = [integrate.quad(arc_length, m, m+1)[0] for m in range(0, self.nG-1)]
        self.region_length = region_length

        #set self's coordinate function (coordinates are coefficients for cartesean displacement representation)
        self.coord_func, self.dcoord_func_dt = self.get_coord_func(C, region_length)
        self.E_func, self.dE_func = Pathway.get_E_func(cE, region_length)

    def Geom_func(self, t):
        """returns a Geometry from the interpolated pathway at point t
        t       float       point on pathway {t|0 <= t <= 1}"""
        Q = self.coord_func(t)
        G = Pathway.q_to_xyz(self.G[0], self.B, Q)
        return G

    def get_coord_func(self, C, region_length):
        """returns function for cartesean displacement rep. coordinate as a function of t (t [0, 1])
        and a derivative of this function
        C               array-like(float, shape = (4*n_subregions, N_Cart))     matrix of cubic polynomial coefficients
        region_length   array-like(float)                                       arc length of each subregion"""
        N_Cart = 3*len(self.G[0].atoms)
        def coord(t):
            #map input t to s and region number
            s, r = Pathway.t_to_s(t, region_length)

            #evaluate polynomial
            q = np.array([C[4*r][i]*(s-r)**3 + C[4*r+1][i]*(s-r)**2 + C[4*r+2][i]*(s-r) + C[4*r+3][i] for i in range(0, N_Cart)])

            return q

        def dcoorddt(t):
            s, r = Pathway.t_to_s(t, region_length)

            Q = np.array([2*C[4*r][i]*(s-r)**2 + 2*C[4*r+1][i]*(s-r) + C[4*r+2][i] for i in range(0, N_Cart)])

            return Q

        return coord, dcoorddt

    def get_E_func(cE, region_length):
        """just like get_coord_func, but for energies"""
        def E_func(t):
            s, r = Pathway.t_to_s(t, region_length)

            E = cE[4*r]*(s-r)**3 + cE[4*r+1]*(s-r)**2 + cE[4*r+2]*(s-r) + cE[4*r+3]

            return E

        def dEdt_func(t):
            s, r = Pathway.t_to_s(t, region_length)

            dE = 3*cE[4*r]*(s-r)**2 + 2*cE[4*r+1]*(s-r) + cE[4*r+2]

            return dE

        return E_func, dEdt_func

    def dXYZ_to_Q(dXYZ, Bi):
        """converts Cartesian changes to whatever basis set Bi is (normal mode displacements/Cartesian)
        returns a vector containing the coefficients of each basis matrix"""
        q = np.reshape(dXYZ, 3*len(dXYZ))
        a = np.dot(Bi, q)
        return a

    def q_to_xyz(Gi, modes, current_q):
        """takes current_q weights for modes (basis matrices), and a Geometry
        returns Geometry that has coordinates modified"""
        G = Gi.copy()
        X = G.coords()
        for i, mode in enumerate(np.transpose(modes)):
            X += current_q[i]*np.reshape(mode, (len(G.atoms), 3))

        G.update_geometry(X)

        return G

    def get_splines_mat(n_nodes):
        """generate matrix for fitting cubic splines to data
        matrix is 4*n_regions x 4*n_regions (n_regions = n_nodes-1)
        additional contraints (that might not be valid) as that
        the first derivatives are 0 at both ends of the interpolation
        region (e.g. f'(0) = 0 and f'(1) = 0)"""
        M = np.zeros( (4*(n_nodes-1), 4*(n_nodes-1)) )

        #function values are equal where regions meet
        for i in range(0, n_nodes-1):
            M[2*i][4*(i+1)-1] = 1
            for k in range(4*i, 4*(i+1)):
                M[2*i+1][k] = 1

        #1st derivatives are equal where regions meet
        for i in range(0, n_nodes-2):
            j = 2*(n_nodes-1) + i
            M[j][4*i] = 3
            M[j][4*i+1] = 2
            M[j][4*i+2] = 1
            M[j][4*i+6] = -1

        #2nd derivatives are equal where regions meet
        for i in range(0, n_nodes-2):
            j = 3*(n_nodes-1) - 1 + i
            M[j][4*i] = 6
            M[j][4*i+1] = 2
            M[j][4*i+5] = -2

        #1st derivatives are 0 at the ends
        M[-2][2] = 1
        M[-1][-2] = 1
        M[-1][-3] = 2
        M[-1][-4] = 3

        return M

    def get_splines_vector(data):
        """organize data into a vector that can be used with cubic splines matrix"""
        n_regions = len(data) - 1

        v = np.zeros( 4*(n_regions) )

        for i in range(0, n_regions):
            v[2*i] = data[i]
            v[2*i+1] = data[i+1]

        return v

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
        region_start = [sum(region_length[:i]) for i in range(0, len(region_length))]
        u = t*path_length
        r = 0
        for l in range(0, n_regions):
            if u > region_start[l]:
                r = l

        s = r+(u-region_start[r])/region_length[r]
        return s, r

    def s_to_t(s, region_length):
        """map s (changes linearly with displacement coordinate) to t (ranges from 0 to 1)
        s               float           point on interpolation arc
        region_length   list(float)     arc length of each region

        returns t   float"""
        n_regions = len(region_length)
        path_length = sum(region_length)
        region_start = [sum(region_length[:i]) for i in range(0, len(region_length))]
        r = int(s)
        while r >= (n_regions):
            r -= 1
        u = (s-r) * region_length[r] + region_start[r]
        t = u/path_length
        return t

    def get_arc_length(C):
        """returns a function that can be integrated to determine the arc length of interpolation Pathway subregions
        C       array-like(float, shape = (4*n_subregions, N_Cart))     matrix of cubic polynomial coefficients

        returns function(s)"""
        N_Cart = len(C[0])
        def unnormalized_func(s):
            r = int(s)
            if r == s:
                r = int(s-1)

            f = 0
            for i in range(0, N_Cart):
                f += (3*C[4*r][i]*(s-r)**2+2*C[4*r+1][i]*(s-r)+C[4*r+2][i])**2

            f = np.sqrt(f)

            return f

        return unnormalized_func

