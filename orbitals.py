import concurrent.futures

import numpy as np

from scipy.spatial import distance_matrix
from scipy.special import factorial2

from AaronTools import addlogger
from AaronTools.const import ELEMENTS, UNIT, VDW_RADII, BONDI_RADII
from AaronTools.utils.utils import lebedev_sphere, gauss_legendre_grid


@addlogger
class Orbitals:
    """
    stores functions for the shells in a basis set
    for evaluation at arbitrary points
    attributes:
    basis_functions - list(len=n_shell) of lists(len=n_prim_per_shell)
                      of functions
                      function takes the arguments:
                      r2 - float array like, squared distance from the
                           shell's center to each point being evaluated
                      x - float or array like, distance from the shell's
                          center to the point(s) being evaluated along
                          the x axis
                      y and z - same as x for the corresponding axis
                      mo_coeffs - list(len=funcs_per_shell), MO coefficients
                                  for the functions in this shell (e.g. 3
                                  coefficients for the p shell); order
                                  might depend on input file format
                                  for example, FCHK files will be px, py, pz
                                  ORCA files will be pz, px, py
    funcs_per_shell - list(len=n_shell), number of basis functions for
                      each shell
    alpha_coefficients - array(shape=(n_mos, n_mos)), coefficients of
                         molecular orbitals for alpha electrons
    beta_coefficients - same as alpha_coefficients for beta electrons
    shell_coords - array(shape=(n_shells, 3)), coordinates of each shell
                   in Angstroms
    shell_types - list(str, len=n_shell), type of each shell (e.g. s,
                  p, sp, 5d, 6d...)
    n_shell - number of shells
    n_prim_per_shell - list(len=n_shell), number of primitives per shell
    n_mos - number of molecular orbitals
    exponents - array, exponents for primitives in Eh
                each shell
    alpha_nrgs - array(len=n_mos), energy of alpha MO's
    beta_nrgs - array(len=n_mos), energy of beta MO's
    contraction_coeff - array, contraction coefficients for each primitive
                        in each shell
    n_alpha - int, number of alpha electrons
    n_beta - int, number of beta electrons
    """

    LOG = None

    def __init__(self, filereader):
        if filereader.file_type == "fchk":
            self._load_fchk_data(filereader)
        elif filereader.file_type == "out":
            self._load_orca_out_data(filereader)
        elif filereader.file_type == "47" or filereader.file_type == "31":
            self._load_nbo_data(filereader)
        else:
            raise NotImplementedError(
                "cannot load orbital info from %s files" % filereader.file_type
            )

    def _load_fchk_data(self, filereader):
        self.alpha_occupancies = None
        self.beta_occupancies = None
        if "Coordinates of each shell" in filereader.other:
            self.shell_coords = np.reshape(
                filereader.other["Coordinates of each shell"],
                (len(filereader.other["Shell types"]), 3),
            )
        else:
            center_coords = []
            for ndx in filereader.other["Shell to atom map"]:
                center_coords.append(filereader.atoms[ndx - 1].coords)
            self.center_coords = np.array(center_coords)
        self.shell_coords *= UNIT.A0_TO_BOHR
        self.contraction_coeff = filereader.other["Contraction coefficients"]
        self.exponents = filereader.other["Primitive exponents"]
        self.n_prim_per_shell = filereader.other["Number of primitives per shell"]
        self.alpha_nrgs = filereader.other["Alpha Orbital Energies"]
        self.scf_density = filereader.other["Total SCF Density"]
        self.beta_nrgs = None
        if "Beta Orbital Energies" in filereader.other:
            self.beta_nrgs = filereader.other["Beta Orbital Energies"]

        self.funcs_per_shell = []

        def gau_norm(a, l):
            """
            normalization for gaussian primitives that depends on
            the exponential (a) and the total angular momentum (l)
            """
            t1 = np.sqrt((2 * a) ** (l + 3 / 2)) / (np.pi ** (3.0 / 4))
            t2 = np.sqrt(2 ** l / factorial2(2 * l - 1))
            return t1 * t2

        # get functions for norm of s, p, 5d, and 7f
        s_norm = lambda a, l=0: gau_norm(a, l)
        p_norm = lambda a, l=1: gau_norm(a, l)
        d_norm = lambda a, l=2: gau_norm(a, l)
        f_norm = lambda a, l=3: gau_norm(a, l)
        g_norm = lambda a, l=4: gau_norm(a, l)
        h_norm = lambda a, l=5: gau_norm(a, l)
        i_norm = lambda a, l=6: gau_norm(a, l)

        self.basis_functions = list()

        self.n_mos = 0
        self.shell_types = []
        shell_i = 0
        for n_prim, shell in zip(
            self.n_prim_per_shell,
            filereader.other["Shell types"],
        ):
            exponents = self.exponents[shell_i : shell_i + n_prim]
            con_coeff = self.contraction_coeff[shell_i : shell_i + n_prim]

            if shell == 0:
                # s functions
                self.shell_types.append("s")
                self.n_mos += 1
                self.funcs_per_shell.append(1)
                norms = s_norm(exponents)
                if n_prim > 1:
                    def s_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        return mo_coeffs[0] * np.dot(con_coeff * norms, e_r2)
                else:
                    def s_shell(
                        r2, x, y, z, mo_coeff,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        return mo_coeff * con_coeff * norms * e_r2
                self.basis_functions.append(s_shell)

            elif shell == 1:
                # p functions
                self.shell_types.append("p")
                self.n_mos += 3
                self.funcs_per_shell.append(3)
                norms = p_norm(exponents)
                if n_prim > 1:
                    def p_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            res += mo_coeffs[0] * x
                        if mo_coeffs[1] != 0:
                            res += mo_coeffs[1] * y
                        if mo_coeffs[2] != 0:
                            res += mo_coeffs[2] * z
                        return res * s_val
                else:
                    def p_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        s_val = con_coeff * norms * e_r2
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            res += mo_coeffs[0] * x
                        if mo_coeffs[1] != 0:
                            res += mo_coeffs[1] * y
                        if mo_coeffs[2] != 0:
                            res += mo_coeffs[2] * z
                        return res * s_val
                self.basis_functions.append(p_shell)

            elif shell == -1:
                # s=p functions
                self.shell_types.append("sp")
                self.n_mos += 4
                self.funcs_per_shell.append(4)
                norm_s = s_norm(exponents)
                norm_p = p_norm(exponents)
                sp_coeff = filereader.other["P(S=P) Contraction coefficients"][shell_i: shell_i + n_prim]
                if n_prim > 1:
                    def sp_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents,
                        s_coeff=con_coeff,
                        p_coeff=sp_coeff,
                        s_norms=norm_s,
                        p_norms=norm_p,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        sp_val_s = np.dot(s_coeff * s_norms, e_r2)
                        sp_val_p = np.dot(p_coeff * p_norms, e_r2)
                        s_res = np.zeros(len(r2))
                        p_res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            s_res += mo_coeffs[0]
                        if mo_coeffs[1] != 0:
                            p_res += mo_coeffs[1] * x
                        if mo_coeffs[2] != 0:
                            p_res += mo_coeffs[2] * y
                        if mo_coeffs[3] != 0:
                            p_res += mo_coeffs[3] * z
                        return s_res * sp_val_s + p_res * sp_val_p
                else:
                    def sp_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents,
                        s_coeff=con_coeff,
                        p_coeff=sp_coeff,
                        s_norms=norm_s,
                        p_norms=norm_p,
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        sp_val_s = s_coeff * s_norms * e_r2
                        sp_val_p = p_coeff * p_norms * e_r2
                        s_res = np.zeros(len(r2))
                        p_res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            s_res += mo_coeffs[0]
                        if mo_coeffs[1] != 0:
                            p_res += mo_coeffs[1] * x
                        if mo_coeffs[2] != 0:
                            p_res += mo_coeffs[2] * y
                        if mo_coeffs[3] != 0:
                            p_res += mo_coeffs[3] * z
                        return s_res * sp_val_s + p_res * sp_val_p
                self.basis_functions.append(sp_shell)

            elif shell == 2:
                # cartesian d functions
                self.shell_types.append("6d")
                self.n_mos += 6
                self.funcs_per_shell.append(6)
                norms = d_norm(exponents)
                if n_prim > 1:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            xx = x ** 2
                            res += mo_coeffs[0] * xx
                        if mo_coeffs[1] != 0:
                            yy = y ** 2
                            res += mo_coeffs[1] * yy
                        if mo_coeffs[2] != 0:
                            zz = z ** 2
                            res += mo_coeffs[2] * zz
                        if mo_coeffs[3] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[3] * xy
                        if mo_coeffs[4] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[4] * xz
                        if mo_coeffs[5] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[5] * yz
                        return res * s_val
                else:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        s_val = con_coeff * norms * e_r2
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            xx = x ** 2
                            res += mo_coeffs[0] * xx
                        if mo_coeffs[1] != 0:
                            yy = y ** 2
                            res += mo_coeffs[1] * yy
                        if mo_coeffs[2] != 0:
                            zz = z ** 2
                            res += mo_coeffs[2] * zz
                        if mo_coeffs[3] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[3] * xy
                        if mo_coeffs[4] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[4] * xz
                        if mo_coeffs[5] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[5] * yz
                        return res * s_val
                self.basis_functions.append(d_shell)

            elif shell == -2:
                # pure d functions
                self.shell_types.append("5d")
                self.n_mos += 5
                self.funcs_per_shell.append(5)
                norms = d_norm(exponents)
                if n_prim > 1:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z2r2 = 0.5 * (3 * z ** 2 - r2)
                            res += mo_coeffs[0] * z2r2
                        if mo_coeffs[1] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[1] * xz
                        if mo_coeffs[2] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[2] * yz
                        if mo_coeffs[3] != 0:
                            x2y2 = np.sqrt(3) * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2y2
                        if mo_coeffs[4] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[4] * xy
                        return res * s_val
                else:
                    def d_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeff=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(-alpha * r2)
                        s_val = con_coeff * norms * e_r2
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z2r2 = 0.5 * (3 * z ** 2 - r2)
                            res += mo_coeffs[0] * z2r2
                        if mo_coeffs[1] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[1] * xz
                        if mo_coeffs[2] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[2] * yz
                        if mo_coeffs[3] != 0:
                            x2y2 = np.sqrt(3) * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2y2
                        if mo_coeffs[4] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[4] * xy
                        return res * s_val
                self.basis_functions.append(d_shell)

            elif shell == 3:
                # 10f functions
                self.shell_types.append("10f")
                self.n_mos += 10
                self.funcs_per_shell.append(10)
                norms = f_norm(exponents)

                def f_shell(
                    r2,
                    x,
                    y,
                    z,
                    mo_coeffs,
                    alpha=exponents,
                    con_coeff=con_coeff,
                    norms=norms,
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    # ** 3 takes ~6x longer than x * x * x or x ** 2 * x
                    if mo_coeffs[0] != 0:
                        xxx = x * x * x
                        res += mo_coeffs[0] * xxx
                    if mo_coeffs[1] != 0:
                        yyy = y * y * y
                        res += mo_coeffs[1] * yyy
                    if mo_coeffs[2] != 0:
                        zzz = z * z * z
                        res += mo_coeffs[2] * zzz
                    if mo_coeffs[3] != 0:
                        xyy = np.sqrt(5) * x * y ** 2
                        res += mo_coeffs[3] * xyy
                    if mo_coeffs[4] != 0:
                        xxy = np.sqrt(5) * x ** 2 * y
                        res += mo_coeffs[4] * xxy
                    if mo_coeffs[5] != 0:
                        xxz = np.sqrt(5) * x ** 2 * z
                        res += mo_coeffs[5] * xxz
                    if mo_coeffs[6] != 0:
                        xzz = np.sqrt(5) * x * z ** 2
                        res += mo_coeffs[6] * xzz
                    if mo_coeffs[7] != 0:
                        yzz = np.sqrt(5) * y * z ** 2
                        res += mo_coeffs[7] * yzz
                    if mo_coeffs[8] != 0:
                        yyz = np.sqrt(5) * y ** 2 * z
                        res += mo_coeffs[8] * yyz
                    if mo_coeffs[9] != 0:
                        xyz = np.sqrt(15) * x * y * z
                        res += mo_coeffs[9] * xyz
                    return res * s_val

                self.basis_functions.append(f_shell)

            elif shell == -3:
                # pure f functions
                self.shell_types.append("7f")
                self.n_mos += 7
                self.funcs_per_shell.append(7)
                norms = f_norm(exponents)
                def f_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeff=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    if mo_coeffs[0] != 0:
                        z3zr2 = z * (5 * z ** 2 - 3 * r2) / 2
                        res += mo_coeffs[0] * z3zr2
                    if mo_coeffs[1] != 0:
                        xz2xr2 = np.sqrt(3) * x * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        res += mo_coeffs[1] * xz2xr2
                    if mo_coeffs[2] != 0:
                        yz2yr2 = np.sqrt(3) * y * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        res += mo_coeffs[2] * yz2yr2
                    if mo_coeffs[3] != 0:
                        x2zr2z = np.sqrt(15) * z * (x ** 2 - y ** 2) / 2
                        res += mo_coeffs[3] * x2zr2z
                    if mo_coeffs[4] != 0:
                        xyz = np.sqrt(15) * x * y * z
                        res += mo_coeffs[4] * xyz
                    if mo_coeffs[5] != 0:
                        x3r2x = np.sqrt(5) * x * (x ** 2 - 3 * y ** 2) / (2 * np.sqrt(2))
                        res += mo_coeffs[5] * x3r2x
                    if mo_coeffs[6] != 0:
                        x2yy3 = np.sqrt(5) * y * (3 * x ** 2 - y ** 2) / (2 * np.sqrt(2))
                        res += mo_coeffs[6] * x2yy3
                    return res * s_val
                self.basis_functions.append(f_shell)
                
            # elif shell == 4:
            elif False:
                # TODO: validate these - order might be wrong
                self.shell_types.append("15g")
                self.n_mos += 15
                self.funcs_per_shell.append(15)
                norms = g_norm(exponents)
                def g_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    if mo_coeffs[0] != 0:
                        x4 = (x ** 2) ** 2
                        res += mo_coeffs[0] * x4
                    if mo_coeffs[1] != 0:
                        x3y = (x ** 3) * x * y
                        res += mo_coeffs[1] * x3y
                    if mo_coeffs[2] != 0:
                        x3z = (x ** 3) * x * z
                        res += mo_coeffs[2] * x3z
                    if mo_coeffs[3] != 0:
                        x2y2 = (x ** 2) * (y ** 2)
                        res += mo_coeffs[3] * x2y2
                    if mo_coeffs[4] != 0:
                        x2yz = (x ** 2) * y * z
                        res += mo_coeffs[4] * x2yz
                    if mo_coeffs[5] != 0:
                        x2z2 = (x ** 2) * (z ** 2)
                        res += mo_coeffs[5] * x2z2
                    if mo_coeffs[6] != 0:
                        xy3 = x * y * y ** 2
                        res += mo_coeffs[6] * xy3
                    if mo_coeffs[7] != 0:
                        xy2z = x * z * y ** 2
                        res += mo_coeffs[7] * xy2z
                    if mo_coeffs[8] != 0:
                        xyz2 = x * y * z ** 2
                        res += mo_coeffs[8] * xyz2
                    if mo_coeffs[9] != 0:
                        xz3 = x * z * z ** 2
                        res += mo_coeffs[9] * xz3
                    if mo_coeffs[10] != 0:
                        y4 = (y ** 2) ** 2
                        res += mo_coeffs[10] * y4
                    if mo_coeffs[11] != 0:
                        y3z = (y ** 2) * y * z
                        res += mo_coeffs[11] * y3z
                    if mo_coeffs[12] != 0:
                        y2z2 = (y * z) ** 2
                        res += mo_coeffs[12] * y2z2
                    if mo_coeffs[13] != 0:
                        yz3 = y * z * z ** 2
                        res += mo_coeffs[13] * yz3
                    if mo_coeffs[14] != 0:
                        z4 = (z ** 2) ** 2
                        res += mo_coeffs[14] * z4

                    return res * s_val
                self.basis_functions.append(g_shell)

            elif shell == -4:
                self.shell_types.append("9g")
                self.n_mos += 9
                self.funcs_per_shell.append(9)
                norms = g_norm(exponents)
                def g_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    if mo_coeffs[0] != 0:
                        z4 = (35 * (z ** 4) - 30 * (r2 * z ** 2) + 3 * r2 ** 2) / 8
                        res += mo_coeffs[0] * z4
                    if mo_coeffs[1] != 0:
                        z3x = np.sqrt(10) * (x * z * (7 * z ** 2 - 3 * r2)) / 4
                        res += mo_coeffs[1] * z3x
                    if mo_coeffs[2] != 0:
                        z3y = np.sqrt(10) * (y * z * (7 * z ** 2 - 3 * r2)) / 4
                        res += mo_coeffs[2] * z3y
                    if mo_coeffs[3] != 0:
                        z2x2y2 = np.sqrt(5) * (x ** 2 - y ** 2) * (7 * z ** 2 - r2) / 4
                        res += mo_coeffs[3] * z2x2y2
                    if mo_coeffs[4] != 0:
                        z2xy = np.sqrt(5) * x * y * (7 * z ** 2 - r2) / 2
                        res += mo_coeffs[4] * z2xy
                    if mo_coeffs[5] != 0:
                        zx3 = np.sqrt(70) * x * z * (x ** 2 - 3 * y ** 2) / 4
                        res += mo_coeffs[5] * zx3
                    if mo_coeffs[6] != 0:
                        zy3 = np.sqrt(70) * z * y * (3 * x ** 2 - y ** 2) / 4
                        res += mo_coeffs[6] * zy3
                    if mo_coeffs[7] != 0:
                        x2 = x ** 2
                        y2 = y ** 2
                        x4y4 = np.sqrt(35) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)) / 8
                        res += mo_coeffs[7] * x4y4
                    if mo_coeffs[8] != 0:
                        xyx2y2 = np.sqrt(35) * x * y * (x ** 2 - y ** 2) / 2
                        res += mo_coeffs[8] * xyx2y2

                    return res * s_val
                self.basis_functions.append(g_shell)

            elif shell == -5:
                self.shell_types.append("11h")
                self.n_mos += 11
                self.funcs_per_shell.append(11)
                norms = h_norm(exponents)
                def h_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    z2 = z ** 2
                    if mo_coeffs[0] != 0:
                        z5z3r2zr4 = z * (63 * z2 ** 2 - 70 * z2 * r2 + 15 * r2 ** 2) / 8
                        res += mo_coeffs[0] * z5z3r2zr4
                    if mo_coeffs[1] != 0:
                        xz4xz2r2xr4 = np.sqrt(15) * x * (21 * z2 ** 2 - 14 * z2 * r2 + r2 ** 2) / 8
                        res += mo_coeffs[1] * xz4xz2r2xr4
                    if mo_coeffs[2] != 0:
                        yz4yz2r2yr4 = np.sqrt(15) * y * (21 * z2 ** 2 - 14 * z2 * r2 + r2 ** 2) / 8
                        res += mo_coeffs[2] * yz4yz2r2yr4
                    if mo_coeffs[3] != 0:
                        x2y3z3zr2 = np.sqrt(105) * (x ** 2 - y ** 2) * (3 * z2 - r2) * z / 4
                        res += mo_coeffs[3] * x2y3z3zr2
                    if mo_coeffs[4] != 0:
                        xyz3zr2 = np.sqrt(105) * x * y * z * (3 * z2 - r2) / 2
                        res += mo_coeffs[4] * xyz3zr2
                    if mo_coeffs[5] != 0:
                        xx2y2z2r2 = 35 * x * (x ** 2 - 3 * y ** 2) * (9 * z2 - r2) / (8 * np.sqrt(70))
                        res += mo_coeffs[5] * xx2y2z2r2
                    if mo_coeffs[6] != 0:
                        yx2y2z2r2 = 35 * y * (3 * x ** 2 - y ** 2) * (9 * z2 - r2) / (8 * np.sqrt(70))
                        res += mo_coeffs[6] * yx2y2z2r2
                    if mo_coeffs[7] != 0:
                        zx4x2y2y4 = 105 * z * ((x ** 2) ** 2 - 6 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(35))
                        res += mo_coeffs[7] * zx4x2y2y4
                    if mo_coeffs[8] != 0:
                        zx3yxy3 = 105 * x * y * z * (4 * x ** 2 - 4 * y ** 2) / (8 * np.sqrt(35))
                        res += mo_coeffs[8] * zx3yxy3
                    if mo_coeffs[9] != 0:
                        xx4y2x2y4 = 21 * x * ((x ** 2) ** 2 - 10 * (x * y) ** 2 + 5 * (y ** 2) ** 2) / (8 * np.sqrt(14))
                        res += mo_coeffs[9] * xx4y2x2y4
                    if mo_coeffs[10] != 0:
                        yx4y2x2y4 = 21 * y * (5 * (x ** 2) ** 2 - 10 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(14))
                        res += mo_coeffs[10] * yx4y2x2y4

                    return res * s_val
                self.basis_functions.append(h_shell)

            elif shell == -6:
                self.shell_types.append("13i")
                self.n_mos += 13
                self.funcs_per_shell.append(13)
                norms = i_norm(exponents)
                def i_shell(
                    r2, x, y, z, mo_coeffs,
                    alpha=exponents, con_coeffs=con_coeff, norms=norms
                ):
                    e_r2 = np.exp(np.outer(-alpha, r2))
                    s_val = np.dot(con_coeff * norms, e_r2)
                    res = np.zeros(len(r2))
                    z2 = z ** 2
                    if mo_coeffs[0] != 0:
                        z6z4r2z2r4r6 = (231 * z2 * z2 ** 2 - 315 * z2 ** 2 * r2 + 105 * z2 * r2 ** 2 - 5 * r2 * r2 ** 2) / 16
                        res += mo_coeffs[0] * z6z4r2z2r4r6
                    if mo_coeffs[1] != 0:
                        xz5z3r2zr4 = np.sqrt(21) * x * z * (33 * z2 ** 2 - 30 * z2 * r2 + 5 * r2 ** 2) / 8
                        res += mo_coeffs[1] * xz5z3r2zr4
                    if mo_coeffs[2] != 0:
                        yz5z3r2zr4 = np.sqrt(21) * y * z * (33 * z2 ** 2 - 30 * z2 * r2 + 5 * r2 ** 2) / 8
                        res += mo_coeffs[2] * yz5z3r2zr4
                    if mo_coeffs[3] != 0:
                        x2y2z4z2r2r3 = 105 * (x ** 2 - y ** 2) * (33 * z2 ** 2 - 18 * z2 * r2 + r2 ** 2) / (16 * np.sqrt(210))
                        res += mo_coeffs[3] * x2y2z4z2r2r3
                    if mo_coeffs[4] != 0:
                        xyz4z2r2r4 = 105 * x * y * (33 * z2 ** 2 - 18 * z2 * r2 + r2 ** 2) / (8 * np.sqrt(210))
                        res += mo_coeffs[4] * xyz4z2r2r4
                    if mo_coeffs[5] != 0:
                        xx2y2z3zr2 = 105 * x * z * (x ** 2 - 3 * y ** 2) * (11 * z2 - 3 * r2) / (8 * np.sqrt(210))
                        res += mo_coeffs[5] * xx2y2z3zr2
                    if mo_coeffs[6] != 0:
                        yx2y2z3zr2 = 105 * y * z * (3 * x ** 2 - y ** 2) * (11 * z2 - 3 * r2) / (8 * np.sqrt(210))
                        res += mo_coeffs[6] * yx2y2z3zr2
                    if mo_coeffs[7] != 0:
                        x4x2y2y4z2r2 = np.sqrt(63) * ((x ** 2) ** 2 - 6 * (x * y) ** 2 + (y ** 2) ** 2) * (11 * z2 - r2) / 16
                        res += mo_coeffs[7] * x4x2y2y4z2r2
                    if mo_coeffs[8] != 0:
                        xyx2y2z2r2 = np.sqrt(63) * x * y * (x ** 2 - y ** 2) * (11 * z2 - r2) / 4
                        res += mo_coeffs[8] * xyx2y2z2r2
                    if mo_coeffs[9] != 0:
                        xzx4x2y2y4 = 231 * x * z * ((x ** 2) ** 2 - 10 * (x * y) ** 2 + 5 * (y ** 2) ** 2) / (8 * np.sqrt(154))
                        res += mo_coeffs[9] * xzx4x2y2y4
                    if mo_coeffs[10] != 0:
                        yzx4x2y2y4 = 231 * y * z * (5 * (x ** 2) ** 2 - 10 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(154))
                        res += mo_coeffs[10] * yzx4x2y2y4
                    if mo_coeffs[11] != 0:
                        x6x4y2x2y4y6 = 231 * ((x * x ** 2) ** 2 - 15 * (x ** 2 * y) ** 2 + 15 * (x * y ** 2) ** 2 - (y * y ** 2) ** 2) / (16 * np.sqrt(462))
                        res += mo_coeffs[11] * x6x4y2x2y4y6
                    if mo_coeffs[12] != 0:
                        yx5x3y3xy5 = 231 * x * y * (6 * (x ** 2) ** 2 - 20 * (x * y) ** 2 + 6 * (y ** 2) ** 2) / (16 * np.sqrt(462))
                        res += mo_coeffs[12] * yx5x3y3xy5

                    return res * s_val
                self.basis_functions.append(i_shell)

            else:
                self.LOG.warning("cannot parse shell with type %i" % shell)

            shell_i += n_prim

        self.alpha_coefficients = np.reshape(
            filereader.other["Alpha MO coefficients"],
            (self.n_mos, self.n_mos),
        )
        if "Beta MO coefficients" in filereader.other:
            self.beta_coefficients = np.reshape(
                filereader.other["Beta MO coefficients"],
                (self.n_mos, self.n_mos),
            )
        else:
            self.beta_coefficients = None
        self.n_alpha = filereader.other["Number of alpha electrons"]
        if "Number of beta electrons" in filereader.other:
            self.n_beta = filereader.other["Number of beta electrons"]

    def _load_nbo_data(self, filereader):
        self.alpha_occupancies = None
        self.beta_occupancies = None
        self.basis_functions = []
        self.exponents = np.array(filereader.other["exponents"])
        self.alpha_coefficients = np.array(filereader.other["alpha_coefficients"])
        self.beta_coefficients = None
        self.shell_coords = []
        self.funcs_per_shell = []
        self.shell_types = []
        self.n_shell = len(filereader.other["n_prim_per_shell"])
        self.alpha_nrgs = [0 for x in self.alpha_coefficients]
        self.n_mos = len(self.alpha_coefficients)
        self.n_alpha = 0
        self.n_beta = 0
        self.beta_nrgs = None

        label_i = 0
        # NBO includes normalization constant with the contraction coefficient
        # so we don't have a gau_norm function like gaussian or orca
        for n_prim, n_funcs, shell_i in zip(
            filereader.other["n_prim_per_shell"],
            filereader.other["funcs_per_shell"],
            filereader.other["start_ndx"],
            
        ):
            shell_i -= 1
            exponents = self.exponents[shell_i: shell_i + n_prim]
            shell_funcs = []
            con_coeffs = []
            shell_type = []
            self.funcs_per_shell.append(n_funcs)
            self.shell_coords.append(
                filereader.atoms[filereader.other["shell_to_atom"][label_i] - 1].coords
            )
            for i in range(0, n_funcs):
                shell = filereader.other["momentum_label"][label_i]
                label_i += 1
                # XXX: each function is treated as a different
                # shell because NBO allows them to be in any order
                # I think that technically means the functions in
                # the d shell for example don't need to be next
                # to each other
                if shell < 100:
                    shell_type.append("s")
                    # s - shell can be 1 or 51
                    con_coeff = filereader.other["s_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    def s_shell(
                        r2, x, y, z, s_val
                    ):
                        return s_val
                    shell_funcs.append(s_shell)
                elif shell < 200:
                    # p - shell can be 101, 102, 103, 151, 152, 153
                    con_coeff = filereader.other["p_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 101 or shell == 151:
                        shell_type.append("px")
                        def px_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x
                        shell_funcs.append(px_shell)
                    elif shell == 102 or shell == 152:
                        shell_type.append("py")
                        def py_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y
                        shell_funcs.append(py_shell)
                    elif shell == 103 or shell == 153:
                        shell_type.append("pz")
                        def pz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z
                        shell_funcs.append(pz_shell)
                elif shell < 300:
                    con_coeff = filereader.other["d_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 201:
                        shell_type.append("dxx")
                        def dxx_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x
                        shell_funcs.append(dxx_shell)
                    elif shell == 202:
                        shell_type.append("dxy")
                        def dxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y
                        shell_funcs.append(dxy_shell)
                    elif shell == 203:
                        shell_type.append("dxz")
                        def dxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * z
                        shell_funcs.append(dxz_shell)
                    elif shell == 204:
                        shell_type.append("dyy")
                        def dyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y
                        shell_funcs.append(dyy_shell)
                    elif shell == 205:
                        shell_type.append("dyz")
                        def dyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * z
                        shell_funcs.append(dyz_shell)
                    elif shell == 206:
                        shell_type.append("dzz")
                        def dzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * z
                        shell_funcs.append(dzz_shell)
                    elif shell == 251:
                        shell_type.append("5dxy")
                        def dxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * x * y
                        shell_funcs.append(dxy_shell)
                    elif shell == 252:
                        shell_type.append("5dxz")
                        def dxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * x * z
                        shell_funcs.append(dxz_shell)
                    elif shell == 253:
                        shell_type.append("5dyz")
                        def dyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * y * z
                        shell_funcs.append(dyz_shell)
                    elif shell == 254:
                        shell_type.append("5dx2-y2")
                        def dx2y2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * (x ** 2 - y ** 2) / 2
                        shell_funcs.append(dx2y2_shell)
                    elif shell == 255:
                        shell_type.append("5dz2")
                        def dz2_shell(
                            r2, x, y, z, s_val
                        ):
                            return (3 * z ** 2 - r2) * s_val / 2
                        shell_funcs.append(dz2_shell)
                elif shell < 400:
                    con_coeff = filereader.other["f_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 301:
                        shell_type.append("fxxx")
                        def fxxx_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x
                        shell_funcs.append(fxxx_shell)
                    if shell == 302:
                        shell_type.append("fxxy")
                        def fxxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * y
                        shell_funcs.append(fxxy_shell)
                    if shell == 303:
                        shell_type.append("fxxz")
                        def fxxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * z
                        shell_funcs.append(fxxz_shell)
                    if shell == 304:
                        shell_type.append("fxyy")
                        def fxyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * y
                        shell_funcs.append(fxyy_shell)
                    if shell == 305:
                        shell_type.append("fxyz")
                        def fxyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * z
                        shell_funcs.append(fxyz_shell)
                    if shell == 306:
                        shell_type.append("fxzz")
                        def fxzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * z * z
                        shell_funcs.append(fxzz_shell)
                    if shell == 307:
                        shell_type.append("fyyy")
                        def fyyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * y
                        shell_funcs.append(fyyy_shell)
                    if shell == 308:
                        shell_type.append("fyyz")
                        def fyyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * z
                        shell_funcs.append(fyyz_shell)
                    if shell == 309:
                        shell_type.append("fyzz")
                        def fyzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * z * z
                        shell_funcs.append(fyzz_shell)
                    if shell == 310:
                        shell_type.append("fzzz")
                        def fzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * z * z
                        shell_funcs.append(fzzz_shell)
                    if shell == 351:
                        shell_type.append("7fz3-zr2")
                        def fz3zr2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * (5 * z ** 2 - 3 * r2) / 2
                        shell_funcs.append(fz3zr2_shell)
                    if shell == 352:
                        shell_type.append("7fxz2-xr2")
                        def fxz2xr2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(3) * s_val * x * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        shell_funcs.append(fxz2xr2_shell)
                    if shell == 353:
                        shell_type.append("7fyz2-yr2")
                        def fyz2yr2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * (5 * z ** 2 - r2) / (2 * np.sqrt(2))
                        shell_funcs.append(fyz2yr2_shell)
                    if shell == 354:
                        shell_type.append("7fzx2-zy2")
                        def fzx2zy2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(15) * s_val * z * (x ** 2 - y ** 2) / 2
                        shell_funcs.append(fzx2zy2_shell)
                    if shell == 355:
                        shell_type.append("7fxyz")
                        def fxyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(15) * s_val * x * y * z
                        shell_funcs.append(fxyz_shell)
                    if shell == 356:
                        shell_type.append("7fx3-xy2")
                        def fx3xy2_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(5) * s_val * x * (x ** 2 - 3 * y ** 2) / (2 * np.sqrt(2))
                        shell_funcs.append(fx3xy2_shell)
                    if shell == 357:
                        shell_type.append("7fyx2-y3")
                        def fyx2y3_shell(
                            r2, x, y, z, s_val
                        ):
                            return np.sqrt(5) * s_val * y * (3 * x ** 2 - y ** 2) / (2 * np.sqrt(2))
                        shell_funcs.append(fyx2y3_shell)
                elif shell < 500:
                    # I can't tell what NBO does with g orbitals
                    # I don't have any reference to compare to
                    self.LOG.warning(
                        "g shell results have not been verified for NBO\n"
                        "any LCAO's may be invalid"
                    )
                    con_coeff = filereader.other["g_coeff"][shell_i: shell_i + n_prim]
                    con_coeffs.append(con_coeff)
                    if shell == 401:
                        shell_type.append("gxxxx")
                        def gxxxx_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x * x
                        shell_funcs.append(gxxxx_shell)
                    if shell == 402:
                        shell_type.append("gxxxy")
                        def gxxxy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x * y
                        shell_funcs.append(gxxxy_shell)
                    if shell == 403:
                        shell_type.append("gxxxz")
                        def gxxxz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * x * z
                        shell_funcs.append(gxxxz_shell)
                    if shell == 404:
                        shell_type.append("gxxyy")
                        def gxxyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * y * y
                        shell_funcs.append(gxxyy_shell)
                    if shell == 405:
                        shell_type.append("gxxyz")
                        def gxxyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * y * z
                        shell_funcs.append(gxxyz_shell)
                    if shell == 406:
                        shell_type.append("gxxzz")
                        def gxxzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * x * z * z
                        shell_funcs.append(gxxzz_shell)
                    if shell == 407:
                        shell_type.append("gxyyy")
                        def gxyyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * y * y
                        shell_funcs.append(gxyyy_shell)
                    if shell == 408:
                        shell_type.append("gxyyz")
                        def gxyyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * y * z
                        shell_funcs.append(gxyyz_shell)
                    if shell == 409:
                        shell_type.append("gxyzz")
                        def gxyzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * y * z * z
                        shell_funcs.append(gxyzz_shell)
                    if shell == 410:
                        shell_type.append("gxzzz")
                        def gxzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * x * z * z * z
                        shell_funcs.append(gxzzz_shell)
                    if shell == 411:
                        shell_type.append("gyyyy")
                        def gyyyy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * y * y
                        shell_funcs.append(gyyyy_shell)
                    if shell == 412:
                        shell_type.append("gyyyz")
                        def gyyyz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * y * z
                        shell_funcs.append(gyyyz_shell)
                    if shell == 413:
                        shell_type.append("gyyzz")
                        def gyyzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * y * z * z
                        shell_funcs.append(gyyzz_shell)
                    if shell == 414:
                        shell_type.append("gyzzz")
                        def gyzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * y * z * z * z
                        shell_funcs.append(gyzzz_shell)
                    if shell == 415:
                        shell_type.append("gzzzz")
                        def gzzzz_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * z * z * z * z
                        shell_funcs.append(gzzzz_shell)
                    if shell == 451:
                        shell_type.append("9gz4")
                        def gz4_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * (35 * (z ** 2) ** 2 - 30 * z ** 2 * r2 + 3 * r2 ** 2) / 8
                        shell_funcs.append(gz4_shell)
                    if shell == 452:
                        shell_type.append("9gz3x")
                        def gz3x_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(10) * (x * z * (7 * z ** 2 - 3 * r2)) / 4
                        shell_funcs.append(gz3x_shell)
                    if shell == 453:
                        shell_type.append("9gz3y")
                        def gz3y_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(10) * (y * z * (7 * z ** 2 - 3 * r2)) / 4
                        shell_funcs.append(gz3y_shell)
                    if shell == 454:
                        shell_type.append("9gz2x2-z2y2")
                        def gz2x2y2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(5) * (x ** 2 - y ** 2) * (7 * z ** 2 - r2) / 4
                        shell_funcs.append(gz2x2y2_shell)
                    if shell == 455:
                        shell_type.append("9gz2xy")
                        def gz2xy_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(5) * x * y * (7 * z ** 2 - r2) / 2
                        shell_funcs.append(gz2xy_shell)
                    if shell == 456:
                        shell_type.append("9gzx3")
                        def gzx3_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(70) * x * z * (x ** 2 - 3 * y ** 2) / 4
                        shell_funcs.append(gzx3_shell)
                    if shell == 457:
                        shell_type.append("9gzy3")
                        def gzy3_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(70) * z * y * (3 * x ** 2 - y ** 2) / 4
                        shell_funcs.append(gzy3_shell)
                    if shell == 458:
                        shell_type.append("9gx4y4")
                        def gx4y4_shell(
                            r2, x, y, z, s_val
                        ):
                            x2 = x ** 2
                            y2 = y ** 2
                            return s_val * np.sqrt(35) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)) / 8
                        shell_funcs.append(gx4y4_shell)
                    if shell == 459:
                        shell_type.append("9gxyx2y2")
                        def gxyx2y2_shell(
                            r2, x, y, z, s_val
                        ):
                            return s_val * np.sqrt(35) * x * y * (x ** 2 - y ** 2) / 2
                        shell_funcs.append(gxyx2y2_shell)
                else:
                    self.LOG.warning("cannot handle shells with momentum label %i" % shell)
        
            def eval_shells(
                r2, x, y, z, mo_coeffs,
                alpha=exponents,
                con_coeffs=con_coeffs,
                shell_funcs=shell_funcs
            ):
                e_r2 =  np.exp(np.outer(-alpha, r2))
                res = np.zeros(len(r2))
                last_con_coeff = None
                for mo_coeff, con_coeff, func in zip(mo_coeffs, con_coeffs, shell_funcs):
                    if mo_coeff == 0:
                        continue
                    if last_con_coeff is None or any(
                        x - y != 0 for x, y in zip(last_con_coeff, con_coeff)
                    ):
                        s_val = np.dot(con_coeff, e_r2)
                    last_con_coeff = con_coeff
                    res += mo_coeff * func(r2, x, y, z, s_val)
                return res
            self.basis_functions.append(eval_shells)
            self.shell_types.append(", ".join(shell_type))
        
        self.shell_coords = np.array(self.shell_coords)

    def _load_orca_out_data(self, filereader):
        self.alpha_occupancies = None
        self.beta_occupancies = None
        self.shell_coords = []
        self.basis_functions = []
        self.alpha_nrgs = np.array(filereader.other["alpha_nrgs"])
        self.alpha_coefficients = np.array(filereader.other["alpha_coefficients"])
        if "beta_nrgs" not in filereader.other or not filereader.other["beta_nrgs"]:
            self.beta_nrgs = None
            self.beta_coefficients = None
        else:
            self.beta_nrgs = np.array(filereader.other["beta_nrgs"])
            self.beta_coefficients = np.array(filereader.other["beta_coefficients"])
        self.shell_types = []
        self.funcs_per_shell = []
        self.n_aos = 0
        self.n_mos = 0

        def gau_norm(a, l):
            """
            normalization for gaussian primitives that depends on
            the exponential (a) and the total angular momentum (l)
            """
            t1 = np.sqrt((2 * a) ** (l + 3 / 2)) / (np.pi ** (3.0 / 4))
            t2 = np.sqrt(2 ** l / factorial2(2 * l - 1))
            return t1 * t2

        # get functions for norm of s, p, 5d, and 7f
        s_norm = lambda a, l=0: gau_norm(a, l)
        p_norm = lambda a, l=1: gau_norm(a, l)
        d_norm = lambda a, l=2: gau_norm(a, l)
        f_norm = lambda a, l=3: gau_norm(a, l)
        g_norm = lambda a, l=4: gau_norm(a, l)
        h_norm = lambda a, l=5: gau_norm(a, l)
        i_norm = lambda a, l=6: gau_norm(a, l)

        # ORCA order differs from FCHK in a few places:
        # pz, px, py instead of px, py, pz
        # f(3xy^2 - x^3) instead of f(x^3 - 3xy^2)
        # f(y^3 - 3x^2y) instead of f(3x^2y - y^3)
        # ORCA doesn't seem to print the coordinates of each
        # shell, but they should be the same as the atom coordinates
        for atom in filereader.atoms:
            ele = atom.element
            if ele not in filereader.other["basis_set_by_ele"]:
                continue
            for shell_type, n_prim, exponents, con_coeff in filereader.other[
                "basis_set_by_ele"
            ][ele]:
                self.shell_coords.append(atom.coords)
                exponents = np.array(exponents)
                con_coeff = np.array(con_coeff)
                if shell_type.lower() == "s":
                    self.shell_types.append("s")
                    self.funcs_per_shell.append(1)
                    self.n_aos += 1
                    norms = s_norm(exponents)
                    if n_prim > 1:
                        def s_shell(
                            r2, x, y, z, mo_coeff,
                            alpha=exponents,
                            con_coeff=con_coeff,
                            norms=norms
                        ):
                            e_r2 = np.exp(np.outer(-alpha, r2))
                            return mo_coeff[0] * np.dot(con_coeff * norms, e_r2)
                    else:
                        def s_shell(
                            r2, x, y, z, mo_coeff,
                            alpha=exponents,
                            con_coeff=con_coeff,
                            norms=norms
                        ):
                            e_r2 = np.exp(-alpha * r2)
                            return mo_coeff * con_coeff * norms * e_r2
                    self.basis_functions.append(s_shell)
                elif shell_type.lower() == "p":
                    self.shell_types.append("p")
                    self.funcs_per_shell.append(3)
                    self.n_aos += 3
                    norms = p_norm(exponents)

                    def p_shell(
                        r2,
                        x,
                        y,
                        z,
                        mo_coeffs,
                        alpha=exponents,
                        con_coeff=con_coeff,
                        norms=norms,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)

                        if isinstance(r2, float):
                            res = 0
                        else:
                            res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            res += mo_coeffs[0] * z
                        if mo_coeffs[1] != 0:
                            res += mo_coeffs[1] * x
                        if mo_coeffs[2] != 0:
                            res += mo_coeffs[2] * y

                        return res * s_val

                    self.basis_functions.append(p_shell)
                elif shell_type.lower() == "d":
                    self.shell_types.append("5d")
                    self.funcs_per_shell.append(5)
                    self.n_aos += 5
                    norms = d_norm(exponents)

                    def d_shell(
                        r2,
                        x,
                        y,
                        z,
                        mo_coeffs,
                        alpha=exponents,
                        con_coeff=con_coeff,
                        norms=norms,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        if isinstance(r2, float):
                            res = 0
                        else:
                            res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z2r2 = 0.5 * (3 * z * z - r2)
                            res += mo_coeffs[0] * z2r2
                        if mo_coeffs[1] != 0:
                            xz = np.sqrt(3) * x * z
                            res += mo_coeffs[1] * xz
                        if mo_coeffs[2] != 0:
                            yz = np.sqrt(3) * y * z
                            res += mo_coeffs[2] * yz
                        if mo_coeffs[3] != 0:
                            x2y2 = np.sqrt(3) * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2y2
                        if mo_coeffs[4] != 0:
                            xy = np.sqrt(3) * x * y
                            res += mo_coeffs[4] * xy
                        return res * s_val

                    self.basis_functions.append(d_shell)
                elif shell_type.lower() == "f":
                    self.shell_types.append("7f")
                    self.funcs_per_shell.append(7)
                    self.n_aos += 7
                    norms = f_norm(exponents)

                    def f_shell(
                        r2,
                        x,
                        y,
                        z,
                        mo_coeffs,
                        alpha=exponents,
                        con_coeff=con_coeff,
                        norms=norms,
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeff * norms, e_r2)
                        if isinstance(r2, float):
                            res = 0
                        else:
                            res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z3zr2 = z * (5 * z ** 2 - 3 * r2) / 2
                            res += mo_coeffs[0] * z3zr2
                        if mo_coeffs[1] != 0:
                            xz2xr2 = (
                                np.sqrt(3)
                                * x
                                * (5 * z ** 2 - r2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[1] * xz2xr2
                        if mo_coeffs[2] != 0:
                            yz2yr2 = (
                                np.sqrt(3)
                                * y
                                * (5 * z ** 2 - r2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[2] * yz2yr2
                        if mo_coeffs[3] != 0:
                            x2zr2z = np.sqrt(15) * z * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[3] * x2zr2z
                        if mo_coeffs[4] != 0:
                            xyz = np.sqrt(15) * x * y * z
                            res += mo_coeffs[4] * xyz
                        if mo_coeffs[5] != 0:
                            x3r2x = (
                                np.sqrt(5)
                                * x
                                * (3 * y ** 2 - x ** 2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[5] * x3r2x
                        if mo_coeffs[6] != 0:
                            x2yy3 = (
                                np.sqrt(5)
                                * y
                                * (y ** 2 - 3 * x ** 2)
                                / (2 * np.sqrt(2))
                            )
                            res += mo_coeffs[6] * x2yy3
                        return res * s_val

                    self.basis_functions.append(f_shell)
                
                elif shell_type.lower() == "g":
                    self.shell_types.append("9g")
                    self.funcs_per_shell.append(9)
                    self.n_aos += 9
                    norms = g_norm(exponents)
                    def g_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeffs=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeffs * norms, e_r2)
                        res = np.zeros(len(r2))
                        if mo_coeffs[0] != 0:
                            z4 = (35 * (z ** 4) - 30 * (r2 * z ** 2) + 3 * r2 ** 2) / 8
                            res += mo_coeffs[0] * z4
                        if mo_coeffs[1] != 0:
                            z3x = np.sqrt(10) * (x * z * (7 * z ** 2 - 3 * r2)) / 4
                            res += mo_coeffs[1] * z3x
                        if mo_coeffs[2] != 0:
                            z3y = np.sqrt(10) * (y * z * (7 * z ** 2 - 3 * r2)) / 4
                            res += mo_coeffs[2] * z3y
                        if mo_coeffs[3] != 0:
                            z2x2y2 = np.sqrt(5) * (x ** 2 - y ** 2) * (7 * z ** 2 - r2) / 4
                            res += mo_coeffs[3] * z2x2y2
                        if mo_coeffs[4] != 0:
                            z2xy = np.sqrt(5) * x * y * (7 * z ** 2 - r2) / 2
                            res += mo_coeffs[4] * z2xy
                        if mo_coeffs[5] != 0:
                            zx3 = -np.sqrt(70) * x * z * (x ** 2 - 3 * y ** 2) / 4
                            res += mo_coeffs[5] * zx3
                        if mo_coeffs[6] != 0:
                            zy3 = -np.sqrt(70) * z * y * (3 * x ** 2 - y ** 2) / 4
                            res += mo_coeffs[6] * zy3
                        if mo_coeffs[7] != 0:
                            x2 = x ** 2
                            y2 = y ** 2
                            x4y4 = -np.sqrt(35) * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2)) / 8
                            res += mo_coeffs[7] * x4y4
                        if mo_coeffs[8] != 0:
                            xyx2y2 = -np.sqrt(35) * x * y * (x ** 2 - y ** 2) / 2
                            res += mo_coeffs[8] * xyx2y2

                        return res * s_val
                    self.basis_functions.append(g_shell)

                elif shell_type.lower() == "h":
                    self.shell_types.append("11h")
                    self.funcs_per_shell.append(11)
                    self.n_aos += 11
                    norms = h_norm(exponents)
                    def h_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeffs=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeffs * norms, e_r2)
                        res = np.zeros(len(r2))
                        z2 = z ** 2
                        if mo_coeffs[0] != 0:
                            z5z3r2zr4 = z * (63 * z2 ** 2 - 70 * z2 * r2 + 15 * r2 ** 2) / 8
                            res += mo_coeffs[0] * z5z3r2zr4
                        if mo_coeffs[1] != 0:
                            xz4xz2r2xr4 = np.sqrt(15) * x * (21 * z2 ** 2 - 14 * z2 * r2 + r2 ** 2) / 8
                            res += mo_coeffs[1] * xz4xz2r2xr4
                        if mo_coeffs[2] != 0:
                            yz4yz2r2yr4 = np.sqrt(15) * y * (21 * z2 ** 2 - 14 * z2 * r2 + r2 ** 2) / 8
                            res += mo_coeffs[2] * yz4yz2r2yr4
                        if mo_coeffs[3] != 0:
                            x2y3z3zr2 = np.sqrt(105) * (x ** 2 - y ** 2) * (3 * z2 - r2) * z / 4
                            res += mo_coeffs[3] * x2y3z3zr2
                        if mo_coeffs[4] != 0:
                            xyz3zr2 = np.sqrt(105) * x * y * z * (3 * z2 - r2) / 2
                            res += mo_coeffs[4] * xyz3zr2
                        if mo_coeffs[5] != 0:
                            xx2y2z2r2 = -35 * x * (x ** 2 - 3 * y ** 2) * (9 * z2 - r2) / (8 * np.sqrt(70))
                            res += mo_coeffs[5] * xx2y2z2r2
                        if mo_coeffs[6] != 0:
                            yx2y2z2r2 = -35 * y * (3 * x ** 2 - y ** 2) * (9 * z2 - r2) / (8 * np.sqrt(70))
                            res += mo_coeffs[6] * yx2y2z2r2
                        if mo_coeffs[7] != 0:
                            zx4x2y2y4 = -105 * z * ((x ** 2) ** 2 - 6 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(35))
                            res += mo_coeffs[7] * zx4x2y2y4
                        if mo_coeffs[8] != 0:
                            zx3yxy3 = -105 * x * y * z * (4 * x ** 2 - 4 * y ** 2) / (8 * np.sqrt(35))
                            res += mo_coeffs[8] * zx3yxy3
                        if mo_coeffs[9] != 0:
                            xx4y2x2y4 = 21 * x * ((x ** 2) ** 2 - 10 * (x * y) ** 2 + 5 * (y ** 2) ** 2) / (8 * np.sqrt(14))
                            res += mo_coeffs[9] * xx4y2x2y4
                        if mo_coeffs[10] != 0:
                            yx4y2x2y4 = 21 * y * (5 * (x ** 2) ** 2 - 10 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(14))
                            res += mo_coeffs[10] * yx4y2x2y4
    
                        return res * s_val
                    self.basis_functions.append(h_shell)

                elif shell_type.lower() == "i":
                    self.shell_types.append("13i")
                    self.funcs_per_shell.append(13)
                    self.n_aos += 13
                    norms = i_norm(exponents)
                    def i_shell(
                        r2, x, y, z, mo_coeffs,
                        alpha=exponents, con_coeffs=con_coeff, norms=norms
                    ):
                        e_r2 = np.exp(np.outer(-alpha, r2))
                        s_val = np.dot(con_coeffs * norms, e_r2)
                        res = np.zeros(len(r2))
                        z2 = z ** 2
                        if mo_coeffs[0] != 0:
                            z6z4r2z2r4r6 = (231 * z2 * z2 ** 2 - 315 * z2 ** 2 * r2 + 105 * z2 * r2 ** 2 - 5 * r2 * r2 ** 2) / 16
                            res += mo_coeffs[0] * z6z4r2z2r4r6
                        if mo_coeffs[1] != 0:
                            xz5z3r2zr4 = np.sqrt(21) * x * z * (33 * z2 ** 2 - 30 * z2 * r2 + 5 * r2 ** 2) / 8
                            res += mo_coeffs[1] * xz5z3r2zr4
                        if mo_coeffs[2] != 0:
                            yz5z3r2zr4 = np.sqrt(21) * y * z * (33 * z2 ** 2 - 30 * z2 * r2 + 5 * r2 ** 2) / 8
                            res += mo_coeffs[2] * yz5z3r2zr4
                        if mo_coeffs[3] != 0:
                            x2y2z4z2r2r3 = 105 * (x ** 2 - y ** 2) * (33 * z2 ** 2 - 18 * z2 * r2 + r2 ** 2) / (16 * np.sqrt(210))
                            res += mo_coeffs[3] * x2y2z4z2r2r3
                        if mo_coeffs[4] != 0:
                            xyz4z2r2r4 = 105 * x * y * (33 * z2 ** 2 - 18 * z2 * r2 + r2 ** 2) / (8 * np.sqrt(210))
                            res += mo_coeffs[4] * xyz4z2r2r4
                        if mo_coeffs[5] != 0:
                            xx2y2z3zr2 = -105 * x * z * (x ** 2 - 3 * y ** 2) * (11 * z2 - 3 * r2) / (8 * np.sqrt(210))
                            res += mo_coeffs[5] * xx2y2z3zr2
                        if mo_coeffs[6] != 0:
                            yx2y2z3zr2 = -105 * y * z * (3 * x ** 2 - y ** 2) * (11 * z2 - 3 * r2) / (8 * np.sqrt(210))
                            res += mo_coeffs[6] * yx2y2z3zr2
                        if mo_coeffs[7] != 0:
                            x4x2y2y4z2r2 = -np.sqrt(63) * ((x ** 2) ** 2 - 6 * (x * y) ** 2 + (y ** 2) ** 2) * (11 * z2 - r2) / 16
                            res += mo_coeffs[7] * x4x2y2y4z2r2
                        if mo_coeffs[8] != 0:
                            xyx2y2z2r2 = -np.sqrt(63) * x * y * (x ** 2 - y ** 2) * (11 * z2 - r2) / 4
                            res += mo_coeffs[8] * xyx2y2z2r2
                        if mo_coeffs[9] != 0:
                            xzx4x2y2y4 = 231 * x * z * ((x ** 2) ** 2 - 10 * (x * y) ** 2 + 5 * (y ** 2) ** 2) / (8 * np.sqrt(154))
                            res += mo_coeffs[9] * xzx4x2y2y4
                        if mo_coeffs[10] != 0:
                            yzx4x2y2y4 = 231 * y * z * (5 * (x ** 2) ** 2 - 10 * (x * y) ** 2 + (y ** 2) ** 2) / (8 * np.sqrt(154))
                            res += mo_coeffs[10] * yzx4x2y2y4
                        if mo_coeffs[11] != 0:
                            x6x4y2x2y4y6 = 231 * ((x * x ** 2) ** 2 - 15 * (x ** 2 * y) ** 2 + 15 * (x * y ** 2) ** 2 - (y * y ** 2) ** 2) / (16 * np.sqrt(462))
                            res += mo_coeffs[11] * x6x4y2x2y4y6
                        if mo_coeffs[12] != 0:
                            yx5x3y3xy5 = 231 * x * y * (6 * (x ** 2) ** 2 - 20 * (x * y) ** 2 + 6 * (y ** 2) ** 2) / (16 * np.sqrt(462))
                            res += mo_coeffs[12] * yx5x3y3xy5
    
                        return res * s_val
                    self.basis_functions.append(i_shell)

                else:
                    self.LOG.warning(
                        "cannot handle shell of type %s" % shell_type
                    )

        self.n_mos = len(self.alpha_coefficients)

        if "n_alpha" not in filereader.other:
            tot_electrons = sum(
                ELEMENTS.index(atom.element) for atom in filereader.atoms
            )
            self.n_beta = tot_electrons // 2
            self.n_alpha = tot_electrons - self.n_beta
        else:
            self.n_alpha = filereader.other["n_alpha"]
            self.n_beta = filereader.other["n_beta"]

        if "alpha_occupancies" in filereader.other:
            self.alpha_occupancies = filereader.other["alpha_occupancies"]

        if "beta_occupancies" in filereader.other and filereader.other["beta_occupancies"]:
            self.beta_occupancies = filereader.other["beta_occupancies"]
        elif "alpha_occupancies" in filereader.other:
            self.alpha_occupancies = [occ / 2 for occ in self.alpha_occupancies]

    def _get_value(self, coords, arr):
        """returns value for the MO coefficients in arr"""
        ao = 0
        prev_center = None
        if coords.ndim == 1:
            val = 0
        else:
            val = np.zeros(len(coords))
        
        for coord, shell, n_func, shell_type in zip(
            self.shell_coords,
            self.basis_functions,
            self.funcs_per_shell,
            self.shell_types,
        ):
            # don't calculate distances until we find an AO
            # in this shell that has a non-zero MO coefficient
            if not np.count_nonzero(arr[ao : ao + n_func]):
                ao += n_func
                continue
            # print(shell_type, arr[ao : ao + n_func])
            # don't recalculate distances unless this shell's coordinates
            # differ from the previous
            if (
                prev_center is None
                or np.linalg.norm(coord - prev_center) > 1e-13
            ):
                prev_center = coord
                d_coord = (coords - coord) / UNIT.A0_TO_BOHR
                if coords.ndim == 1:
                    r2 = np.dot(d_coord, d_coord)
                else:
                    r2 = np.sum(d_coord * d_coord, axis=1)
            if coords.ndim == 1:
                res = shell(
                    r2,
                    d_coord[0],
                    d_coord[1],
                    d_coord[2],
                    arr[ao : ao + n_func],
                )
            else:
                res = shell(
                    r2,
                    d_coord[:, 0],
                    d_coord[:, 1],
                    d_coord[:, 2],
                    arr[ao : ao + n_func],
                )
            val += res
            ao += n_func
        return val

    def mo_value(self, mo, coords, alpha=True, n_jobs=1):
        """
        get the MO evaluated at the specified coords
        m - index of molecular orbital or an array of MO coefficients
        coords - numpy array of points (N,3) or (3,)
        alpha - use alpha coefficients (default)
        n_jobs - number of parallel threads to use
                 this is on top of NumPy's multithreading, so
                 if NumPy uses 8 threads and n_jobs=2, you can
                 expect to see 16 threads in use
        """
        # val is the running sum of MO values
        if alpha:
            coeff = self.alpha_coefficients
        else:
            coeff = self.beta_coefficients

        if isinstance(mo, int):
            coeff = coeff[mo]
        else:
            coeff = mo

        # calculate AO values for each shell at each point
        # multiply by the MO coefficient and add to val
        if n_jobs > 1:
            # get all shells grouped by coordinates
            # this reduces the number of times we will need to
            # calculate the distance from all the coords to
            # a shell's center
            prev_coords = []
            arrays = []
            ndx = 0
            add_to = 0
            
                
            for i, coord in enumerate(self.shell_coords):
                for j, prev_coord in enumerate(prev_coords):
                    if np.linalg.norm(coord - prev_coord) < 1e-13:
                        add_to = j
                        break
                else:
                    prev_coords.append(coord)
                    add_to = len(arrays)
                    arrays.append(np.zeros(self.n_mos))
                arrays[add_to][ndx : ndx + self.funcs_per_shell[i]] = coeff[
                    ndx : ndx + self.funcs_per_shell[i]
                ]
                ndx += self.funcs_per_shell[i]
            
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                out = [executor.submit(self._get_value, coords, arr) for arr in arrays]
            return sum([shells.result() for shells in out])
        val = self._get_value(coords, coeff)
        return val

    def density_value(
        self,
        coords,
        n_jobs=1,
        alpha_occ=None,
        beta_occ=None,
        low_mem=False,
        spin=True
    ):
        """
        returns the eletron density
        coords - coordinates to calculate e density at
        n_jobs - number of concurrent threads to use in calculation
        alpha_occ - array of alpha occupancies
                    if not specified, defaults to lowest self.n_alpha
                    orbitals
        beta_occ - same at alpha_occ, but for beta electrons
        spin - plot spin density
        """

        print(spin)

        # set default occupancy
        if alpha_occ is None:
            if self.alpha_occupancies:
                alpha_occ = np.array(self.alpha_occupancies)
            else:
                if not self.n_alpha:
                    self.LOG.warning("number of alpha electrons was not read")
                alpha_occ = np.zeros(self.n_mos, dtype=int)
                alpha_occ[0:self.n_alpha] = 1
        
        if beta_occ is None:
            if self.beta_occupancies:
                beta_occ = np.array(self.beta_occupancies)
            else:
                if not self.n_beta:
                    self.LOG.warning("number of beta electrons was not read")
                beta_occ = np.zeros(self.n_mos, dtype=int)
                beta_occ[0:self.n_beta] = 1

        if spin:
            beta_occ = -1 * beta_occ

        if low_mem:
            return self._low_mem_density_value(
                coords,
                alpha_occ,
                beta_occ,
                n_jobs=n_jobs,
                spin=spin,
            )
        
        # val is output data
        # func_vals is the value of each basis function
        # at all coordinates
        if coords.ndim == 1:
            val = 0
            func_vals = np.zeros(
                len(self.basis_functions), dtype="float32"
            )
        else:
            val = np.zeros(len(coords))
            if spin and self.beta_coefficients is None:
                return val
            func_vals = np.zeros(
                (self.n_mos, *coords.shape,)
            )

        # get values of basis functions at all points
        arrays = np.eye(self.n_mos)
        if n_jobs > 1:
            # get all shells grouped by coordinates
            # this reduces the number of times we will need to
            # calculate the distance from all the coords to
            # a shell's center
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                out = [
                    executor.submit(self._get_value, coords, arr) for arr in arrays
                ]
            data = np.array([shells.result() for shells in out])

        else:
            data = np.array([
                self._get_value(coords, arr) for arr in arrays
            ])

        # multiply values by orbital coefficients and square
        for i, occ in enumerate(alpha_occ):
            if occ == 0:
                continue
            val += occ * np.dot(data.T, self.alpha_coefficients[i]) ** 2

        if self.beta_coefficients is not None:
            for i, occ in enumerate(beta_occ):
                if occ == 0:
                    continue
                val += occ * np.dot(data.T, self.beta_coefficients[i]) ** 2
        else:
            val *= 2

        return val

    def _low_mem_density_value(
        self,
        coords,
        alpha_occ,
        beta_occ,
        n_jobs=1,
        spin=False,
    ):
        """
        returns the eletron density
        same at self.density_value, but uses less memory at
        the cost of performance
        """

        # val is output array
        if coords.ndim == 1:
            val = 0
        else:
            val = np.zeros(len(coords), dtype="float32")

        if spin and self.beta_coefficients is None:
            return val

        # calculate each occupied orbital
        # square it can add to output
        if n_jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:
                out = [
                    executor.submit(self.mo_value, i, coords, n_jobs=1)
                    for i, occ in enumerate(alpha_occ) if occ != 0
                ]
            val += sum([occ * orbit.result() ** 2 for orbit, occ in zip(out, alpha_occ)])
            
            if self.beta_coefficients is not None:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_jobs
                ) as executor:
                    out = [
                        executor.submit(self.mo_value, i, coords, alpha=False, n_jobs=1)
                        for i, occ in enumerate(beta_occ) if occ != 0
                    ]
                val += sum([occ * orbit.result() ** 2 for orbit, occ in zip(out, beta_occ)])
            else:
                val *= 2
        
        else:
            for i in range(0, self.n_alpha):
                val += self.mo_value(i, coords) ** 2
            
            if self.beta_coefficients is not None:
                for i in range(0, self.n_beta):
                    val += self.mo_value(i, coords, alpha=False) ** 2
            else:
                val *= 2

        return val

    def fukui_donor_value(self, coords, delta=0.1, **kwargs):
        """
        orbital-weighted fukui donor function
        electron density change for removing an electron
        orbital weighting from DOI 10.1002/jcc.24699 accounts
        for nearly degenerate orbitals
        coords - coordinate to evaluate function at
        delta - parameter for weighting
        kwargs - passed to density_value
        """
        CITATION = "doi:10.1002/jcc.24699"
        self.LOG.citation(CITATION)

        if self.beta_coefficients is None:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            minus_e = np.zeros(self.n_mos)
            for i in range(0, self.n_alpha):
                minus_e[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )

            minus_e /= sum(minus_e)
            minus_density = self.density_value(
                coords, alpha_occ=minus_e, beta_occ=None, **kwargs
            )
        
        else:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            if self.n_beta > self.n_alpha:
                homo_nrg = self.beta_nrgs[self.n_beta - 1]
            
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            if self.n_beta > self.n_alpha:
                lumo_nrg = self.beta_nrgs[self.n_beta]

            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            alpha_occ = beta_occ = np.zeros(self.n_mos)
            for i in range(0, self.n_alpha):
                alpha_occ[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )
            
            for i in range(0, self.n_beta):
                beta_occ[i] = np.exp(
                    -((chem_pot - self.beta_nrgs[i]) / delta) ** 2
                )

            alpha_occ /= sum(alpha_occ)
            beta_occ /= sum(beta_occ)
            minus_density = self.density_value(
                coords, alpha_occ=alpha_occ, beta_occ=beta_occ, **kwargs
            )

        return minus_density
    
    def fukui_acceptor_value(self, coords, delta=0.1, **kwargs):
        """
        orbital-weighted fukui acceptor function
        electron density change for removing an electron
        orbital weighting from DOI 10.1021/acs.jpca.9b07516 accounts
        for nearly degenerate orbitals
        coords - coordinate to evaluate function at
        delta - parameter for weighting
        kwargs - passed to density_value
        """
        CITATION = "doi:10.1021/acs.jpca.9b07516"
        self.LOG.citation(CITATION)
        
        alpha_occ = np.zeros(self.n_mos)
        alpha_occ[self.n_alpha - 1] = 1
        beta_occ = None
        if self.beta_coefficients is not None:
            beta_occ = np.zeros(self.n_mos)
            beta_occ[self.n_beta - 1] = 1

        if self.beta_coefficients is None:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            plus_e = np.zeros(self.n_mos)
            for i in range(self.n_alpha, self.n_mos):
                plus_e[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )

            plus_e /= sum(plus_e)
            plus_density = self.density_value(
                coords, alpha_occ=plus_e, beta_occ=beta_occ, **kwargs
            )
        
        else:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            if self.n_beta > self.n_alpha:
                homo_nrg = self.beta_nrgs[self.n_beta - 1]
            
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            if self.n_beta > self.n_alpha:
                lumo_nrg = self.beta_nrgs[self.n_beta]

            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            alpha_occ = np.zeros(self.n_mos)
            beta_occ = np.zeros(self.n_mos)
            for i in range(self.n_alpha, self.n_mos):
                alpha_occ[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )
            
            for i in range(self.n_beta, self.n_mos):
                beta_occ[i] = np.exp(
                    -((chem_pot - self.beta_nrgs[i]) / delta) ** 2
                )

            alpha_occ /= sum(alpha_occ)
            beta_occ /= sum(beta_occ)
            plus_density = self.density_value(
                coords, alpha_occ=alpha_occ, beta_occ=beta_occ, **kwargs
            )

        return plus_density

    def fukui_dual_value(self, coords, delta=0.1, **kwargs):
        CITATION = "doi:10.1021/acs.jpca.9b07516"
        self.LOG.citation(CITATION)
        
        alpha_occ = np.zeros(self.n_mos)
        alpha_occ[self.n_alpha - 1] = 1
        beta_occ = None
        if self.beta_coefficients is not None:
            beta_occ = np.zeros(self.n_mos)
            beta_occ[self.n_beta - 1] = 1

        if self.beta_coefficients is None:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            plus_e = np.zeros(self.n_mos)
            minus_e = np.zeros(self.n_mos)
            for i in range(0, self.n_alpha):
                minus_e[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )

            for i in range(self.n_alpha, self.n_mos):
                plus_e[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )

            minus_e /= sum(minus_e)
            plus_e /= sum(plus_e)
            dual_density = self.density_value(
                coords, alpha_occ=plus_e - minus_e, beta_occ=beta_occ, **kwargs
            )
        
        else:
            homo_nrg = self.alpha_nrgs[self.n_alpha - 1]
            if self.n_beta > self.n_alpha:
                homo_nrg = self.beta_nrgs[self.n_beta - 1]
            
            lumo_nrg = self.alpha_nrgs[self.n_alpha]
            if self.n_beta > self.n_alpha:
                lumo_nrg = self.beta_nrgs[self.n_beta]

            chem_pot = 0.5 * (lumo_nrg + homo_nrg)
            alpha_occ = np.zeros(self.n_mos)
            beta_occ = np.zeros(self.n_mos)
            for i in range(0, self.n_alpha):
                alpha_occ[i] = -np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )
            
            for i in range(0, self.n_beta):
                beta_occ[i] = -np.exp(
                    -((chem_pot - self.beta_nrgs[i]) / delta) ** 2
                )

            for i in range(self.n_alpha, self.n_mos):
                alpha_occ[i] = np.exp(
                    -((chem_pot - self.alpha_nrgs[i]) / delta) ** 2
                )
            
            for i in range(self.n_beta, self.n_mos):
                beta_occ[i] = np.exp(
                    -((chem_pot - self.beta_nrgs[i]) / delta) ** 2
                )

            alpha_occ[self.n_alpha:] /= abs(sum(alpha_occ[self.n_alpha:]))
            beta_occ[self.n_beta:] /= abs(sum(beta_occ[self.n_beta:]))
            alpha_occ[:self.n_alpha] /= sum(alpha_occ[:self.n_alpha])
            beta_occ[:self.n_beta] /= sum(beta_occ[:self.n_beta])
            dual_density = self.density_value(
                coords, alpha_occ=alpha_occ, beta_occ=beta_occ, **kwargs
            )

        return dual_density

    @staticmethod
    def get_cube_array(
        geom,
        padding=4,
        spacing=0.2,
        standard_axes=False,
    ):
        """returns n_pts1, n_pts2, n_pts3, v1, v2, v3, com, u
        n_pts1 is the number of points along the first axis
        n_pts2 ... second axis
        n_pts3 ... third axis
        v1 is the vector for the first axis, norm should be close to spacing
        v2 ... second axis
        v3 ... third axis
        com is the center of the cube
        u is a rotation matrix for the v1, v2, v3 axes relative to xyz
        geom - Geometry() used to define the cube
        padding - extra space around atoms in angstrom
        spacing - distance between adjacent points in angstrom
        standard_axes - True to use x, y, and z axes
            by default, the cube will be oriented to fit
            the geom and have the smallest volume possible
        """

        def get_standard_axis():
            """returns info to set up a grid along the x, y, and z axes"""
            geom_coords = geom.coords

            # get range of geom's coordinates
            x_min = np.min(geom_coords[:, 0])
            x_max = np.max(geom_coords[:, 0])
            y_min = np.min(geom_coords[:, 1])
            y_max = np.max(geom_coords[:, 1])
            z_min = np.min(geom_coords[:, 2])
            z_max = np.max(geom_coords[:, 2])

            # add padding, figure out vectors
            r1 = 2 * padding + x_max - x_min
            n_pts1 = int(r1 // spacing) + 1
            d1 = r1 / (n_pts1 - 1)
            v1 = np.array((d1, 0., 0.))
            r2 = 2 * padding + y_max - y_min
            n_pts2 = int(r2 // spacing) + 1
            d2 = r2 / (n_pts2 - 1)
            v2 = np.array((0., d2, 0.))
            r3 = 2 * padding + z_max - z_min
            n_pts3 = int(r3 // spacing) + 1
            d3 = r3 / (n_pts3 - 1)
            v3 = np.array((0., 0., d3))
            com = np.array([x_min, y_min, z_min]) - padding
            u = np.eye(3)
            return n_pts1, n_pts2, n_pts3, v1, v2, v3, com, u

        if standard_axes:
            n_pts1, n_pts2, n_pts3, v1, v2, v3, com, u = get_standard_axis()
        else:
            test_coords = geom.coords - geom.COM()
            covar = np.dot(test_coords.T, test_coords)
            try:
                # use SVD on the coordinate covariance matrix
                # this decreases the volume of the box we're making
                # that means less work for higher resolution
                # for many structures, this only decreases the volume
                # by like 5%
                u, s, vh = np.linalg.svd(covar)
                v1 = u[:, 0]
                v2 = u[:, 1]
                v3 = u[:, 2]
                # change basis of coordinates to the singular vectors
                # this is how we determine the range + padding
                new_coords = np.dot(test_coords, u)
                xr_max = np.max(new_coords[:, 0])
                xr_min = np.min(new_coords[:, 0])
                yr_max = np.max(new_coords[:, 1])
                yr_min = np.min(new_coords[:, 1])
                zr_max = np.max(new_coords[:, 2])
                zr_min = np.min(new_coords[:, 2])
                com = np.array([xr_min, yr_min, zr_min]) - padding
                # move the COM back to the xyz space of the original molecule
                com = np.dot(u, com)
                com += geom.COM()
                r1 = 2 * padding + np.linalg.norm(xr_max - xr_min)
                r2 = 2 * padding + np.linalg.norm(yr_max - yr_min)
                r3 = 2 * padding + np.linalg.norm(zr_max - zr_min)
                n_pts1 = int(r1 // spacing) + 1
                n_pts2 = int(r2 // spacing) + 1
                n_pts3 = int(r3 // spacing) + 1
                v1 = v1 * r1 / (n_pts1 - 1)
                v2 = v2 * r2 / (n_pts2 - 1)
                v3 = v3 * r3 / (n_pts3 - 1)
            except np.linalg.LinAlgError:
                n_pts1, n_pts2, n_pts3, v1, v2, v3, com, u = get_standard_axis()
        
        return n_pts1, n_pts2, n_pts3, v1, v2, v3, com, u

    @staticmethod
    def get_cube_points(
        n_pts1, n_pts2, n_pts3, v1, v2, v3, com, sort=True
    ):
        """
        returns coords, n_list
        coords is an array of points in the cube
        n_list specifies where each point is along the axes
        e.g. 5th point along v1, 4th point along v2, 0th point along v3
        """
        v_list = [v1, v2, v3]
        n_list = [n_pts1, n_pts2, n_pts3]

        if sort:
            v_list = []
            n_list = []
            for n, v in sorted(
                zip([n_pts1, n_pts2, n_pts3], [v1, v2, v3]),
                key=lambda p: np.linalg.norm(p[1]),
            ):
                v_list.append(v)
                n_list.append(n)
            
        ndx = (
            np.vstack(
                np.mgrid[
                    0 : n_list[0],
                    0 : n_list[1],
                    0 : n_list[2],
                ]
            )
            .reshape(3, np.prod(n_list))
            .T
        )
        coords = np.matmul(ndx, v_list)
        del ndx
        coords += com

        return coords, n_list

    def memory_estimate(
        self,
        func_name,
        n_points=None,
        low_mem=False,
        n_jobs=1,
        apoints=None,
        rpoints=None,
        n_atoms=None,
    ):
        """
        returns the estimated memory use (in GB) for calling the
        specified function on the specified number of points
        if func_name is a condensed fukui function, apoints,
        and rpoints must be given
        otherwise, n_points must be given
        """
        test_array = np.ones(1)
        if test_array.dtype == np.float64:
            # bytes - 8 bits per byte
            num_size = 8
        else:
            # hopefully float32
            num_size = 4
        
        size = n_points
        if any(func_name == x for x in [
                "density_value",
                "fukui_acceptor_value",
                "fukui_donor_value",
                "fukui_dual_value",
            ]
        ):
            size *= num_size * 4 * max(n_jobs, n_atoms)
            if not low_mem:
                size *= self.n_mos / (2 * max(n_jobs, n_atoms))
        elif func_name == "mo_value":
            size *= num_size * (4 * n_jobs + max(n_atoms - n_jobs, 0))
        elif any(func_name == x for x in [
                "condensed_fukui_acceptor_values",
                "condensed_fukui_donor_values",
                "condensed_fukui_dual_values",
            ]
        ):
            density_size = self.memory_estimate(
                "density_value",
                n_points=apoints * rpoints,
                n_jobs=n_jobs,
                n_atoms=n_atoms,
                low_mem=low_mem,
            )
            mat_size = num_size * n_atoms * rpoints * apoints
            size = max(density_size, mat_size)
        
        return size * 1e-9

    def voronoi_integral(
        self,
        target,
        geom,
        *args,
        rpoints=32,
        apoints=1454,
        func=None,
        rmax=None,
        **kwargs,
    ):
        """
        integrates func in the Voronoi cell of the specified target
        geom - Geometry() target belongs to
        args - passed to func
        rpoints - radial points used for Gauss-Legendre integral
        apoints - angular points for Lebedev integral
        func - function to evaluate
        kwargs - passed to func
        """
        
        atom = geom.find(target)[0]
    
        if rmax is None:
            rmax = 10 * atom._vdw

        rgrid, rweights = gauss_legendre_grid(
            start=0, stop=rmax, num=rpoints
        )
        # grab Lebedev grid for unit sphere at origin
        agrid, aweights = lebedev_sphere(
            radius=1, center=np.zeros(3), num=apoints
        )

        # TODO: switch to np.zeros((n_ang * n_rad, 3))
        # this eliminates appending
        # build a list of points and weights around the atom
        all_points = np.empty((0, 3))
        weights = np.empty(0)
        
        for rvalue, rweight in zip(rgrid, rweights):
            agrid_r = agrid * rvalue
            agrid_r += atom.coords
            all_points = np.append(all_points, agrid_r, axis=0)
            weights = np.append(weights, rweight * aweights)
        
        # find points that are closest to this atom
        # than any other
        dist_mat = distance_matrix(geom.coords, all_points)
        atom_ndx = geom.atoms.index(atom)
        mask = np.argmin(dist_mat, axis=0) == atom_ndx
        
        voronoi_points = all_points[mask]
        voronoi_weights = weights[mask]
        
        # evaluate function
        vals = func(voronoi_points, *args, **kwargs)
        
        # multiply values by weights, add them up, and return the sum
        return np.dot(vals, voronoi_weights)

    def power_integral(
        self,
        target,
        geom,
        *args,
        radii="umn",
        rpoints=32,
        apoints=1454,
        func=None,
        rmax=None,
        **kwargs,
    ):
        """
        integrates func in the power cell of the specified target
        power diagrams are a form of weighted Voronoi diagrams
        that form cells based on the smallest d^2 - r^2
        see wikipedia article: https://en.wikipedia.org/wiki/Power_diagram
        radii - "bondi" - Bondi vdW radii
                "umn"   - vdW radii from Mantina, Chamberlin, Valero, Cramer, and Truhlar
                dict()  - radii are values and elements are keys
                list()  - list of radii corresponding to targets
        geom - Geometry() target belongs to
        args - passed to func
        rpoints - radial points used for Gauss-Legendre integral
        apoints - angular points for Lebedev integral
        func - function to evaluate
        kwargs - passed to func
        """
        
        if func is None:
            func = self.density_value
        
        target = geom.find(target)[0]
        target_ndx = geom.atoms.index(target)

        radius_list = []
        radii_dict = None
        if isinstance(radii, dict):
            radii_dict = radii
        elif isinstance(radii, list):
            radius_list = radii
        elif radii.lower() == "bondi":
            radii_dict = BONDI_RADII
        elif radii.lower() == "umn":
            radii_dict = VDW_RADII
        else:
            raise TypeError(
                "radii must be list, dict, \"UMN\", or \"BONDI\": %s" % radii
            )

        if not radius_list:
            for atom in geom.atoms:
                radius_list.append(radii_dict[atom.element])

        radius_list = np.array(radius_list)

        if rmax is None:
            rmax = 5 * radius_list[target_ndx]

        radius_list = radius_list ** 2

        rgrid, rweights = gauss_legendre_grid(
            start=0, stop=rmax, num=rpoints
        )
        # grab Lebedev grid for unit sphere at origin
        agrid, aweights = lebedev_sphere(
            radius=1, center=np.zeros(3), num=apoints
        )

        # TODO: switch to np.zeros((n_ang * n_rad, 3))
        # this eliminates appending
        # build a list of points and weights around the atom
        power_points = np.empty((0, 3))
        power_weights = np.empty(0)
        
        atom_ndx = geom.atoms.index(target)
        found_pts = False
        for rvalue, rweight in zip(rgrid, rweights):
            agrid_r = agrid * rvalue
            agrid_r += target.coords
            dist_mat = distance_matrix(geom.coords, agrid_r) ** 2
            dist_mat = np.transpose(dist_mat.T - radius_list)
            mask = np.argmin(dist_mat, axis=0) == atom_ndx
            # find points that are closest to this atom's vdw sphere
            # than any other
            if any(mask):
                power_points = np.append(power_points, agrid_r[mask], axis=0)
                power_weights = np.append(power_weights, rweight * aweights[mask])
                found_pts = True
            elif found_pts:
                break

        # with open("test_%s.bild" % target.name, "w") as f:
        #     s = ""
        #     for p in power_points:
        #         s += ".sphere %.4f %.4f %.4f 0.05\n" % tuple(p)
        #     f.write(s)
        
        # evaluate function
        vals = func(power_points, *args, **kwargs)
        
        # multiply values by weights, add them up, and return the sum
        return np.dot(vals, power_weights)

    def condensed_fukui_donor_values(
        self,
        geom,
        *args,
        **kwargs,
    ):
        """
        uses power_integral to integrate the fukui_donor_value
        for all atoms in geom
        values are normalized so they sum to 1
        geom - Geometry()
        args and kwargs are passed to power_integral
        returns array for each atom's condensed Fukui donor values
        """
        out = np.zeros(len(geom.atoms))
        for i, atom in enumerate(geom.atoms):
            out[i] = self.power_integral(
                atom, geom, *args, func=self.fukui_donor_value, **kwargs,
            )
        
        out /= sum(out)
        return out

    def condensed_fukui_acceptor_values(
        self,
        geom,
        *args,
        **kwargs,
    ):
        """
        uses power_integral to integrate the fukui_acceptor_value
        for all atoms in geom
        values are normalized so they sum to 1
        geom - Geometry()
        args and kwargs are passed to power_integral
        returns array for each atom's condensed Fukui acceptor values
        """
        out = np.zeros(len(geom.atoms))
        for i, atom in enumerate(geom.atoms):
            out[i] = self.power_integral(
                atom, geom, *args, func=self.fukui_acceptor_value, **kwargs,
            )
        
        out /= sum(out)
        return out

    def condensed_fukui_dual_values(
        self,
        geom,
        *args,
        **kwargs,
    ):
        """
        returns the difference between condensed_fukui_acceptor_values
        and condensed_fukui_donor_values
        """
        out = self.condensed_fukui_acceptor_values(
            geom, *args, **kwargs,
        ) - self.condensed_fukui_donor_values(
            geom, *args, **kwargs,
        )
        
        return out
