import numpy as np

import scipy.interpolate
import scipy.integrate


class Pathway:
    """
    Cartesian interpolation of a molecule's coordinates
    """
    def __init__(
        self,
        coordinates,
        interpolation="clampedcubic",
        other_vars=None,
        x_vals=None,
        weights=None,
        **interpolator_kwargs
    ):
        self.coords = coordinates
        self.interpolation = interpolation
        self.other_vars = other_vars
        self._interpolation = None
        self._other_vars_interpolation = None
        self._region_lengths = None
        self.weights = weights
        
        if x_vals is None:
            x_vals = np.arange(0, len(self.coords))
        
        self.x_vals = x_vals
        
        self._build_interpolation(**interpolator_kwargs)
    
    def _build_interpolation(self, **interpolator_kwargs):
        n_nodes = len(self.coords)
        n_atoms = len(self.coords[0])
        coords = np.reshape(self.coords, (n_nodes, 3 * n_atoms))
        if self.weights is None:
            self.weights = np.ones(n_atoms)
        weights = np.repeat(self.weights, 3)
        for i in range(0, n_nodes):
            coords[i] *= weights

        if self.other_vars:
            self._other_vars_interpolation = dict()

        if self.interpolation.lower().startswith("akima"):
            interp_func = scipy.interpolate.Akima1DInterpolator
        
        elif self.interpolation.lower().startswith("pchip"):
            interp_func = scipy.interpolate.PchipInterpolator
        
        elif self.interpolation.lower().startswith("naturalcubic"):
            interp_func = lambda *args: scipy.interpolate.CubicSpline(
                *args, bc_type="natural"
            )

        elif self.interpolation.lower().startswith("clampedcubic"):
            interp_func = lambda *args: scipy.interpolate.CubicSpline(
                *args, bc_type="clamped"
            )
        
        elif self.interpolation.lower().startswith("nakcubic"):
            interp_func = lambda *args: scipy.interpolate.CubicSpline(
                *args, bc_type="not-a-knot"
            )
        
        else:
            raise RuntimeError(
                "interpolation method not known: %s" % self.interpolation
            )
        
        self._interpolation = [
            interp_func(self.x_vals, coord, **interpolator_kwargs) for coord in coords.T
        ]
        if self.other_vars:
            for var, vals in self.other_vars.items():
                self._other_vars_interpolation[var] = interp_func(
                    self.x_vals,
                    vals,
                )
    
        deriv = [interp.derivative(1) for interp in self._interpolation]
        region_starts = self.x_vals[:-1]
        region_stops = self.x_vals[1:]
        self._region_lengths = np.zeros(n_nodes - 1)
        n_pts = 33
        for i in range(0, len(region_starts)):
            total = 0
            a = region_starts[i]
            b = region_stops[i]
            for k, x in enumerate(
                np.linspace(a, b, num=n_pts)
            ):
                region_length = 0
                for j in range(0, 3 * n_atoms):
                    val = deriv[j](x) ** 2
                    if k == 0 or k == n_pts - 1:
                        region_length += val
                    elif k % 2 == 0:
                        region_length += 2 * val
                    else:
                        region_length += 4 * val
                
                total += np.sqrt(region_length)
            
            total = (b - a) * total / (n_pts * 6)
            self._region_lengths[i] = total

    def s_at_t(self, t):
        total_length = sum(self._region_lengths)
        u = t * total_length
        r = 0
        for i in range(0, len(self._region_lengths)):
            if u > sum(self._region_lengths[:i]):
                r = i
    
        # print("point is in region %i" % r)
        s = self.x_vals[r]
        s += (self.x_vals[r + 1] - self.x_vals[r]) * (u - sum(self._region_lengths[:r])) / self._region_lengths[r]
        return min(s, max(self.x_vals))

    def interpolate_coords(self, t):
        """
        returns array for the interpolated pathway
        at point t
        
        t should be between 0 and 1
        """
        s = self.s_at_t(t)
        # print("getting coordinates at s=%.4f (t=%.2f)" % (s, t))
        interpolated_coords = np.array([
            coord(s) for coord in self._interpolation
        ])
        interpolated_coords
        interpolated_coords = interpolated_coords.reshape(
            (len(interpolated_coords) // 3, 3)
        )
        return interpolated_coords / self.weights[:, np.newaxis]
    
    def interpolate_geometry(self, t, geom):
        geom = geom.copy()
        coords = self.interpolate_coords(t)
        geom.coords = coords
        return geom
    
    def interpolate_other_var(self, t, var):
        s = self.s_at_t(t)
        return self._other_vars_interpolation[var](s)

    def other_var_derivative(self, t, var):
        s = self.s_at_t(t)
        return self._other_vars_interpolation[var].derivative(1)(s)
    
    def tangent(self, t):
        s = self.s_at_t(t)
        deriv = [interp.derivative(1) for interp in self._interpolation]
        tangent = np.array([f(s) for f in deriv])
        tangent = tangent.reshape(
            (len(tangent) // 3, 3)
        )
        return tangent / self.weights[:, np.newaxis]
