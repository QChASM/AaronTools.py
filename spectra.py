"""classes for storing signal data and plotting various spectra (IR, UV/vis, etc.)"""
import re

import numpy as np

from AaronTools import addlogger
from AaronTools.const import (
    UNIT,
    PHYSICAL,
    COMMONLY_ODD_ISOTOPES,
)
from AaronTools.utils.utils import (
    float_num,
    pascals_triangle,
    shortest_path,
)


class Signal:
    """
    parent class for each signal in a spectrum
    """
    # attribute for the x position of this signal
    x_attr = None
    required_attrs = ()
    nested = None
    def __init__(self, x_var, **kwargs):
        for attr in self.required_attrs:
            setattr(self, attr, None)

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

        setattr(self, self.x_attr, x_var)

    def __repr__(self):
        s = self.__class__.__name__
        s += "("
        s += str(getattr(self, self.x_attr))
        s += ", "
        for attr in self.__dict__:
            if attr == self.x_attr:
                continue
            s += attr
            s += "="
            s += str(getattr(self, attr))
            s += ", "
        s.rstrip(", ")
        s += ")"
        return s


@addlogger
class Signals:
    """
    parent class for storing data for different signals in the
    spectrum and plotting a simulated spectrum

    Attributes: 

    * data
    * style
    * lines
    """
    
    # label for x axis - should be set by child classes
    x_label = None
    
    LOG = None
    
    def __init__(self, data, style="gaussian", *args, **kwargs):
        self.data = []
        if isinstance(data[0], Signal):
            self.data = data
            return

        lines = False
        if isinstance(data, str):
            lines = data.splitlines()

        if lines and style == "gaussian":
            self.parse_gaussian_lines(lines, *args, **kwargs)
        elif lines and style == "orca":
            self.parse_orca_lines(lines, *args, **kwargs)
        elif lines and style == "psi4":
            self.parse_psi4_lines(lines, *args, **kwargs)
        elif lines and style == "qchem":
            self.parse_qchem_lines(lines, *args, **kwargs)
        else:
            raise NotImplementedError("cannot parse data for %s" % style)

    def parse_gaussian_lines(self, lines, *args, **kwargs):
        """parse data from Gaussian output files related to this spectrum"""
        raise NotImplementedError(
            "parse_gaussian_lines not implemented by %s" %
            self.__class__.__name__
        )

    def parse_orca_lines(self, lines, *args, **kwargs):
        """parse data from ORCA output files related to this spectrum"""
        raise NotImplementedError(
            "parse_orca_lines not implemented by %s" %
            self.__class__.__name__
        )

    def parse_psi4_lines(self, lines, *args, **kwargs):
        """parse data from Psi4 output files related to this spectrum"""
        raise NotImplementedError(
            "parse_psi4_lines not implemented by %s" %
            self.__class__.__name__
        )

    def parse_qchem_lines(self, lines, *args, **kwargs):
        """parse data from Q-Chem output files related to this spectrum"""
        raise NotImplementedError(
            "parse_qchem_lines not implemented by %s" %
            self.__class__.__name__
        )

    def filter_data(self, signal):
        """
        used to filter out some data from the spectrum (e.g.
        imaginary modes from an IR spec)
        return False if signal should not be in the spectrum
        """
        return True

    def get_spectrum_functions(
        self,
        fwhm=15.0,
        peak_type="pseudo-voigt",
        voigt_mixing=0.5,
        scalar_scale=0.0,
        linear_scale=0.0,
        quadratic_scale=0.0,
        intensity_attr="intensity",
        data_attr="data",
    ):
        """
        returns a list of functions that can be evaluated to
        produce a spectrum
        
        :param float fwhm: full width at half max of each peak
        :param str peak_type: gaussian, lorentzian, pseudo-voigt, or delta
        :param float voigt_mixing: ratio of pseudo-voigt that is gaussian
        :param float scalar_scale: shift x data
        :param float linear_scale: scale x data
        :param float quadratic_scale: scale x data
            
            x' = (1 - linear_scale * x - quadratic_scale * x^2 - scalar_scale)
        :param str intensity_attr: attribute of Signal used for the intensity
            of that signal
        :param str data_attr: attribute of self for the list of Signal()
        """
        data = getattr(self, data_attr)
        x_attr = data[0].x_attr
        
        
        # scale x positions
        if not data[0].nested:
            x_positions = np.array(
                [getattr(d, x_attr) for d in data if self.filter_data(d)]
            )
    
            intensities = [
                getattr(d, intensity_attr) for d in data if self.filter_data(d)
            ]
        else:
            x_positions = []
            intensities = []
            x_positions.extend(
                [getattr(d, x_attr) for d in data if self.filter_data(d)]
            )
            intensities.extend(
                [getattr(d, intensity_attr) for d in data if self.filter_data(d)]
            )
            for nest in data[0].nested:
                for d in data:
                    nest_attr = getattr(d, nest)
                    if isinstance(nest_attr, dict):
                        for value in nest_attr.values():
                            if hasattr(value, "__iter__"):
                                for item in value:
                                    x_positions.append(getattr(item, x_attr))
                                    intensities.append(getattr(item, intensity_attr))
                            else:
                                x_positions.append(getattr(value, x_attr))
                                intensities.append(getattr(value, intensity_attr))
                    elif hasattr(nest_attr, "__iter__"):
                        for item in nest_attr:
                            x_positions.append(getattr(item, x_attr))
                            intensities.append(getattr(item, intensity_attr))
                    else:
                        x_positions.append(getattr(nest_attr, x_attr))
                        intensities.append(getattr(nest_attr, intensity_attr))
                
            x_positions = np.array(x_positions)

        x_positions -= (
            linear_scale * x_positions + quadratic_scale * x_positions ** 2
        )
        x_positions += scalar_scale

        e_factor = -4 * np.log(2) / fwhm ** 2
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        functions = []

        for x_pos, intensity in zip(x_positions, intensities):
            if intensity is not None:
                if peak_type.lower() == "gaussian":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * np.exp(e_factor * (x - x0) ** 2)
                        * fwhm / (2 * np.sqrt(2 * np.log(2)))
                    )

                elif peak_type.lower() == "lorentzian":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * (
                            0.5 * fwhm
                            / (np.pi * ((x - x0) ** 2 + (0.5 * fwhm) ** 2))
                        )
                    )

                elif peak_type.lower() == "pseudo-voigt":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * (
                            (1 - voigt_mixing)
                            * (
                                (0.5 * fwhm) ** 2
                                / (((x - x0) ** 2 + (0.5 * fwhm) ** 2))
                            )
                            + voigt_mixing
                            * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
                        )
                    )

                elif peak_type.lower() == "delta":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * int(x == x0)
                    )

        return functions, x_positions, intensities

    @staticmethod
    def get_plot_data(
        functions,
        signal_centers,
        point_spacing=None,
        transmittance=False,
        peak_type="pseudo-voigt",
        normalize=True,
        fwhm=15.0,
        change_x_unit_func=None,
        show_functions=None,
    ):
        """
        :returns: arrays of x_values, y_values for a spectrum
        
        :param float point_spacing: spacing between points, default is higher resolution around
            each peak (i.e. not uniform)
            this is pointless if peak_type == delta
        :param float fwhm: full width at half max
        :param bool transmittance: if true, take 10^(2 - y_values) before returning
            to get transmittance as a %
        :param str peak_type: pseudo-voigt, gaussian, lorentzian, or delta
        :param float voigt_mixing: fraction of pseudo-voigt that is gaussian
        :param float linear_scale: subtract linear_scale * frequency off each mode
        :param float quadratic_scale: subtract quadratic_scale * frequency^2 off each mode
        """
        other_y_list = []
        if peak_type.lower() != "delta":
            if point_spacing is not None:
                x_values = []
                x = -point_spacing
                stop = max(signal_centers)
                stop += 5 * fwhm
                while x < stop:
                    x += point_spacing
                    x_values.append(x)
            
                x_values = np.array(x_values)
            
            else:
                xmax = max(signal_centers) + 10 * (fwhm + 1)
                xmin = min(signal_centers) - 10 * (fwhm + 1)
                if xmax > 0:
                    xmax *= 1.1
                else:
                    xmax *= 0.9
                if xmin < 0:
                    xmin *= 1.1
                else:
                    xmin = 0
                x_values = np.linspace(
                    xmin,
                    xmax,
                    num=1000,
                ).tolist()
            
                prev_pt = None
                for x in signal_centers:
                    if prev_pt is not None and abs(x - prev_pt) < fwhm / 10:
                        continue
                    prev_pt = x
                    x_values.extend(
                        np.linspace(
                            x - (10 * fwhm),
                            x + (10 * fwhm),
                            num=250,
                        ).tolist()
                    )
                    x_values.append(x)

                if not point_spacing:
                    x_values = np.array(list(set(x_values)))
                    x_values.sort()

            y_values = np.zeros(len(x_values))
            for i, f in enumerate(functions):
                y_values += f(x_values)

            if show_functions:
                if (
                    len(show_functions[0]) == 2 and
                    all(isinstance(n, int) for n in show_functions[0])
                ):
                    for (ndx1, ndx2) in show_functions:
                        other_y_list.append(
                            np.sum(
                                [f(x_values) for f in functions[ndx1: ndx2]],
                                axis=0,
                            )
                        )
                else:
                    for comp in show_functions:
                        other_y_list.append(
                            np.sum([f(x_values) for f in comp], axis=0)
                        )

        else:
            x_values = []
            y_values = []

            for freq, func in zip(signal_centers, functions):
                y_values.append(func(freq))
                x_values.append(freq)

            y_values = np.array(y_values)

        if len(y_values) == 0:
            Signals.LOG.warning("nothing to plot")
            return None

        if normalize or transmittance:
            max_val = abs(max(y_values.max(), y_values.min(), key=abs))
            y_values /= max_val
            for y_vals in other_y_list:
                y_vals /= max_val

        if transmittance:
            y_values = np.array([10 ** (2 - y) for y in y_values])
            for i in range(0, len(other_y_list)):
                other_y_list[i] = np.array(
                    [10 ** (2 - y) for y in other_y_list[i]]
                )

        if change_x_unit_func:
            x_values, ndx = change_x_unit_func(x_values)
            y_values = y_values[ndx]
            for i in range(0, len(other_y_list)):
                other_y_list[i] = other_y_list[i][ndx]

        return x_values, y_values, other_y_list

    @classmethod
    def plot_spectrum(
        cls,
        figure,
        x_values,
        y_values,
        other_y_values=None,
        other_y_style=None,
        centers=None,
        widths=None,
        exp_data=None,
        reverse_x=None,
        y_label=None,
        plot_type="transmittance",
        x_label=r"wavenumber (cm$^{-1}$)",
        peak_type="pseudo-voigt",
        rotate_x_ticks=False,
    ):
        """
        plot the x_data and y_data on figure (matplotlib figure)
        this is intended for IR spectra

        :param np.ndarray centers: array-like of float, plot is split into sections centered
            on the frequency specified by centers
            
            default is to not split into sections
        :param np.ndarray widths: array-like of float, defines the width of each section
        :param list exp_data: other data to plot
            should be a list of (x_data, y_data, color)
        :param bool reverse_x: if True, 0 cm^-1 will be on the right
        """

        if not centers:
            # if no centers were specified, pretend they were so we
            # can do everything the same way
            axes = [figure.subplots(nrows=1, ncols=1)]
            y_nonzero = np.nonzero(y_values)[0]
            x_values = np.array(x_values)
            widths = [max(x_values[y_nonzero])]
            centers = [max(x_values[y_nonzero]) / 2]
        else:
            n_sections = len(centers)
            figure.subplots_adjust(wspace=0.05)
            # sort the sections so we don't jump around
            widths = [
                x
                for _, x in sorted(
                    zip(centers, widths),
                    key=lambda p: p[0],
                    reverse=reverse_x,
                )
            ]
            centers = sorted(centers, reverse=reverse_x)

            axes = figure.subplots(
                nrows=1,
                ncols=n_sections,
                sharey=True,
                gridspec_kw={"width_ratios": widths},
            )
            if not hasattr(axes, "__iter__"):
                # only one section was specified (e.g. zooming in on a peak)
                # make sure axes is iterable
                axes = [axes]

        for i, ax in enumerate(axes):
            if i == 0:
                ax.set_ylabel(y_label)

                # need to split plot into sections
                # put a / on the border at the top and bottom borders
                # of the plot
                if len(axes) > 1:
                    ax.spines["right"].set_visible(False)
                    ax.tick_params(labelright=False, right=False)
                    ax.plot(
                        [1, 1],
                        [0, 1],
                        marker=((-1, -1), (1, 1)),
                        markersize=5,
                        linestyle="none",
                        color="k",
                        mec="k",
                        mew=1,
                        clip_on=False,
                        transform=ax.transAxes,
                    )

            elif i == len(axes) - 1 and len(axes) > 1:
                # last section needs a set of / too, but on the left side
                ax.spines["left"].set_visible(False)
                ax.tick_params(labelleft=False, left=False)
                ax.plot(
                    [0, 0],
                    [0, 1],
                    marker=((-1, -1), (1, 1)),
                    markersize=5,
                    linestyle="none",
                    color="k",
                    mec="k",
                    mew=1,
                    clip_on=False,
                    transform=ax.transAxes,
                )

            elif len(axes) > 1:
                # middle sections need two sets of /
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.tick_params(
                    labelleft=False, labelright=False, left=False, right=False
                )
                ax.plot(
                    [0, 0],
                    [0, 1],
                    marker=((-1, -1), (1, 1)),
                    markersize=5,
                    linestyle="none",
                    label="Silence Between Two Subplots",
                    color="k",
                    mec="k",
                    mew=1,
                    clip_on=False,
                    transform=ax.transAxes,
                )
                ax.plot(
                    [1, 1],
                    [0, 1],
                    marker=((-1, -1), (1, 1)),
                    markersize=5,
                    label="Silence Between Two Subplots",
                    linestyle="none",
                    color="k",
                    mec="k",
                    mew=1,
                    clip_on=False,
                    transform=ax.transAxes,
                )

            if peak_type.lower() != "delta":
                ax.plot(
                    x_values,
                    y_values,
                    color="k",
                    linewidth=1,
                    label="computed",
                )
                
                if other_y_values:
                    for y_vals, style in zip(other_y_values, other_y_style):
                        ax.plot(
                            x_values,
                            y_vals,
                            color=style[0],
                            linestyle=style[1],
                            linewidth=1,
                            label=style[2],
                            zorder=-1,
                        )

            else:
                if plot_type.lower() == "transmittance":
                    ax.vlines(
                        x_values,
                        y_values,
                        [100 for y in y_values],
                        linewidth=1,
                        colors=["k" for x in x_values],
                        label="computed",
                    )
                    ax.hlines(
                        100,
                        0,
                        max(4000, *x_values),
                        linewidth=1,
                        colors=["k" for y in y_values],
                        label="computed",
                    )
                    
                    if other_y_values:
                        for y_vals, style in zip(other_y_values, other_y_style):
                            ax.vlines(
                                x_values,
                                y_vals,
                                [100 for y in y_vals],
                                colors=[style[0] for x in x_values],
                                linestyles=style[1],
                                linewidth=1,
                                label=style[2],
                                zorder=-1,
                            )
                else:
                    ax.vlines(
                        x_values,
                        [0 for y in y_values],
                        y_values,
                        linewidth=1,
                        colors=["k" for x in x_values],
                        label="computed",
                    )
                    ax.hlines(
                        0,
                        0,
                        max(4000, *x_values),
                        linewidth=1,
                        colors=["k" for y in y_values],
                        label="computed",
                    )
                    
                    if other_y_values:
                        for y_vals, style in zip(other_y_values, other_y_style):
                            ax.vlines(
                                x_values,
                                [0 for y in y_vals],
                                y_vals,
                                colors=[style[0] for x in x_values],
                                linestyles=style[1],
                                linewidth=1,
                                label=style[2],
                                zorder=-1,
                            )

            if exp_data:
                for x, y, color in exp_data:
                    ax.plot(
                        x,
                        y,
                        color=color,
                        zorder=1,
                        linewidth=1,
                        label="observed",
                    )

            center = centers[i]
            width = widths[i]
            high = center + width / 2
            low = center - width / 2
            if reverse_x:
                ax.set_xlim(high, low)
            else:
                ax.set_xlim(low, high)

        # b/c we're doing things in sections, we can't add an x-axis label
        # well we could, but which section would be put it one?
        # it wouldn't be centered
        # so instead the x-axis label is this
        figure.text(
            0.5, 0.0, x_label, ha="center", va="bottom"
        )
        if rotate_x_ticks:
            figure.autofmt_xdate(rotation=-45, ha="center")

    @classmethod
    def get_mixed_signals(
        cls,
        signal_groups,
        weights,
        fractions=None,
        data_attr="data",
        **kwargs,
    ):
        """
        get signals for a mixture of components or conformers
        
        :param list(Signal)|list(list(Signal)) signal_groups: list of Signals() instances or list of lists of Signals()
            
            a list of Signals() is a group of conformers
            
            a list of lists of Signals() are the different components
        
        :param iterable weights: weights for each conformer, organized according to signal_groups
        :param iterable fractions: fraction of each component in the mixture
            default: all components have equal fractions
        :param str data_attr: attribute of Signals() for data
        :param kwargs: passed to cls.__init__, along with a new list of data
        """
        if not hasattr(signal_groups[0], "__iter__"):
            signal_groups = [signal_groups]
        
        if not hasattr(weights[0], "__iter__"):
            weights = [weights]
        
        if fractions is None:
            fractions = np.ones(len(signal_groups))
        
        new_data = []
        for group, weighting, fraction in zip(signal_groups, weights, fractions):
            for signals, weight in zip(group, weighting):
                data = getattr(signals, data_attr)
                for d in data:
                    x_val = getattr(d, d.x_attr)
                    vals = d.__dict__
                    data_cls = d.__class__
                    new_vals = dict()
                    for key, item in vals.items():
                        if isinstance(item, float):
                            new_vals[key] = fraction * weight * item
                        else:
                            new_vals[key] = item
                    
                    if d.nested:
                        if not isinstance(d.nested, str):
                            for attr in d.nested:
                                nest = getattr(d, attr)
                                nest_vals = dict()
                                if isinstance(nest, dict):
                                    for k, items in nest.items():
                                        for i, item in enumerate(items):
                                            nest_x_val = getattr(item, item.x_attr)
                                            vals = item.__dict__
                                            nest_cls = item.__class__
                                            for k2, j in vals.items():
                                                if isinstance(item, float):
                                                    nest_vals[k2] = fraction * weight * j
                                                else:
                                                    nest_vals[k2] = j
                                            new_vals[attr][k][i] = nest_cls(
                                                nest_x_val, **nest_vals
                                            )
                                elif hasattr(nest, "__iter__"):
                                    for i, item in enumerate(nest):
                                        nest_x_val = getattr(item, item.x_attr)
                                        vals = item.__dict__
                                        nest_cls = item.__class__
                                        for k, j in vals.items():
                                            if isinstance(item, float):
                                                nest_vals[k] = fraction * weight * j
                                            else:
                                                nest_vals[k] = j
                                        new_vals[attr][i] = nest_cls(
                                            nest_x_val, **nest_vals
                                        )
                                else:
                                    nest_x_val = getattr(nest, nest.x_attr)
                                    vals = nest.__dict__
                                    nest_cls = nest.__class__
                                    for k, j in vals.items():
                                        if isinstance(nest, float):
                                            nest_vals[k] = fraction * weight * j
                                        else:
                                            nest_vals[k] = j
                                    new_vals[attr][k] = nest_cls(
                                        nest_x_val, **nest_vals
                                    )
                    
                    new_data.append(data_cls(x_val, **new_vals))

        return cls(new_data, **kwargs)


class HarmonicVibration(Signal):
    x_attr = "frequency"
    required_attrs = (
        "intensity", "vector", "symmetry", "rotation", "raman_activity", "forcek", "red_mass",
    )


class AnharmonicVibration(Signal):
    x_attr = "frequency"
    required_attrs = (
        "intensity", "harmonic", "overtones", "combinations",
        "rotation", "raman_activity",
    )
    nested = ("overtones", "combinations")

    @property
    def harmonic_frequency(self):
        return self.harmonic.frequency

    @property
    def delta_anh(self):
        return self.frequency - self.harmonic.frequency


class Frequency(Signals):
    """for spectra in the IR/NIR region based on vibrational modes"""
    
    def __init__(self, *args, harmonic=True, hpmodes=None, **kwargs):
        super().__init__(*args, harmonic=harmonic, hpmodes=hpmodes, **kwargs)
        self.anharm_data = None
        self.imaginary_frequencies = None
        self.real_frequencies = None
        self.lowest_frequency = None
        self.by_frequency = {}
        self.is_TS = None
        self.sort_frequencies()
        
        # some software doesn't print reduced mass or force constants
        # we can calculate them if we have atom with mass, displacement
        # vectors, and vibrational frequencies
        try:
            if self.data and (
                not self.data[-1].forcek or
                not self.data[-1].red_mass
            ) and "atoms" in kwargs:
                atoms = kwargs["atoms"]
                for mode in self.data:
                    norm = sum(np.sum(mode.vector ** 2, axis=1))
                    disp = mode.vector / norm
                    mode.vector = disp
                    mu = 0
                    for i in range(0, len(mode.vector)):
                        mu += np.dot(disp[i], disp[i]) * atoms[i].mass
                    mode.red_mass = mu
                    k = 4 * np.pi ** 2 * mode.frequency ** 2
                    k *= PHYSICAL.SPEED_OF_LIGHT ** 2 * mu
                    k *= UNIT.AMU_TO_KG * 1e-2
                    mode.forcek = k
        except (IndexError, AttributeError) as e:
            # some software can compute frequencies with a user-supplied
            # hessian, so it never prints the structure
            # ORCA can do this. It will print the input structure, but
            # we don't parse that
            self.LOG.warning("issue calcing red mass", e)
            pass
    
    def parse_gaussian_lines(
        self, lines, *args, hpmodes=None, harmonic=True, **kwargs
    ):
        if harmonic:
            return self._parse_harmonic_gaussian(lines, hpmodes=hpmodes)
        return self._parse_anharmonic_gaussian(lines)

    def _parse_harmonic_gaussian(self, lines, hpmodes):
        if hpmodes is None:
            raise TypeError(
                "hpmodes argument required when data is a string"
            )
        hpmodes_re = re.compile("^\s+\d+\s+\d+\s+\d+(\s+[+-]?\d+\.\d+)+$")
        regmodes_re = re.compile("^\s+\d+\s+\d+(\s+[+-]?\d+\.\d+)+$")
        num_head = 0
        for line in lines:
            if "Harmonic frequencies" in line:
                num_head += 1
        if hpmodes and num_head != 2:
            raise RuntimeError("Log file damaged, cannot get frequencies")

        num_head = 0
        idx = -1
        modes = []
        for k, line in enumerate(lines):
            if "Harmonic frequencies" in line:
                num_head += 1
                if hpmodes and num_head == 2:
                    # if hpmodes, want just the first set of freqs
                    break
                continue
            if not hpmodes and "Frequencies" in line and "---" in line:
                hpmodes = True
            if "Frequencies" in line and (
                (hpmodes and "---" in line) or (" -- " in line and not hpmodes)
            ):
                for i, symm in zip(
                    float_num.findall(line), lines[k - 1].split()
                ):
                    self.data += [HarmonicVibration(float(i), symmetry=symm)]
                    modes += [[]]
                    idx += 1
                continue

            if ("Force constants" in line and "---" in line and hpmodes) or (
                "Frc consts" in line and "--" in line and not hpmodes
            ):
                force_constants = float_num.findall(line)
                for i in range(-len(force_constants), 0, 1):
                    self.data[i].forcek = float(force_constants[i])
                continue

            if ("Reduced masses" in line and "---" in line and hpmodes) or (
                "Red. masses" in line and "--" in line and not hpmodes
            ):
                red_masses = float_num.findall(line)
                for i in range(-len(red_masses), 0, 1):
                    self.data[i].red_mass = float(red_masses[i])
                continue

            if ("Rot. strength" in line and "---" in line and hpmodes) or (
                "Rot. str." in line and "--" in line and not hpmodes
            ):
                roational_strength = float_num.findall(line)
                for i in range(-len(roational_strength), 0, 1):
                    self.data[i].rotation = float(roational_strength[i])
                continue

            if ("Raman Activities" in line and "---" in line and hpmodes) or (
                "Raman Activ" in line and "--" in line and not hpmodes
            ):
                roational_strength = float_num.findall(line)
                for i in range(-len(roational_strength), 0, 1):
                    self.data[i].raman_activity = float(roational_strength[i])
                continue

            if "IR Inten" in line and (
                (hpmodes and "---" in line) or (not hpmodes and "--" in line)
            ):
                intensities = float_num.findall(line)
                for i in range(-len(intensities), 0, 1):
                    self.data[i].intensity = float(intensities[i])
                continue

            if hpmodes:
                match = hpmodes_re.search(line)
                if match is None:
                    continue
                values = float_num.findall(line)
                coord = int(values[0]) - 1
                atom = int(values[1]) - 1
                moves = values[3:]
                for i, m in enumerate(moves):
                    tmp = len(moves) - i
                    mode = modes[-tmp]
                    try:
                        vector = mode[atom]
                    except IndexError:
                        vector = [0, 0, 0]
                        modes[-tmp] += [[]]
                    vector[coord] = m
                    modes[-tmp][atom] = vector
            else:
                match = regmodes_re.search(line)
                if match is None:
                    continue
                data = line.split()
                # values = float_num.findall(line)
                atom = int(data[0]) - 1
                moves = np.array(data[2:], dtype=float)
                n_moves = len(moves) // 3
                for i in range(-n_moves, 0):
                    modes[i].append(
                        moves[3 * n_moves + 3 * i : 3 * n_moves + 3 * (i + 1)]
                    )

        for mode, data in zip(modes, self.data):
            data.vector = np.array(mode, dtype=np.float64)

    def _parse_anharmonic_gaussian(self, lines):
        reading_combinations = False
        reading_overtones = False
        reading_fundamentals = False

        combinations = []
        overtones = []
        fundamentals = []

        mode_re = re.compile(r"(\d+)\((\d+)\)")

        for line in lines:
            if "---" in line or "Mode" in line or not line.strip():
                continue
            if "Fundamental Bands" in line:
                reading_fundamentals = True
                continue
            if "Overtones" in line:
                reading_overtones = True
                continue
            if "Combination Bands" in line:
                reading_combinations = True
                continue

            if reading_combinations:
                info = line.split()
                mode1 = mode_re.search(info[0])
                mode2 = mode_re.search(info[1])
                try:
                    ndx_1 = int(mode1.group(1))
                    exp_1 = int(mode1.group(2))
                    ndx_2 = int(mode2.group(1))
                    exp_2 = int(mode2.group(2))
                except AttributeError:
                    raise RuntimeError("error while parsing anharmonic frequencies: %s" % line)
                harm_freq = float(info[2])
                anharm_freq = float(info[3])
                anharm_inten = float(info[4])
                harm_inten = 0
                combinations.append(
                    (
                        ndx_1,
                        ndx_2,
                        exp_1,
                        exp_2,
                        anharm_freq,
                        anharm_inten,
                        harm_freq,
                        harm_inten,
                    )
                )
            elif reading_overtones:
                info = line.split()
                mode = mode_re.search(info[0])
                try:
                    ndx = int(mode.group(1))
                    exp = int(mode.group(2))
                except AttributeError:
                    raise RuntimeError("error while parsing overtones: %s" % line)
                harm_freq = float(info[1])
                anharm_freq = float(info[2])
                anharm_inten = float(info[3])
                harm_inten = 0
                overtones.append(
                    (
                        ndx,
                        exp,
                        anharm_freq,
                        anharm_inten,
                        harm_freq,
                        harm_inten,
                    )
                )
            elif reading_fundamentals:
                info = line.split()
                harm_freq = float(info[1])
                anharm_freq = float(info[2])
                anharm_inten = float(info[4])
                harm_inten = float(info[3])
                fundamentals.append(
                    (anharm_freq, anharm_inten, harm_freq, harm_inten)
                )

        self.anharm_data = []
        for i, mode in enumerate(
            sorted(fundamentals, key=lambda pair: pair[2])
        ):
            self.anharm_data.append(
                AnharmonicVibration(mode[0], intensity=mode[1], harmonic=self.data[i])
            )
            self.anharm_data[-1].overtones = []
            self.anharm_data[-1].combinations = dict()

        for overtone in overtones:
            ndx = len(fundamentals) - overtone[0]
            data = self.anharm_data[ndx]
            harm_data = HarmonicVibration(overtone[4], intensity=overtone[5])
            data.overtones.append(
                AnharmonicVibration(
                    overtone[2], intensity=overtone[3], harmonic=harm_data
                )
            )
        for combo in combinations:
            ndx1 = len(fundamentals) - combo[0]
            ndx2 = len(fundamentals) - combo[1]
            data = self.anharm_data[ndx1]
            harm_data = HarmonicVibration(combo[6], intensity=combo[7])
            data.combinations[ndx2] = [
                AnharmonicVibration(combo[4], intensity=combo[5], harmonic=harm_data)
            ]
    
    def parse_qchem_lines(self, lines, *args, **kwargs):
        num_head = 0
        modes = []
        for k, line in enumerate(lines):
            if "Frequency:" in line:
                ndx = 0
                for i in float_num.findall(line):
                    self.data += [HarmonicVibration(float(i))]
                    modes += [[]]
                continue

            if "Force Cnst:" in line:
                force_constants = float_num.findall(line)
                for i in range(-len(force_constants), 0, 1):
                    self.data[i].forcek = float(force_constants[i])
                continue

            if "Red. Mass:" in line:
                red_masses = float_num.findall(line)
                for i in range(-len(red_masses), 0, 1):
                    self.data[i].red_mass = float(red_masses[i])
                continue

            if "IR Intens:" in line:
                intensities = float_num.findall(line)
                for i in range(-len(intensities), 0, 1):
                    self.data[i].intensity = float(intensities[i])
                continue

            if "Raman Intens:" in line:
                intensities = float_num.findall(line)
                for i in range(-len(intensities), 0, 1):
                    self.data[i].raman_activity = float(intensities[i])
                continue

            match = re.search(r"^\s?[A-Z][a-z]?\s+(\s+[+-]?\d+\.\d+)+$", line)
            if match is None:
                continue
            ndx += 1
            values = float_num.findall(line)
            moves = np.array(values, dtype=float)
            n_moves = len(moves) // 3
            for i in range(-n_moves, 0):
                modes[i].append(
                    moves[3 * n_moves + 3 * i : 4 * n_moves + 3 * i]
                )

        for mode, data in zip(modes, self.data):
            data.vector = np.array(mode, dtype=np.float64)

    def parse_orca_lines(self, lines, *args, **kwargs):
        """parse lines of orca output related to frequency
        hpmodes is not currently used"""
        # vibrational frequencies appear as a list, one per line
        # block column 0 is the index of the mode
        # block column 1 is the frequency in 1/cm
        # skip line one b/c its just "VIBRATIONAL FREQUENCIES" with the way we got the lines
        for n, line in enumerate(lines[1:]):
            if line == "NORMAL MODES":
                break

            if "cm**-1" not in line:
                continue

            freq = line.split()[1]
            self.data += [HarmonicVibration(float(freq))]

        atoms = kwargs["atoms"]
        masses = np.array([atom.mass for atom in atoms])

        for line in lines[n:]:
            try:
                [int(x) for x in line.split()]
                break
            except ValueError:
                n += 1

        # all 3N modes are printed with six modes in each block
        # each column corresponds to one mode
        # the rows of the columns are x_1, y_1, z_1, x_2, y_2, z_2, ...
        displacements = np.zeros((len(self.data), len(self.data)))
        carryover = 0
        row = None
        columns = None
        i = n
        symmetries = []
        prev_columns = [-1]
        while i < len(lines):
            if "IR SPECTRUM" in lines[i]:
                break

            if re.search("\s+\d+\s*$", lines[i]):
                columns = [int(x) for x in re.findall("\d+", lines[i])]
                if columns[0] != prev_columns[0]:
                    try:
                        [float(x) for x in lines[i + 1].split()]
                    except ValueError:
                        sym = lines[i + 1].split()
                        symmetries.extend(sym)
                prev_columns = columns

            elif re.search("\d+\s+-?\d+\.\d+", lines[i]):
                mode_info = lines[i].split()
                row = int(mode_info[0])
                disp = [float(x) for x in mode_info[1:]]

                displacements[row][columns] = disp
            
            i += 1

        # reshape columns into Nx3 arrays
        for k, data in enumerate(self.data):
            data.vector = np.reshape(
                displacements[:, k], (len(self.data) // 3, 3)
            )
            if symmetries:
                data.symmetry = symmetries[k]
            

        # purge rotational and translational modes
        n_data = len(self.data)
        k = 0
        while k < n_data:
            if self.data[k].frequency == 0:
                del self.data[k]
                n_data -= 1
            else:
                k += 1

        intensity_start = None
        for k, line in enumerate(lines):
            if line.strip() == "IR SPECTRUM":
                order = lines[k + 1].split()
                if "Int" in order:
                    ndx = order.index("Int")
                else:
                    ndx = order.index("T**2") - 1
                intensity_start = k + 2

        # IR intensities are only printed for vibrational
        # the first column is the index of the mode
        # the second column is the frequency
        # the third is the intensity, which we read next
        if intensity_start is not None:
            t = sum([1 for mode in self.data if mode.frequency < 0])
            for line in lines[intensity_start:]:
                if not re.match(r"\s*\d+:", line):
                    continue
                ir_info = line.split()
                inten = float(ir_info[ndx])
                self.data[t].intensity = inten
                t += 1
                if t >= len(self.data):
                    break
    
            for k, line in enumerate(lines):
                if line.strip() == "RAMAN SPECTRUM":
                    t = 0
                    for line in lines[k + 1:]:
                        if not re.match(r"\s*\d+:", line):
                            continue
                        ir_info = line.split()
                        inten = float(ir_info[2])
                        self.data[t].raman_activity = inten
                        t += 1
                        if t >= len(self.data):
                            break

    def parse_psi4_lines(self, lines, *args, **kwargs):
        """parse lines of psi4 output related to frequencies
        hpmodes is not used"""
        # normal mode info appears in blocks, with up to 3 modes per block
        # at the top is the index of the normal mode
        # next is the frequency in wavenumbers (cm^-1)
        # after a line of '-----' are the normal displacements
        read_displacement = False
        modes = []
        for n, line in enumerate(lines):
            if len(line.strip()) == 0:
                read_displacement = False
                for i, data in enumerate(self.data[-nmodes:]):
                    data.vector = np.array(modes[i])

            elif read_displacement:
                info = [float(x) for x in line.split()[2:]]
                for i, mode in enumerate(modes):
                    mode.append(info[3 * i : 3 * (i + 1)])

            elif line.strip().startswith("Vibration"):
                nmodes = len(line.split()) - 1

            elif line.strip().startswith("Freq"):
                freqs = [-1 * float(x.strip("i")) if x.endswith("i") else float(x) for x in line.split()[2:]]
                for freq in freqs:
                    self.data.append(HarmonicVibration(float(freq)))

            elif line.strip().startswith("Force const"):
                force_consts = [float(x) for x in line.split()[3:]]
                for i, data in enumerate(self.data[-nmodes:]):
                    data.forcek = force_consts[i]

            elif line.strip().startswith("IR activ"):
                intensities = [float(x) for x in line.split()[3:]]
                for i, data in enumerate(self.data[-nmodes:]):
                    data.intensity = intensities[i]

            elif line.strip().startswith("Irrep"):
                # sometimes psi4 doesn't identify the irrep of a mode, so we can't
                # use line.split()
                symm = [
                    x.strip() if x.strip() else None
                    for x in [line[31:40], line[51:60], line[71:80]]
                ]
                for i, data in enumerate(self.data[-nmodes:]):
                    data.symmetry = symm[i]

            elif line.strip().startswith("----"):
                read_displacement = True
                modes = [[] for i in range(0, nmodes)]

    def sort_frequencies(self):
        self.imaginary_frequencies = []
        self.real_frequencies = []
        for data in self.data:
            freq = data.frequency
            if freq < 0:
                self.imaginary_frequencies += [freq]
            elif freq > 0:
                self.real_frequencies += [freq]
            self.by_frequency[freq] = {
                "intensity": data.intensity,
            }
            if hasattr(data, "vector"):
                # anharmonic data might not have a vector
                self.by_frequency[freq]["vector"] = data.vector
        if len(self.data) > 0:
            self.lowest_frequency = self.data[0].frequency
        else:
            self.lowest_frequency = None
        self.is_TS = True if len(self.imaginary_frequencies) == 1 else False

    def filter_data(self, signal):
        return signal.frequency > 0

    def plot_ir(
        self,
        figure,
        centers=None,
        widths=None,
        exp_data=None,
        plot_type="transmittance",
        peak_type="pseudo-voigt",
        reverse_x=True,
        y_label=None,
        point_spacing=None,
        normalize=True,
        fwhm=15.0,
        anharmonic=False,
        rotate_x_ticks=False,
        show_functions=None,
        **kwargs,
    ):
        """
        plot IR data on figure
        
        :param matplotlib.pyplot.Figure figure: matplotlib figure
        :param np.ndarray centers: array-like of float, plot is split into sections centered
            on the frequency specified by centers
            
            default is to not split into sections
        :param np.ndarray widths: array-like of float, defines the width of each section
        :param list exp_data: other data to plot
            
            should be a list of (x_data, y_data, color)
        :param bool reverse_x: if True, 0 cm^-1 will be on the right
        :param str plot_type: see Frequency.get_plot_data
        :param str peak_type: any value allowed by Frequency.get_plot_data
        :param kwargs: keywords for Frequency.get_spectrum_functions
        """

        if "intensity_attr" not in kwargs:
            intensity_attr = "intensity"
            if plot_type.lower() == "vcd":
                intensity_attr = "rotation"
            elif plot_type.lower() == "raman":
                intensity_attr = "raman_activity"
            elif plot_type.lower() == "absorbance":
                intensity_attr = "intensity"
            elif plot_type.lower() == "transmittance":
                intensity_attr = "intensity"
            else:
                self.LOG.warning("unrecognized plot type: %s\nDefaulting to absorbance" % plot_type)
            kwargs["intensity_attr"] = intensity_attr

        data_attr = "data"
        if anharmonic:
            data_attr = "anharm_data"

        functions, frequencies, intensities = self.get_spectrum_functions(
            peak_type=peak_type,
            fwhm=fwhm,
            data_attr=data_attr,
            **kwargs,
        )

        other_y_style = None
        ndx_list = None
        if show_functions is not None:
            ndx_list = [info[0] for info in show_functions]
            other_y_style = list(info[1:] for info in show_functions)

        data = self.get_plot_data(
            functions,
            frequencies,
            fwhm=fwhm,
            transmittance=plot_type.lower().startswith("transmittance"),
            peak_type=peak_type,
            point_spacing=point_spacing,
            normalize=normalize,
            show_functions=ndx_list,
        )
        if data is None:
            return

        x_values, y_values, other_y_values = data

        if y_label is None and plot_type.lower().startswith("transmittance"):
            y_label = "Transmittance (%)"
        elif y_label is None and plot_type.lower() == "absorbance":
            y_label = "Absorbance (arb.)"
        elif y_label is None and plot_type.lower() == "vcd":
            y_label = "ΔAbsorbance (arb.)"
        elif y_label is None and plot_type.lower() == "raman":
            y_label = "Activity (arb.)"

        self.plot_spectrum(
            figure,
            x_values,
            y_values,
            other_y_values=other_y_values,
            other_y_style=other_y_style,
            centers=centers,
            widths=widths,
            exp_data=exp_data,
            reverse_x=reverse_x,
            peak_type=peak_type,
            plot_type=plot_type,
            y_label=y_label,
            rotate_x_ticks=rotate_x_ticks,
        )


class ValenceExcitation(Signal):
    x_attr = "excitation_energy"
    required_attrs = (
        "rotatory_str_len", "rotatory_str_vel", "oscillator_str",
        "oscillator_str_vel", "symmetry", "multiplicity", 
        "dipole_len_vec", "dipole_vel_vec", "magnetic_mom",
    )

    @property
    def dipole_str_len(self):
        if self.oscillator_str is None:
            return None
        return self.oscillator_str / self.excitation_energy

    @property
    def dipole_str_vel(self):
        if self.oscillator_str_vel is None:
            return None
        return self.oscillator_str_vel / self.excitation_energy

    @property
    def delta_abs_len(self):
        if self.rotatory_str_len is None:
            return None
        return self.rotatory_str_len * self.excitation_energy

    @property
    def delta_abs_vel(self):
        if self.rotatory_str_vel is None:
            return None
        return self.rotatory_str_vel * self.excitation_energy


class SOCExcitation(ValenceExcitation):
    pass


class TransientExcitation(ValenceExcitation):
    x_attr = "excitation_energy"
    required_attrs = (
        "rotatory_str_len", "rotatory_str_vel", "oscillator_str",
        "oscillator_str_vel", "symmetry", "multiplicity",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ValenceExcitations(Signals):
    """for UV/vis data, primarily from TD-DFT"""
    def __init__(self, *args, **kwargs):
        self.transient_data = None
        self.spin_orbit_data = None
        super().__init__(*args, **kwargs)

    def parse_gaussian_lines(self, lines, *args, **kwargs):
        i = 0
        nrgs = []
        rotatory_str_len = []
        rotatory_str_vel = []
        oscillator_str = []
        oscillator_vel = []
        dipole_moments_len = []
        dipole_moments_vel = []
        magnetic_moments = []
        quadrupole_moments_len = []
        symmetry = []
        multiplicity = []
        while i < len(lines):
            if "Ground to excited state transition electric" in lines[i]:
                i += 2
                line = lines[i]
                while line and line.split()[0].isdigit():
                    oscillator_str.append(float(line.split()[-1]))
                    dipole_moments_len.append(
                        [float(x) for x in line.split()[1:4]]
                    )
                    i += 1
                    line = lines[i]
            elif "Ground to excited state transition velocity" in lines[i]:
                i += 2
                line = lines[i]
                while line and line.split()[0].isdigit():
                    oscillator_vel.append(float(line.split()[-1]))
                    dipole_moments_vel.append(
                        [float(x) for x in line.split()[1:4]]
                    )
                    i += 1
                    line = lines[i]
            elif "Ground to excited state transition magnetic dipole" in lines[i]:
                i += 2
                line = lines[i]
                while line and line.split()[0].isdigit():
                    magnetic_moments.append(
                        [float(x) for x in line.split()[1:4]]
                    )
                    i += 1
                    line = lines[i]
            elif "R(length)" in lines[i]:
                i += 1
                line = lines[i]
                while line and line.split()[0].isdigit():
                    rotatory_str_len.append(float(line.split()[-1]))
                    i += 1
                    line = lines[i]
            elif "R(velocity)" in lines[i]:
                i += 1
                line = lines[i]
                while line and line.split()[0].isdigit():
                    rotatory_str_vel.append(float(line.split()[-2]))
                    i += 1
                    line = lines[i]
            elif re.search(r"Excited State\s*\d+:", lines[i]):
                excitation_data = re.search(
                    r"Excited State\s*\d+:\s*([\D]+)-([\S]+)\s+-?(\d+\.\d+)",
                    lines[i],
                )
                # sometimes gaussian cannot determine the symmetry
                if excitation_data is None:
                    excitation_data = re.search(
                        r"Excited State\s*\d+:\s*[\S]+-?Sym\s+-?(\d+\.\d+)",
                        lines[i],
                    )
                    multiplicity.append(None)
                    symmetry.append(None)
                    try:
                        nrgs.append(float(excitation_data.group(1)))
                    except AttributeError:
                        raise RuntimeError("error while parsing gaussian UV/vis data:\n%s" % lines[i])
                else:
                    try:
                        multiplicity.append(excitation_data.group(1))
                        symmetry.append(excitation_data.group(2))
                        nrgs.append(float(excitation_data.group(3)))
                    except AttributeError:
                        raise RuntimeError("error while parsing gaussian UV/vis data:\n%s" % lines[i])

                i += 1
            else:
                i += 1

        for nrg, rot_len, rot_vel, osc_len, osc_vel, sym, mult, dip_vec, dip_vec_vel, mag_mom in zip(
            nrgs, rotatory_str_len, rotatory_str_vel, oscillator_str,
            oscillator_vel, symmetry, multiplicity, dipole_moments_len,
            dipole_moments_vel, magnetic_moments,
        ):
            self.data.append(
                ValenceExcitation(
                    nrg,
                    rotatory_str_len=rot_len,
                    rotatory_str_vel=rot_vel,
                    oscillator_str=osc_len,
                    oscillator_str_vel=osc_vel,
                    symmetry=sym,
                    multiplicity=mult,
                    dipole_len_vec=dip_vec,
                    dipole_vel_vec=dip_vec_vel,
                    magnetic_mom=mag_mom,
                )
            )

    def parse_orca_lines(self, lines, *args, orca_version=6, **kwargs):
        i = 0
        nrgs = []
        corr = []
        rotatory_str_len = []
        rotatory_str_vel = []
        oscillator_str = []
        oscillator_vel = []
        dipole_moments_len = []
        dipole_moments_vel = []
        magnetic_moments = []
        magnetic_velocities = []
        multiplicity = []
        mult = "Singlet"
        
        soc_nrgs = []
        soc_oscillator_str = []
        soc_oscillator_vel = []
        soc_rotatory_str_len = []
        soc_rotatory_str_vel = []

        transient_oscillator_str = []
        transient_oscillator_vel = []
        transient_rot_str = []
        transient_nrg = []
        
        while i < len(lines):
            line = lines[i]
            if "SINGLETS" in line:
                mult = "Singlet"
                i += 1
            
            elif "TRIPLETS" in line:
                mult = "Triplet"
                i += 1
            
            elif re.search("IROOT=.+?(\d+\.\d+)\seV", line):
                # could use walrus
                info = re.search("IROOT=.+?(\d+\.\d+)\seV", line)
                nrgs.append(float(info.group(1)))
                i += 1
            
            elif re.search("STATE\s*\d+:\s*E=\s*\S+\s*au\s*(-?\d+\.\d+)", line):
                info = re.search("STATE\s*\d+:\s*E=\s*\S+\s*au\s*(-?\d+\.\d+)", line)
                try:
                    nrgs.append(float(info.group(1)))
                except AttributeError:
                    raise RuntimeError("error while parsing ORCA UV/vis data: %s" % line)
                multiplicity.append(mult)
                i += 1
            
            elif (
                "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line and
                "TRANSIENT" not in line and
                "SOC" not in line and
                "SPIN ORBIT" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip() and line.strip()[0].isdigit():
                    info = line.split()
                    if orca_version <= 5:
                        if info[3] == "spin":
                            oscillator_str.append(0)
                            dipole_moments_len.append([0, 0, 0])
                        else:
                            oscillator_str.append(float(info[3]))
                            dipole_moments_len.append([float(x) for x in info[5:8]])
                    else:
                        oscillator_str.append(float(info[6]))
                        dipole_moments_len.append([float(x) for x in info[7:10]])
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
            
            elif (
                "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line and
                "TRANSIENT" not in line and
                "SPIN ORBIT" not in line and
                "SOC" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip() and line.strip()[0].isdigit():
                    info = line.split()
                    if orca_version <= 5:
                        if info[3] == "spin":
                            oscillator_vel.append(0)
                            dipole_moments_vel.append([0, 0, 0])
                        else:
                            oscillator_vel.append(float(info[3]))
                            dipole_moments_vel.append([float(x) for x in info[5:8]])
                    else:
                        oscillator_vel.append(float(info[6]))
                        dipole_moments_vel.append([float(x) for x in info[7:10]])

                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
            
            elif (
                line.endswith("CD SPECTRUM") and
                "TRANSIENT" not in line and
                "SPIN ORBIT" not in line and 
                "SOC CORRECTED" not in line
            ) or (
                line.strip() == "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
            ):
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        if info[3] == "spin":
                            rotatory_str_len.append(0)
                            magnetic_moments.append([0, 0, 0])
                        else:
                            rotatory_str_len.append(float(info[3]))
                            magnetic_moments.append([float(x) for x in info[4:7]])
                    else:
                        rotatory_str_len.append(float(info[6]))
                        magnetic_moments.append([float(x) for x in info[7:10]])
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
            
            elif (
                "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line and
                "TRANSIENT" not in line and
                "SPIN ORBIT" not in line and
                "SOC CORRECTED" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        if info[3] == "spin":
                            rotatory_str_vel.append(0)
                        else:
                            rotatory_str_vel.append(float(info[3]))
                    else:
                        rotatory_str_vel.append(float(info[6]))
                        magnetic_velocities.append([float(x) for x in info[7:10]])
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
            
            elif "TRANSIENT ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        transient_oscillator_str.append(float(info[3]))
                        transient_nrg.append(self.nm_to_ev(float(info[2])))
                    else:
                        transient_oscillator_str.append(float(info[6]))
                        transient_nrg.append(float(info[3]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
            
            elif "TRANSIENT ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        transient_oscillator_vel.append(float(info[3]))
                    else:
                        transient_oscillator_vel.append(float(info[6]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
            
            elif "TRANSIENT CD SPECTRUM" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        transient_rot_str.append(float(info[3]))
                    else:
                        transient_rot_str.append(float(info[6]))
                    i += 1
                    line = lines[i]
            
            elif "CALCULATED SOLVENT SHIFTS" in line:
                i += 8
                line = lines[i]
                while line.strip():
                    info = line.split()
                    corr.append(float(info[-1]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]
        
            elif line.startswith("Eigenvalues of the SOC matrix:"):
                i += 4
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_nrgs.append(float(info[-1]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SPIN ORBIT CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_oscillator_str.append(float(info[4]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        soc_oscillator_str.append(float(info[4]))
                    else:
                        soc_oscillator_str.append(float(info[6]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SPIN ORBIT CORRECTED ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_oscillator_vel.append(float(info[4]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if orca_version <= 5:
                        soc_oscillator_vel.append(float(info[4]))
                    else:
                        soc_oscillator_vel.append(float(info[6]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SPIN ORBIT CORRECTED CD SPECTRUM" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_rotatory_str_len.append(float(info[4]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SOC CORRECTED CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_rotatory_str_len.append(float(info[6]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            elif "SOC CORRECTED CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_rotatory_str_vel.append(float(info[6]))
                    i += 1
                    if i == len(lines):
                        break
                    line = lines[i]

            else:
                i += 1
        
        if corr:
            for i in range(0, len(nrgs)):
                nrgs[i] = corr[i]

        if not multiplicity:
            multiplicity = [None for x in nrgs]

        if not rotatory_str_vel:
            rotatory_str_vel = [None for x in rotatory_str_len]

        if not oscillator_vel:
            oscillator_vel = [None for x in oscillator_str]

        if not magnetic_velocities:
            magnetic_velocities = [None for x in oscillator_str]

        if not soc_rotatory_str_vel and soc_oscillator_str:
            soc_rotatory_str_vel = [None for x in soc_oscillator_str]

        if not dipole_moments_vel:
            dipole_moments_vel = [None for x in oscillator_str]

        for (
            nrg,
            rot_len,
            rot_vel,
            osc_len,
            osc_vel,
            dip_vec_len,
            dip_vec_vel,
            mag_mom,
            mag_vel,
            mult
        ) in zip(
            nrgs, rotatory_str_len, rotatory_str_vel, oscillator_str,
            oscillator_vel, dipole_moments_len, dipole_moments_vel,
            magnetic_moments, magnetic_velocities, multiplicity,
        ):
            self.data.append(
                ValenceExcitation(
                    nrg,
                    rotatory_str_len=rot_len,
                    rotatory_str_vel=rot_vel,
                    oscillator_str=osc_len,
                    oscillator_str_vel=osc_vel,
                    dipole_len_vec=dip_vec_len,
                    dipole_vel_vec=dip_vec_vel,
                    magnetic_mom=mag_mom,
                    magnetic_vel=mag_vel,
                    multiplicity=mult,
                )
            )

        for nrg, rot_len, osc_len, osc_vel in zip(
            transient_nrg,
            transient_rot_str,
            transient_oscillator_str,
            transient_oscillator_vel,
        ):
            if not hasattr(self, "transient_data") or not self.transient_data:
                self.transient_data = []
            self.transient_data.append(
                TransientExcitation(
                    nrg,
                    rotatory_str_len=rot_len,
                    oscillator_str=osc_len,
                    oscillator_str_vel=osc_vel,
                )
            )

        for nrg, rot_len, rot_vel, osc_len, osc_vel in zip(
            soc_nrgs,
            soc_rotatory_str_len,
            soc_rotatory_str_vel,
            soc_oscillator_str,
            soc_oscillator_vel,
        ):
            if not hasattr(self, "spin_orbit_data") or not self.spin_orbit_data:
                self.spin_orbit_data = []
            self.spin_orbit_data.append(
                SOCExcitation(
                    nrg,
                    rotatory_str_len=rot_len,
                    rotatory_str_vel=rot_vel,
                    oscillator_str=osc_len,
                    oscillator_str_vel=osc_vel,
                )
            )

    def parse_psi4_lines(self, lines, *args, **kwargs):
        symmetry = []
        energy = []
        oscillator_str = []
        oscillator_str_vel = []
        rotation_str = []
        rotation_str_vel = []
        for line in lines:
            if "->" not in line and line.split()[0].isdigit():
                info = line.split()
                symmetry.append(info[1])
                energy.append(float(info[2]))
                oscillator_str.append(float(info[6]))
                rotation_str.append(float(info[7]))
                rotation_str_vel.append(float(info[7]))
                oscillator_str_vel.append(None)
            elif re.search("\| State\s*\d+", line):
                info = re.search("(\d+\.\d+)\s*eV", line)
                try:
                    energy.append(float(info.group(1)))
                except AttributeError:
                    raise RuntimeError("error while parsing psi4 UV/vis data: %s" % line)
            elif re.search("Oscillator strength \(length", line):
                oscillator_str.append(float(line.split()[-1]))
            elif re.search("Oscillator strength \(velocity", line):
                oscillator_str_vel.append(float(line.split()[-1]))
            elif re.search("Rotational strength \(length", line):
                rotation_str.append(float(line.split()[-1]))
            elif re.search("Rotational strength \(velocity", line):
                rotation_str_vel.append(float(line.split()[-1]))
            elif line.split()[0].isdigit():
                info = line[46:].split()
                symmetry.append(line.split("(")[1].split(")")[0])
                energy.append(float(info[0]))
                oscillator_str_vel.append(float(info[2]))
                oscillator_str.append(float(info[3]))
                rotation_str.append(float(info[4]))
                rotation_str_vel.append(float(info[5]))
        
        lists = [
            symmetry, energy, oscillator_str_vel, oscillator_str,
            rotation_str, rotation_str_vel,
        ]
        
        max_list = max(lists, key=len)
        
        for l in lists:
            while len(l) < len(max_list):
                l.append(None)

        for nrg, sym, osc_v, osc, r_l, r_v in zip(
            energy, symmetry, oscillator_str_vel, oscillator_str, 
            rotation_str, rotation_str_vel,
        ):
            self.data.append(
                ValenceExcitation(
                    nrg, symmetry=sym, oscillator_str=osc, rotatory_str_len=r_l,
                    rotatory_str_vel=r_v, oscillator_str_vel=osc_v,
                )
            )

    def parse_qchem_lines(self, lines, *args, **kwargs):
        multiplicity = []
        energy = []
        oscillator_str = []
        symmetry = []
        rotation_str = []
        rotation_str_vel = []
        for line in lines:
            if re.search("Excited state\s+\d+: excitation energy", line):
                energy.append(float(line.split()[-1]))
            
            if re.search("Multiplicity:", line):
                multiplicity.append(line.split()[-1])
            
            if re.search("Strength", line):
                oscillator_str.append(float(line.split()[-1]))
                
            if re.search("Excited state\s+\d+\s*\(", line):
                info = re.search("\((\S+), (\S+)\)", line)
                try:
                    multiplicity.append(info.group(1).capitalize())
                    symmetry.append(info.group(2))
                except AttributeError:
                    raise RuntimeError("error while parsing Q-Chem UV/vis data: %s" % line)
            
            if re.search("Excitation energy:", line):
                if len(energy) > len(oscillator_str):
                    oscillator_str.append(0)
                energy.append(float(line.split()[-2]))

            if re.search("Osc. strength:", line):
                oscillator_str.append(float(line.split()[-1]))
            
            if re.search("State B:", line):
                symmetry.append(line.split("/")[-1])
            
            if re.search("Oscillator strength", line):
                oscillator_str.append(float(line.split()[-1]))

            if re.search("Energy GAP", line):
                energy.append(float(line.split()[-2]))

            if re.search("Rotatory strength, length gauge", line):
                rotation_str.append(float(line.split()[-1]))

            if re.search("Rotatory strength, velocity gauge", line):
                rotation_str_vel.append(float(line.split()[-1]))
        
        lists = [
            symmetry, energy, oscillator_str, multiplicity,
            rotation_str, rotation_str_vel,
        ]
        max_list = max(lists, key=len)
        
        for l in lists:
            while len(l) < len(max_list):
                l.append(None)

        for nrg, mult, osc, symm, rot, rot_vel in zip(
            energy, multiplicity, oscillator_str, symmetry,
            rotation_str, rotation_str_vel,
        ):
            self.data.append(
                ValenceExcitation(
                    nrg, multiplicity=mult, oscillator_str=osc, symmetry=symm,
                    rotatory_str_len=rot, rotatory_str_vel=rot_vel,
                )
            )

    @staticmethod
    def nm_to_ev(x):
        """convert x nm to eV"""
        if isinstance(x, float):
            return PHYSICAL.SPEED_OF_LIGHT * 1e7 * PHYSICAL.PLANCK * UNIT.JOULE_TO_EV / x

        x = np.array(x)
        ndx = np.where(x > 0)
        return PHYSICAL.SPEED_OF_LIGHT * 1e7 * PHYSICAL.PLANCK * UNIT.JOULE_TO_EV / x[ndx], ndx

    @staticmethod
    def ev_to_nm(x):
        """convert x eV to nm"""
        if isinstance(x, float):
            return PHYSICAL.SPEED_OF_LIGHT * 1e7 * PHYSICAL.PLANCK * UNIT.JOULE_TO_EV / x

        x = np.array(x)
        ndx = np.where(x > 0)
        return PHYSICAL.SPEED_OF_LIGHT * 1e7 * PHYSICAL.PLANCK * UNIT.JOULE_TO_EV / x[ndx], ndx

    def plot_uv_vis(
        self,
        figure,
        centers=None,
        widths=None,
        exp_data=None,
        plot_type="uv-vis-veloctiy",
        peak_type="gaussian",
        reverse_x=False,
        y_label=None,
        point_spacing=None,
        normalize=True,
        fwhm=15.0,
        units="nm",
        rotate_x_ticks=False,
        show_functions=None,
        transient=False,
        **kwargs,
    ):
        """
        plot UV/vis data on figure
        
        :param matplotlib.pyplot.Figure figure: matplotlib figure
        :param np.ndarray centers: array-like of float, plot is split into sections centered
            on the frequency specified by centers
            
            default is to not split into sections
        :param np.ndarray widths: array-like of float, defines the width of each section
        :param list exp_data: other data to plot
            
            should be a list of (x_data, y_data, color)
        :param bool reverse_x: if True, 0 cm^-1 will be on the right
        :param str plot_type: what type of data to plot: uv-vis,
            uv-vis-velocity, ecd, ecd-velocity, transmittance, transmittance-velocity
        :param str peak_type: any value allowed by :func:`Signals.get_plot_data <AaronTools.spectra.Signals.get_plot_data>`
        :param kwargs: keywords for :func:`Signals.get_spectrum_functions <AaronTools.spectra.Signals.get_spectrum_functions>`
        
        most other input is passed to :func:`Signals.plot_spectrum <AaronTools.spectra.Signals.plot_spectrum>`
        """

        data_attr = "data"
        if transient:
            data_attr = "transient_data"
        
        if "intensity_attr" not in kwargs:
            intensity_attr = "oscillator_str"
            if plot_type.lower() == "uv-vis-velocity":
                intensity_attr = "oscillator_str_vel"
            elif plot_type.lower() == "transmittance-velocity":
                intensity_attr = "oscillator_str_vel"
            elif plot_type.lower() == "transmittance":
                intensity_attr = "oscillator_str"
            elif plot_type.lower() == "uv-vis":
                intensity_attr = "oscillator_str"
            elif plot_type.lower() == "ecd":
                intensity_attr = "delta_abs_len"
            elif plot_type.lower() == "ecd-velocity":
                intensity_attr = "delta_abs_vel"
            else:
                self.LOG.warning("unrecognized plot type: %s\nDefaulting to uv-vis" % plot_type)
            kwargs["intensity_attr"] = intensity_attr


        if getattr(self.data[0], kwargs["intensity_attr"]) is None:
            raise RuntimeError("no data was parsed for %s" % kwargs["intensity_attr"])

        if not centers and units == "nm":
            data_list = getattr(self, data_attr)
            data_min = None
            data_max = None
            for data in data_list:
                wavelength = self.ev_to_nm(data.excitation_energy)
                if data_min is None or wavelength < data_min:
                    data_min = wavelength
                if data_max is None or wavelength > data_max:
                    data_max = wavelength
            centers = [(data_min + data_max) / 2]
            widths = [1.5 * (data_max - data_min)]
            if widths[0] == 0:
                widths = [4 * fwhm]

        change_x_unit_func = None
        x_label = "wavelength (nm)"
        change_x_unit_func = self.ev_to_nm
        if units == "eV":
            change_x_unit_func = None
            x_label = r"$h\nu$ (eV)"

        functions, energies, intensities = self.get_spectrum_functions(
            peak_type=peak_type,
            fwhm=fwhm,
            data_attr=data_attr,
            **kwargs,
        )

        other_y_style = None
        ndx_list = None
        if show_functions is not None:
            ndx_list = [info[0] for info in show_functions]
            other_y_style = list(info[1:] for info in show_functions)

        data = self.get_plot_data(
            functions,
            energies,
            fwhm=fwhm,
            transmittance=plot_type.lower().startswith("transmittance"),
            peak_type=peak_type,
            point_spacing=point_spacing,
            change_x_unit_func=change_x_unit_func,
            normalize=normalize,
            show_functions=ndx_list,
        )
        if data is None:
            return

        x_values, y_values, other_y_values = data

        if y_label is None and plot_type.lower().startswith("transmittance"):
            y_label = "Transmittance (%)"
        elif y_label is None and "uv-vis" in plot_type.lower():
            y_label = "Aborptivity (arb.)"
        elif y_label is None and "ecd" in plot_type.lower():
            y_label = "ΔAborptivity (arb.)"

        self.plot_spectrum(
            figure,
            x_values,
            y_values,
            other_y_values=other_y_values,
            other_y_style=other_y_style,
            centers=centers,
            widths=widths,
            exp_data=exp_data,
            reverse_x=reverse_x,
            peak_type=peak_type,
            plot_type=plot_type,
            x_label=x_label,
            y_label=y_label,
            rotate_x_ticks=rotate_x_ticks,
        )


class Shift(Signal):
    x_attr = "shift"
    required_attrs = ("ndx", "element")


class NMR(Signals):
    x_label = "shift (ppm)"
    
    def __init__(self, *args, n_atoms=0, coupling=None, **kwargs):
        self.coupling = {}
        if coupling is not None:
            self.coupling = coupling
        self.n_atoms = n_atoms
        super().__init__(*args, **kwargs)
    
    def parse_orca_lines(self, lines, *args, **kwargs):
        nuc = []
        element = None
        ndx = None
        ndx_a = None
        ndx_b = None
        ele_a = None
        ele_a = None
        nuc_regex = re.compile("Nucleus\s*(\d+)([A-Za-z]+)")
        coupl_regex = re.compile(
            "NUCLEUS A =\s*([A-Za-z]+)\s*(\d+)\s*NUCLEUS B =\s*([A-Za-z]+)\s*(\d+)"
        )
        for line in lines:
            if nuc_regex.search(line):
                # could use walrus
                shift_info = nuc_regex.search(line)
                ndx = int(shift_info.group(1))
                element = shift_info.group(2)
            elif line.startswith(" Total") and "iso=" in line:
                if ndx_a is None:
                    shift = float(line.split()[-1])
                    self.data.append(Shift(shift, ndx=ndx, element=element, intensity=1))
                else:
                    # orca 5 coupling
                    coupling = float(line.split()[-1])
                    self.coupling[ndx_a][ndx_b] = coupling
                    self.coupling[ndx_b][ndx_a] = coupling
            elif line.startswith(" NUCLEUS A ="):
                coupl_info = coupl_regex.search(line)
                try:
                    ele_a = coupl_info.group(1)
                    ndx_a = int(coupl_info.group(2))
                    ele_b = coupl_info.group(3)
                    ndx_b = int(coupl_info.group(4))
                except AttributeError:
                    raise RuntimeError("error while parsing ORCA NMR data: %s" % line)
                self.coupling.setdefault(ndx_a, {})
                self.coupling.setdefault(ndx_b, {})
            # orca 6 coupling
            elif re.search("J\[\d+,\d+\]\(Total\)", line):
                coupling = float(line.split()[-1])
                self.coupling[ndx_a][ndx_b] = coupling
                self.coupling[ndx_b][ndx_a] = coupling

    def parse_gaussian_lines(self, lines, *args, **kwargs):
        nuc = []
        element = None
        ndx = None
        ndx_a = None
        ndx_b = None
        ele_a = None
        ele_a = None
        nuc_regex = re.compile("(\d+)\s*([A-Za-z]+)\s*Isotropic =\s*(-?\d+\.\d+)\s*Anisotropy")
        i = 0
        while i < len(lines):
            line = lines[i]
            if nuc_regex.search(line):
                # could use walrus
                shift_info = nuc_regex.search(line)
                ndx = int(shift_info.group(1)) - 1
                element = shift_info.group(2)
                iso = float(shift_info.group(3))
                self.data.append(
                    Shift(iso, ndx=ndx, element=element, intensity=1)
                )

            elif "Total nuclear spin-spin coupling J" in line:
                i += 1
                while i < len(lines):
                    line = lines[i]
                    if "D" not in line:
                        header = [int(x) - 1 for x in line.split()]
                        i += 1
                        continue
                    info = line.split()
                    try:
                        m = int(info[0]) - 1
                    except ValueError:
                        break
                    self.coupling.setdefault(m, {})
                    for j, n in zip(info[1:], header):
                        j = float(j.replace("D", "e"))
                        if abs(j) != 0.0:
                            self.coupling[m][n] = j

                    i += 1
            
            i += 1

    @classmethod
    def get_mixed_signals(
        cls,
        signal_groups,
        weights,
        data_attr="data",
        **kwargs,
    ):
        """
        get signals for a mixture of components or conformers
        
        :param list(Signal) signal_groups: list of Signals() instances or list of lists of Signals()
            
            a list of Signals() is a group of conformers

        :param iterable weights: weights for each conformer, organized according to signal_groups
        :param str data_attr: attribute of Signals() for data
        :param kwargs: passed to cls.__init__, along with a new list of data
        """

        # TODO: warn about data mismatch
        new_data = []
        new_coupling = {}
        for signals, weight in zip(signal_groups, weights):
            data = getattr(signals, data_attr)
            for i, d in enumerate(data):
                x_val = getattr(d, d.x_attr)
                vals = d.__dict__
                data_cls = d.__class__
                new_vals = dict()
                try:
                    new_data[i]
                except IndexError:
                    new_data.append(d.__class__(0, **d.__dict__))
 
                for key, item in vals.items():
                    if isinstance(item, float):
                        setattr(new_data[i], key, getattr(new_data[i], key) + weight * item)
                    else:
                        setattr(new_data[i], key, item)

                for j, d2 in enumerate(data[:i]):
                    try:
                        new_coupling.setdefault(d.ndx, {})
                        new_coupling[d.ndx].setdefault(d2.ndx, 0)
                        new_coupling[d.ndx][d2.ndx] += (
                            weight * signals.coupling[d.ndx][d2.ndx]
                        )
                    except KeyError:
                        pass

        out = cls(new_data, **kwargs)
        out.coupling = new_coupling
        out.n_atoms = signal_groups[0].n_atoms
        return out
    
    def get_spectrum_functions(
        self,
        fwhm=2.5,
        peak_type="lorentzian",
        voigt_mixing=0.5,
        scalar_scale=0.0,
        linear_scale=-1.0,
        quadratic_scale=0.0,
        intensity_attr="intensity",
        data_attr="data",
        pulse_frequency=60.0,
        equivalent_nuclei=None,
        geometry=None,
        graph=None,
        coupling_threshold=0.0,
        element="H",
        couple_with=COMMONLY_ODD_ISOTOPES,
        shifts_only=False,
    ):
        """
        returns a list of functions that can be evaluated to
        produce a spectrum
        
        :param float fwhm: full width at half max of each peak (Hz)
        :param str peak_type: gaussian, lorentzian, pseudo-voigt, or delta
        :param float voigt_mixing: ratio of pseudo-voigt that is gaussian
        :param float scalar_scale: shift x data
        :param float linear_scale: scale x data
        :param float quadratic_scale: scale x data
            x' = scalar_scale - linear_scale * x - quadratic_scale * x^2
        :param str intensity_attr: attribute of Signal used for the intensity
            of that signal
        :param str data_attr: attribute of self for the list of Signal()
        :param float pulse_frequency: magnet pulse frequency (megahertz)
        :param list equivalent_nuclei: list of lists of equivalent nuclei
        :param Geometry geometry: used to determine equivalent nuclei if equivalent_nuclei is not given
        :param list graph: used to determine equivalent nuclei if given. See :func:`Geometry.get_graph <AaronTools.geometry.Geometry.get_graph>` for example graphs.
        :param float coupling_threshold: coupling threshold for whether or not to split into multiplet
        :param list element: include signals with these specified elements
        :param list couple_with: list of element symbols to use when determining which nuclei cause splitting
        :param bool shifts_only: only determine centers of shifts and not complete splitting pattern
        
        For nuclei to be coupling-equivalent, the must be equivalent nuclei, the
        same number of bonds away from the coupled nucleus, and the sign of their
        coupling constant must be the same. 
        """
        data = getattr(self, data_attr)
        x_attr = data[0].x_attr
        if isinstance(element, str):
            element = set([element])

        if couple_with == "all" and geometry:
            couple_with = set(geometry.elements)
        elif isinstance(couple_with, str) and couple_with.lower() == "none":
            couple_with = []

        if couple_with and not any(signal.element in couple_with for signal in self.data):
            from warnings import warn
            warn("coupling requested for %s, but these signals are not present" % repr(couple_with))
        
        if geometry is not None and graph is None:
            graph = geometry.get_graph()
        
        # determine equivalent nuclei for shifts
        if equivalent_nuclei is None and geometry is not None:
            # I don't know why, but not using invariants is often more reliable
            ranks = geometry.canonical_rank(break_ties=False, invariant=False)
            rank_map = {}
            equivalent_nuclei = []
            i = 0
            while ranks:
                rank = ranks.pop(0)
                try:
                    pos = rank_map[rank]
                except KeyError:
                    pos = len(equivalent_nuclei)
                    rank_map[rank] = pos
                    equivalent_nuclei.append([])
                equivalent_nuclei[pos].append(i)
                i += 1

        elif equivalent_nuclei is None and geometry is None:
            equivalent_nuclei = [[shift.ndx] for shift in data]

        new_data = []
        for group in equivalent_nuclei:
            valid_element = False
            average_shift = 0
            intensity = 0
            for shift in data:
                if shift.ndx in group:
                    valid_element = shift.element in element
                    if not valid_element:
                        break
                    average_shift += shift.shift / len(group)
                    intensity += shift.intensity
            if not valid_element:
                continue
            if intensity == 0:
                continue
            intensities = [intensity]
            centers = [average_shift]
            # determine splitting pattern based on how many
            # equivalent nuclei are the same numer of bonds away
            for group_b in equivalent_nuclei:
                splits = dict()
                split_count = dict()
                
                if group_b is group:
                    continue
                
                for nuc_a in group:
                    for nuc_b in group_b:
                        signal_b = [d for d in data if d.ndx == nuc_b]
                        if not signal_b:
                            continue
                        signal_b = signal_b[0]
                        if signal_b.element not in couple_with:
                            continue
                        if graph is None:
                            d = 0
                        else:
                            d = shortest_path(graph, nuc_a, nuc_b)
                            if d is not None:
                                d = len(d)
                        splits.setdefault(d, {"+": [], "-": []})
                        split_count.setdefault(d, {"+": 0, "-": 0})
                        try:
                            if self.coupling[nuc_a][nuc_b] < 0:
                                splits[d]["-"].append(self.coupling[nuc_a][nuc_b])
                                split_count[d]["-"] += 1
                            else:
                                splits[d]["+"].append(self.coupling[nuc_a][nuc_b])
                                split_count[d]["+"] += 1

                        except KeyError:
                            try:
                                if self.coupling[nuc_b][nuc_a] < 0:
                                    splits[d]["-"].append(self.coupling[nuc_b][nuc_a])
                                    split_count[d]["-"] += 1
                                else:
                                    splits[d]["+"].append(self.coupling[nuc_b][nuc_a])
                                    split_count[d]["+"] += 1
                            except KeyError:
                                pass

                for d in split_count:
                    split_count[d] = {sign: int(split_count[d][sign] / len(group)) for sign in split_count[d]}

                for d in split_count:
                    for sign in ["+", "-"]:
                        if not split_count[d][sign]:
                            continue
                        j = sum([x for x in splits[d][sign]]) / len(splits[d][sign])
                        if abs(j) <= coupling_threshold:
                            continue
                        if abs(j) <= 0:
                            continue
                        j /= pulse_frequency
                        pattern = pascals_triangle(split_count[d][sign])
                        pattern = [x / sum(pattern) for x in pattern]
                        new_split_intensities = []
                        new_split_positions = []
                        for position, intensity in zip(centers, intensities):
                            for i, ratio in enumerate(pattern):
                                i -= len(pattern) // 2
                                if len(pattern) % 2 == 0:
                                    i += 0.5
                                new_split_intensities.append(ratio * intensity)
                                new_split_positions.append(i * abs(j) + position)
                        intensities = new_split_intensities
                        centers = new_split_positions
            
            for x, y in zip(centers, intensities):
                new_data.append(
                    Shift(x, intensity=y, origin=average_shift, ndx=group)
                )
        
        data = new_data

        if not data:
            raise RuntimeError("no shifts for the specified element: %s" % element)
        
        if shifts_only:
            return data
        
        # scale x positions
        if not data[0].nested:
            x_positions = np.array(
                [getattr(d, x_attr) for d in data if self.filter_data(d)]
            )
    
            intensities = [
                getattr(d, intensity_attr) for d in data if self.filter_data(d)
            ]
        else:
            x_positions = []
            intensities = []
            x_positions.extend(
                [getattr(d, x_attr) for d in data if self.filter_data(d)]
            )
            intensities.extend(
                [getattr(d, intensity_attr) for d in data if self.filter_data(d)]
            )
            for nest in data[0].nested:
                for d in data:
                    nest_attr = getattr(d, nest)
                    if isinstance(nest_attr, dict):
                        for value in nest_attr.values():
                            if hasattr(value, "__iter__"):
                                for item in value:
                                    x_positions.append(getattr(item, x_attr))
                                    intensities.append(getattr(item, intensity_attr))
                            else:
                                x_positions.append(getattr(value, x_attr))
                                intensities.append(getattr(value, intensity_attr))
                    elif hasattr(nest_attr, "__iter__"):
                        for item in nest_attr:
                            x_positions.append(getattr(item, x_attr))
                            intensities.append(getattr(item, intensity_attr))
                    else:
                        x_positions.append(getattr(nest_attr, x_attr))
                        intensities.append(getattr(nest_attr, intensity_attr))
                
            x_positions = np.array(x_positions)

        x_positions = scalar_scale + linear_scale * x_positions + quadratic_scale * x_positions ** 2

        e_factor = -4 * np.log(2) / fwhm ** 2
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        functions = []

        for x_pos, intensity in zip(x_positions, intensities):
            if intensity is not None:
                if peak_type.lower() == "gaussian":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * np.exp(e_factor * (x - x0) ** 2)
                        * fwhm / (2 * np.sqrt(2 * np.log(2)))
                    )

                elif peak_type.lower() == "lorentzian":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * (
                            0.5 * fwhm
                            / (np.pi * ((x - x0) ** 2 + (0.5 * fwhm) ** 2))
                        )
                    )

                elif peak_type.lower() == "pseudo-voigt":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * (
                            (1 - voigt_mixing)
                            * (
                                (0.5 * fwhm) ** 2
                                / (((x - x0) ** 2 + (0.5 * fwhm) ** 2))
                            )
                            + voigt_mixing
                            * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
                        )
                    )

                elif peak_type.lower() == "delta":
                    functions.append(
                        lambda x, x0=x_pos, inten=intensity: inten
                        * int(x == x0)
                    )

        return functions, x_positions, intensities

    def plot_nmr(
        self,
        figure,
        centers=None,
        widths=None,
        exp_data=None,
        plot_type="nmr",
        peak_type="lorentzian",
        reverse_x=True,
        y_label=None,
        point_spacing=None,
        normalize=False,
        fwhm=2.5,
        rotate_x_ticks=False,
        show_functions=None,
        pulse_frequency=60.0,
        scalar_scale=0.0,
        linear_scale=-1.0,
        quadratic_scale=0.0,
        **kwargs,
    ):
        """
        plot NMR data on figure
        
        :param matplotlib.pyplot.Figure figure: matplotlib figure
        :param np.ndarray centers: array-like of float, plot is split into sections centered
            on the frequency specified by centers
            
            default is to not split into sections
        :param np.ndarray widths: array-like of float, defines the width of each section
        :param list exp_data: other data to plot
            
            should be a list of (x_data, y_data, color)
        :param bool reverse_x: if True, 0 cm^-1 will be on the right
        :param str plot_type: see :func:`Signals.get_plot_data <AaronTools.spectra.Signals.get_plot_data>`
        :param str peak_type: any value allowed by :func:`Signals.get_plot_data <AaronTools.spectra.Signals.get_plot_data>`
        :param float pulse_frequency: pulse frequency in MHz
        :param kwargs: keywords for :func:`NMR.get_spectrum_functions <AaronTools.spectra.NMR.get_spectrum_functions>`
        
        other keyword arguments are passed to :func:`NMR.get_spectrum_functions <AaronTools.spectra.NMR.get_spectrum_functions>`
        or :func:`Signals.plot_spectrum <AaronTools.spectra.Signals.plot_spectrum>`
        """

        if "intensity_attr" not in kwargs:
            kwargs["intensity_attr"] = "intensity"

        data_attr = "data"
        
        fwhm /= pulse_frequency

        functions, shifts, intensities = self.get_spectrum_functions(
            peak_type=peak_type,
            fwhm=fwhm,
            data_attr=data_attr,
            pulse_frequency=pulse_frequency,
            scalar_scale=scalar_scale,
            linear_scale=linear_scale,
            quadratic_scale=quadratic_scale,
            **kwargs,
        )

        other_y_style = None
        func_list = None
        if show_functions is not None:
            nmr_list = [
                info[0].get_spectrum_functions(
                    peak_type=peak_type,
                    fwhm=fwhm,
                    data_attr=data_attr,
                    pulse_frequency=pulse_frequency,
                    scalar_scale=scalar_scale,
                    linear_scale=linear_scale,
                    quadratic_scale=quadratic_scale,
                    **kwargs,
                ) for info in show_functions
            ]
            func_list = [x[0] for x in nmr_list]
            for x in nmr_list:
                shifts = np.concatenate((shifts, x[1]), axis=None)
            other_y_style = list(info[1:] for info in show_functions)

        data = self.get_plot_data(
            functions,
            shifts,
            fwhm=fwhm,
            transmittance=False,
            peak_type=peak_type,
            point_spacing=point_spacing,
            normalize=normalize,
            show_functions=func_list,
        )
        if data is None:
            return

        x_values, y_values, other_y_values = data

        if y_label is None:
            y_label = "intensity (arb.)"

        self.plot_spectrum(
            figure,
            x_values,
            y_values,
            other_y_values=other_y_values,
            other_y_style=other_y_style,
            centers=centers,
            widths=widths,
            exp_data=exp_data,
            reverse_x=reverse_x,
            peak_type=peak_type,
            plot_type=plot_type,
            y_label=y_label,
            x_label="shift (ppm)",
            rotate_x_ticks=rotate_x_ticks,
        )
