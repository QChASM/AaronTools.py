"""classes for storing signal data and plotting various spectra (IR, UV/vis, etc.)"""
import re

import numpy as np

from AaronTools import addlogger
from AaronTools.const import UNIT, PHYSICAL
from AaronTools.utils.utils import float_num


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


@addlogger
class Signals:
    """
    parent class for storing data for different signals in the
    spectrum and plotting a simulated spectrum
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
            if point_spacing is not None and peak_type.lower():
                x_values = []
                x = -point_spacing
                stop = max(signal_centers)
                if peak_type.lower() != "delta":
                    stop += 5 * fwhm
                while x < stop:
                    x += point_spacing
                    x_values.append(x)
            
                x_values = np.array(x_values)
            
            else:
                x_values = np.linspace(
                    0,
                    max(signal_centers) - 10 * fwhm,
                    num=100
                ).tolist()
            
                for freq in signal_centers:
                    x_values.extend(
                        np.linspace(
                            max(freq - (7.5 * fwhm), 0),
                            freq + (7.5 * fwhm),
                            num=75,
                        ).tolist()
                    )
                    x_values.append(freq)

                if not point_spacing:
                    x_values = np.array(list(set(x_values)))
                    x_values.sort()

            y_values = np.sum([f(x_values) for f in functions], axis=0)
            
            if show_functions:
                for (ndx1, ndx2) in show_functions:
                    other_y_list.append(
                        np.sum(
                            [f(x_values) for f in functions[ndx1: ndx2]],
                            axis=0,
                        )
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
        except (IndexError, AttributeError):
            # some software can compute frequencies with a user-supplied
            # hessian, so it never prints the structure
            # ORCA can do this. It will print the input structure, but
            # we don't parse that
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
            if "Frequencies" in line and (
                (hpmodes and "---" in line) or ("--" in line and not hpmodes)
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
                match = re.search(
                    r"^\s+\d+\s+\d+\s+\d+(\s+[+-]?\d+\.\d+)+$", line
                )
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
                match = re.search(r"^\s+\d+\s+\d+(\s+[+-]?\d+\.\d+)+$", line)
                if match is None:
                    continue
                values = float_num.findall(line)
                atom = int(values[0]) - 1
                moves = np.array(values[2:], dtype=np.float)
                n_moves = len(moves) // 3
                for i in range(-n_moves, 0):
                    modes[i].append(
                        moves[3 * n_moves + 3 * i : 4 * n_moves + 3 * i]
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
                ndx_1 = int(mode1.group(1))
                exp_1 = int(mode1.group(2))
                ndx_2 = int(mode2.group(1))
                exp_2 = int(mode2.group(2))
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
                ndx = int(mode.group(1))
                exp = int(mode.group(2))
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
            moves = np.array(values, dtype=np.float)
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

            if line.startswith("Scaling"):
                continue

            freq = line.split()[1]
            self.data += [HarmonicVibration(float(freq))]

        atoms = kwargs["atoms"]
        masses = np.array([atom.mass for atom in atoms])

        # all 3N modes are printed with six modes in each block
        # each column corresponds to one mode
        # the rows of the columns are x_1, y_1, z_1, x_2, y_2, z_2, ...
        displacements = np.zeros((len(self.data), len(self.data)))
        carryover = 0
        start = 0
        stop = 6
        for i, line in enumerate(lines[n + 2 :]):
            if "IR SPECTRUM" in line:
                break

            if i % (len(self.data) + 1) == 0:
                carryover = i // (len(self.data) + 1)
                start = 6 * carryover
                stop = start + 6
                continue

            ndx = (i % (len(self.data) + 1)) - 1
            mode_info = line.split()[1:]

            displacements[ndx][start:stop] = [float(x) for x in mode_info]

        # reshape columns into Nx3 arrays
        for k, data in enumerate(self.data):
            data.vector = np.reshape(
                displacements[:, k], (len(self.data) // 3, 3)
            )
            

        # purge rotational and translational modes
        n_data = len(self.data)
        k = 0
        while k < n_data:
            if self.data[k].frequency == 0:
                del self.data[k]
                n_data -= 1
            else:
                k += 1

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
                    nrgs.append(float(excitation_data.group(1)))
                else:
                    multiplicity.append(excitation_data.group(1))
                    symmetry.append(excitation_data.group(2))
                    nrgs.append(float(excitation_data.group(3)))
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

    def parse_orca_lines(self, lines, *args, **kwargs):
        i = 0
        nrgs = []
        corr = []
        rotatory_str_len = []
        rotatory_str_vel = []
        oscillator_str = []
        oscillator_vel = []
        multiplicity = []
        mult = "Singlet"
        
        soc_nrgs = []
        soc_oscillator_str = []
        soc_oscillator_vel = []
        soc_rotatory_str_len = []

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
                info = re.search("IROOT=.+?(\d+\.\d+)\seV", line)
                nrgs.append(float(info.group(1)))
                i += 1
            elif line.startswith("STATE"):
                info = re.search("STATE\s*\d+:\s*E=\s*\S+\s*au\s*(-?\d+\.\d+)", line)
                nrgs.append(float(info.group(1)))
                multiplicity.append(mult)
                i += 1
            elif (
                "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line and
                "TRANSIENT" not in line and
                "SOC" not in line and "SPIN ORBIT" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if info[3] == "spin":
                        oscillator_str.append(0)
                    else:
                        oscillator_str.append(float(info[3]))
                    i += 1
                    line = lines[i]
            elif (
                "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line and
                "TRANSIENT" not in line and
                "SPIN ORBIT" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if info[3] == "spin":
                        oscillator_vel.append(0)
                    else:
                        oscillator_vel.append(float(info[3]))
                    i += 1
                    line = lines[i]
            elif (
                line.endswith("CD SPECTRUM") and
                "TRANSIENT" not in line and
                "SPIN ORBIT" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if info[3] == "spin":
                        rotatory_str_len.append(0)
                    else:
                        rotatory_str_len.append(float(info[3]))
                    i += 1
                    line = lines[i]
            elif (
                "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line and
                "TRANSIENT" not in line and
                "SPIN ORBIT" not in line
            ):
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    if info[3] == "spin":
                        rotatory_str_vel.append(0)
                    else:
                        rotatory_str_vel.append(float(info[3]))
                    i += 1
                    line = lines[i]
            elif "TRANSIENT ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    transient_oscillator_str.append(float(info[3]))
                    transient_nrg.append(self.nm_to_ev(float(info[2])))
                    i += 1
                    line = lines[i]
            elif "TRANSIENT ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    transient_oscillator_vel.append(float(info[3]))
                    i += 1
                    line = lines[i]
            elif "TRANSIENT CD SPECTRUM" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    transient_rot_str.append(float(info[3]))
                    i += 1
                    line = lines[i]
            elif "CALCULATED SOLVENT SHIFTS" in line:
                i += 8
                line = lines[i]
                while line.strip():
                    info = line.split()
                    corr.append(float(info[-1]))
                    i += 1
                    line = lines[i]
        
            elif line.startswith("Eigenvalues of the SOC matrix:"):
                i += 4
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_nrgs.append(float(info[-1]))
                    i += 1
                    line = lines[i]

            elif "SPIN ORBIT CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_oscillator_str.append(float(info[4]))
                    i += 1
                    line = lines[i]

            elif "SPIN ORBIT CORRECTED ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_oscillator_vel.append(float(info[4]))
                    i += 1
                    line = lines[i]

            elif "SPIN ORBIT CORRECTED CD SPECTRUM" in line:
                i += 5
                line = lines[i]
                while line.strip():
                    info = line.split()
                    soc_rotatory_str_len.append(float(info[4]))
                    i += 1
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

        for nrg, rot_len, rot_vel, osc_len, osc_vel, mult in zip(
            nrgs, rotatory_str_len, rotatory_str_vel, oscillator_str,
            oscillator_vel, multiplicity,
        ):
            self.data.append(
                ValenceExcitation(
                    nrg, rotatory_str_len=rot_len,
                    rotatory_str_vel=rot_vel, oscillator_str=osc_len,
                    oscillator_str_vel=osc_vel, multiplicity=mult,
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

        for nrg, rot_len, osc_len, osc_vel in zip(
            soc_nrgs,
            soc_rotatory_str_len,
            soc_oscillator_str,
            soc_oscillator_vel,
        ):
            if not hasattr(self, "spin_orbit_data") or not self.spin_orbit_data:
                self.spin_orbit_data = []
            self.spin_orbit_data.append(
                SOCExcitation(
                    nrg,
                    rotatory_str_len=rot_len,
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
                energy.append(float(info.group(1)))
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
                multiplicity.append(info.group(1).capitalize())
                symmetry.append(info.group(2))
            
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
