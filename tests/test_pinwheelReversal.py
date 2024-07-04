""" This file tests the correspondence between experiment and simulation for a
    two-dimensional pinwheel artifical spin ice, which is initialized in a uniform
    state and reversed using an external field. Ideally, the reversal should take place
    in two steps, roughly corresponding to two 90° rotations.
"""
import os
os.environ["HOTSPICE_USE_GPU"] = "False"

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Iterable, Literal

from context import hotspice
xp = hotspice.xp


class PinwheelReversalTest:
    def __init__(self, size: int = 50, m_perp_factor: float = 1, scheme: Literal['Néel', 'Metropolis'] = 'Néel', E_B: float = None, E_B_std: float = 0.05, dilution: float = 0):
        """ This test attempts to reproduce
                "Superferromagnetism and Domain-Wall Topologies in Artificial “Pinwheel” Spin Ice"
                by Y. Li, G. W. Paterson, G. M. Macauley et al. ACS Nano 2019 13 (2), 2213-2222, doi:10.1021/acsnano.8b08884
            using various approximations: using (or not using) the exact energy of the metastable perpendicularly magnetized state,
            or using dipoles vs. monopoles, and whether we use Néel or Metropolis (though that should not matter at all).
            
            A system size of 50 gives the exact same geometry as in the paper, but with the x- and y-axes flipped.
            But the paper also defines their angle clockwise from the y-axis, which thus corresponds to our definition of the angle.

            @param `scheme` ['Néel'|'Metropolis'] ('Néel'): The update scheme to use.
        """
        ## Basic system: 50x50 system of 470x170x10nm islands, 420nm center-to-center NN spacing
        l, d, t, Msat = 470e-9, 170e-9, 10e-9, 800e3
        self.moment = Msat*((d*d*np.pi/4) + (l-d)*d)*t # [Am²] magnetic moment of a single stadium-shaped magnet
        self.a = 420e-9*math.sqrt(2) # [m] Lattice spacing (for 420nm center-to-center NN spacing in Pinwheel)
        self.T = 300 # [K] Assume room temperature
        self.size = size - (size % 2) # Edge length of simulation as a number of cells, always multiple of 2
        
        self.E_B = hotspice.utils.eV_to_J(71) if E_B is None else E_B # [J] Energy barrier, calculated with mumax³ for stadium-shaped 470x170x10nm islands
        self.E_B_std = E_B_std
        self.dilution = dilution
        self.scheme = scheme
        self.m_perp_factor = m_perp_factor
        
        ## CREATE MM
        self.mm = hotspice.ASI.IP_Pinwheel(self.a, self.size, moment=self.moment, T=self.T, PBC=False,
                                           m_perp_factor=self.m_perp_factor, params=hotspice.SimParams(UPDATE_SCHEME=self.scheme))
        self.mm.m[xp.where(xp.random.random(size=self.mm.m.shape) < self.dilution)] = 0
        self.mm.E_B = self.E_B*xp.random.normal(1, self.E_B_std, size=self.mm.E_B.shape)
    
    @property
    def parameters(self):
        return {
            'E_B': self.E_B,
            'E_B_std': self.E_B_std,
            'dilution': self.dilution,
            'm_perp_factor': self.m_perp_factor,
            'scheme': self.mm.params.UPDATE_SCHEME,
            'size': self.size,
            'fixed_values': {
                'l': 470e-9, 'd': 170e-9, 't': 10e-9, 'Msat': 800e3,
                'moment': self.moment,
                'a': self.a,
                'T': self.T
            }
        }
    
    def hysteresis(self, H_angle: float = 0, half: bool = True, save: bool|Path = False, **kwargs):
        ''' If specified (class `Path`), `save` should be the DIRECTORY to put the summary in. '''
        ## Clip angle to range [-45°, 135°] due to symmetry in the Pinwheel ASI
        H_angle = (H_angle + 45) % 360 - 45 # Put angles between -45° and 315°
        H_angle = H_angle if H_angle <= 135 else 135 - (H_angle - 135) # Since there is a mirror symmetry about 135° in the system.
        ## Run hysteresis
        hysteresis = Hysteresis(self.mm, H_angle=H_angle, half=half, **kwargs)
        ## Save, if so desired
        if save:
            parameters = self.parameters | {"H_angle": hysteresis.H_angle, "half": hysteresis.half}
            data = hysteresis
            fig = Hysteresis.plot(hysteresis, show=False, save=False)
            hotspice.utils.save_results(parameters=parameters, data=data, figures=fig, outdir=save if isinstance(save, Path) else None)
        return hysteresis


## GENERAL HYSTERESIS METHODS, THAT ARE NOT SPECIFIC TO THE PINWHEEL REVERSAL
class Threshold:
    def __init__(self, thresholds: list[float], start_value=0):
        """ Use self.pass_check(value) to check if <value> is beyond any of the thresholds
            when seen from the <old_value> of the previous self.pass_check(old_value) call.
            The most recently passed threshold is disabled until another threshold is passed.
            @param thresholds: a list containing floats, each float representing a threshold.
            @param start_value (0): the <old_value> for the first time self.pass_check() is called.
        """
        self.thresholds = np.unique(thresholds) # List of thresholds where self.check() returns true if passed
        self.value = start_value # Last seen value
        self._last_threshold_passed = None # Used to make sure we don't activate the same threshold twice in row (is annoying)

    def pass_check(self, value):
        """ Returns True if a threshold was passed when going from <value> to <self.value>. """
        thresholds_passed = np.where(np.sign(self.thresholds - value) != np.sign(self.thresholds - self.value))[0]
        self.value = value

        if thresholds_passed.size == 0: return False # No thresholds passed
        if self._last_threshold_passed is not None:
            if thresholds_passed.size == 1:
                if self._last_threshold_passed == thresholds_passed[0]: # The only passed threshold is the last passed one
                    return False
        
        # We get here if we passed a threshold AND we either never passed a threshold before, passed multiple thresholds,
        # or the threshold we passed is not the previously passed one. In any case, store the nearest threshold and return True.
        nearest_threshold_i = (np.abs(self.thresholds - value)).argmin()
        self._last_threshold_passed = nearest_threshold_i
        return True


class Hysteresis:
    def __init__(self, mm: hotspice.Magnets, H_angle: float = 0, half: bool = True, **kwargs):
        """ Performs a hysteresis on `mm`, with the external field angled at `H_angle` (in DEGREES),
            and stores the relevant quantities. `kwargs` get passed to `self.run()`.
            `half`: If True, only the increasing part of the hysteresis is calculated.
        """
        self.mm = mm
        self.half = half
        if H_angle != 0 and np.any(np.isclose(0, np.array((H_angle, -H_angle)) % (np.pi/5040))): # Check if it is a suspicious multiple of pi (using highly composite number 5040)
            warnings.warn("H_angle in Hysteresis() must be in degrees, yet a suspicious multiple of pi was passed.\nAre you sure you used degrees?")
        self.H_angle = H_angle % 360
        self.run(**kwargs)
    
    def run(self, thresholds: list = None, H_max: float = 0.1, _N: int = 20000, verbose: bool = False):
        """ @param `H_max` [float] (0.1): The maximum field strength in Tesla.
                The hysteresis occurs between `-H_max` → `H_max`, and if `self.half` is False, it goes back to `-H_max`.
            @param `_N` [int] (20000): The number of steps between `H_max` and `-H_max`, in either direction.
            @param `thresholds` [list] (None): A list of threshold values for m_parallel_norm.
                Default (if None) is [0.98, s, 0, -s, -0.98], where s is the m_parallel for 90° reversal.
            
            Sets the following attributes:
            `H_fields` [np.array]: Magnitude of external field at each step
            `m_avg` [np.array]: Average magnetization at each step (magnitude)
            `m_angle` [np.array]: Average magnetization at each step (angle)
            `m_avg_sat` [float]: Saturation magnetization for external field along `self.H_angle`
            `m_angle_sat` [float]: Angle of magnetization at saturation for an external field along `self.H_angle`
            `threshold_indices` [list]: Indices of the `H_fields` array to which the respective `threshold_states` belong
            `threshold_states` [list[np.ndarray]]: List of `self.mm.m` arrays at a few moments during the hysteresis (controlled by `thresholds`)
        """
        angle_rad = np.deg2rad(self.H_angle)
        ZeemanEnergy: hotspice.ZeemanEnergy = self.mm.add_energy(hotspice.ZeemanEnergy(), exist_ok=True)
        
        m_parallel = lambda: self.mm.m_avg*np.cos(self.mm.m_avg_angle - angle_rad)
        self.mm.initialize_m(pattern='uniform', angle=angle_rad+np.pi) # Initialize uniform in the direction of the field
        self.m_avg_sat, self.m_angle_sat, m_parallel_sat = self.mm.m_avg, self.mm.m_avg_angle, abs(m_parallel()) # since we are at saturation, this can be useful to know without temperature effects
        m_parallel_normalized = lambda: m_parallel()/m_parallel_sat
        standardized_angle = (angle_rad + np.pi/4) % (np.pi/2) - np.pi/4 # Between 45° and -45°, such that halfway_reversed is in range [-1,1]
        halfway_reversed = np.sin(standardized_angle)/np.cos(standardized_angle) # Value of m_parallel at 90° 'reversal' (halfway point)
        if halfway_reversed < 0.2: halfway_reversed = 0 # Don't take too close to zero because we have a threshold there already
        if thresholds is None: thresholds = [.98, halfway_reversed, 0, -halfway_reversed, -.98]
        self.thresholds = Threshold(thresholds, start_value=m_parallel_normalized()) # If the average magnetization crosses this, a plot is shown.
        
        # TODO: Auto-detect at which energy the reversal starts, and when m_parallel_sat is reached (so not really H_max), use step size tesla_per_ZeemankBT, but need good way to detect if saturation is achieved to start coming back, currently commented code below is useless
        # desired_barrier = self.mm.kBT*40 # For attempt frequency of 10GHz, 40kBT barrier switches once a year, so is safe if we mm.progress() 1 second.
        # self.ZeemanEnergy.magnitude = 0
        # barrier_0 = self.mm.E_barrier(min_only=True)
        # self.ZeemanEnergy.magnitude = 1
        # barrier_1 = self.mm.E_barrier(min_only=True)
        # with np.errstate(divide='ignore'):
        #     tesla_per_ZeemankBT = xp.min(self.mm.kBT/xp.maximum(xp.abs(self.ZeemanEnergy.E), xp.abs(self.ZeemanEnergy.E_perp))) # Max. field [T] that does not increase Zeeman E by more than kBT anywhere
        #     self.ZeemanEnergy.magnitude = float(xp.max((desired_barrier - barrier_0)/(barrier_1 - barrier_0)))
        # while xp.any(self.mm.E_barrier(min_only=True) < desired_barrier):
        #     self.ZeemanEnergy.magnitude += 40*tesla_per_ZeemankBT

        self.H_fields = np.linspace(-H_max, H_max, _N) # One sweep to opposite fields (N steps)
        if not self.half: self.H_fields = np.append(self.H_fields, np.flip(self.H_fields)) # Sweep both ways (2*N steps)
        self.m_avg, self.m_angle = np.zeros_like(self.H_fields), np.zeros_like(self.H_fields)
        self.threshold_states, self.threshold_indices = [], []
        for i, H in enumerate(self.H_fields):
            if verbose and hotspice.utils.is_significant(i, _N, order=2): print(f"[{hotspice.utils.J_to_eV(self.mm.E_B_avg):.0f}eV] {i+1}/{_N*(2-self.half)}")
            ZeemanEnergy.set_field(angle=angle_rad, magnitude=H)
            if self.mm.params.UPDATE_SCHEME == 'Metropolis': self.mm.progress(r=1e-7)
            else: self.mm.progress()
            self.m_avg[i], self.m_angle[i] = self.mm.m_avg, self.mm.m_avg_angle
            if self.thresholds.pass_check(m_parallel_normalized()):
                self.threshold_indices.append(i)
                self.threshold_states.append(self.mm.m.copy())
        
        # TODO: Only retain the last value of a series where m_avg and m_angle are identical, except for the last value of one monodirectional field sweep, there retain the first value (otherwise we could get a long tail).
        #       So in essence, we want to keep only the relevant points from the hysteresis loop.

    @property
    def results(self):
        return {
            'H_fields': self.H_fields,
            'm_avg': self.m_avg, 'm_angle': self.m_angle,
            'm_avg_sat': self.m_avg_sat, 'm_angle_sat': self.m_angle_sat,
            'threshold_indices': self.threshold_indices, 'threshold_states': self.threshold_states, 'thresholds': self.thresholds.thresholds
        }

    @staticmethod
    # TODO: make this capable of plotting relaxations as well
    def plot(*hystereses: 'Hysteresis', labels: list[str] = None, colors: list[str] = None, legend_title: str = None, H_SIprefix: str = 'm', FIELD_POLAR_ZERO: bool = False, SHOW_SNAPSHOTS: bool = None, _vertical: bool = None, show: bool = True, save: bool|Path = False):
        """ Can pass multiple hysteresis calculations as the first few arguments, with legend `labels` and line/symbol `colors`. """
        hotspice.plottools.init_style()
        
        n = len(hystereses)
        H_factor = hotspice.utils.SIprefix_to_mul(H_SIprefix) # Default H_unit is milli(Tesla)
        is_monotonic = [all(data.H_fields[i] <= data.H_fields[i-1] for i in range(1, data.H_fields.size))
                    or all(data.H_fields[i] >= data.H_fields[i-1] for i in range(1, data.H_fields.size))
                    for data in hystereses] # Whether data[j] is a half hysteresis or not # TODO: refactor this using hysteresis.half
        if colors is None:
            colors = [plt.get_cmap('jet', 2*n+1)(2*i+1) if i < 5 else plt.get_cmap('jet', n)(i) for i in range(n)]
            # colors = np.resize(hotspice.plottools.colorcycle(), n)
        if labels is None: labels = [None]*len(hystereses)
        markers = np.resize(['o', 's', 'D', 'P', 'X', 'p', '*', '^'], n) # Circle, square, diamond, plus, cross, pentagon, star, triangle up (and repeat enough times)
        if SHOW_SNAPSHOTS is None: SHOW_SNAPSHOTS = n == 1
        if _vertical is None: _vertical = n == 1 # Whether the line plots are above the snapshot plots (True), or next to them (False). Also affects layout of snapshot plot subgrid.
        
        fig = plt.figure(figsize=(8, 6) if _vertical else (10, 4))
        if SHOW_SNAPSHOTS:
            snapshot_ratio = np.clip(n/2, 1, 2.5) # Don't use too much space for the snapshots, but don't squeeze them too much either
            if _vertical: gs_fig = fig.add_gridspec(2, 1, height_ratios=(2,snapshot_ratio))
            else: gs_fig = fig.add_gridspec(1, 2, width_ratios=(4,snapshot_ratio))
        else: # No snapshots so no subgrids needed
            gs_fig = fig.add_gridspec(1, 1)
        
        ## HYSTERESIS LINE PLOTS
        gs_sub_hysteresis = gs_fig[0].subgridspec(1, 2, width_ratios=(1,1), wspace=0.2)
        # NORMAL HYSTERESIS PLOT
        ax1 = fig.add_subplot(gs_sub_hysteresis[0]) # Normal hysteresis plot of parallel component
        ax1.grid(color='grey', linestyle=':')
        ax1.set_xlabel(f"External field magnitude [{H_SIprefix}T]")
        ax1.set_ylabel(r"Magnetization $M_{\parallel}/M_\mathrm{\parallel,sat}$")
        # POLAR PLOT
        ax2 = fig.add_subplot(gs_sub_hysteresis[1], projection='polar') # Polar plot
        ax2.set_rticks([0.2, 0.4, 0.6, 0.8, 1])  # Less radial ticks
        ax2.set_rlim(0, 1)
        ax2.set_theta_zero_location('N')
        ax2.set_thetagrids(angles=np.arange(0, 360, 30), zorder=-1)
        ax2.set_axisbelow(True)
        ax2.set_rlabel_position(285) # Move radial labels to an empty region
        if FIELD_POLAR_ZERO:
            ax2.set_xticklabels(["Field ↑"] + ax2.get_xticklabels()[1:]) # Theta ticks are 'xticks'
        ax2.grid(True)
        
        for j, data in enumerate(hystereses):
            ## Remove a lot of points that are identical to their neighbors (otherwise many thousands of points plotted, now a few hundred)
            nonremovable_i = np.array([i for i in range(1, data.H_fields.size - 1)
                                       if (data.m_avg[i] != data.m_avg[i+1]) or (data.m_angle[i] != data.m_angle[i+1]) # Only keep the first occurrence of a state, because we use where="post"
                                       or (data.m_avg[i] != data.m_avg[i-1]) or (data.m_angle[i] != data.m_angle[i-1])]) # in plt.step(). Would need "pre" if we would keep the last occurrence. # Only keep 'down' sweep if requested
            if not is_monotonic[j]: nonremovable_i = np.insert(nonremovable_i, 0, nonremovable_i[-1]) # Only need to connect end to start if is_half
            H_fields, m_avg, m_angle = data.H_fields[nonremovable_i], data.m_avg[nonremovable_i], data.m_angle[nonremovable_i]
            
            d_angle = m_angle - np.deg2rad(data.H_angle)
            m_parallel = m_avg*np.cos(d_angle)
            m_parallel_sat = abs(data.m_avg_sat*np.cos(data.m_angle_sat - np.deg2rad(data.H_angle)))
            
            color = colors[j]
            X = H_fields/H_factor
            Y = m_parallel/m_parallel_sat
            # NORMAL HYSTERESIS PLOT
            ax1.step(X, Y, where="post", linewidth=1, color=color, marker=markers[j], fillstyle='none', label=labels[j])
            if SHOW_SNAPSHOTS:
                for i, state in enumerate(data.threshold_states):
                    H_index = data.threshold_indices[i]
                    idx = np.searchsorted(nonremovable_i, H_index)
                    ax1.annotate(f"({i+1})", xy=(X[idx], Y[idx]), color=color, xycoords='data', xytext=(-3, 3), textcoords='offset points', va='bottom', ha='right')
            # POLAR PLOT
            angle = d_angle if FIELD_POLAR_ZERO else m_angle # If FIELD_POLAR_ZERO, then the 0° on the polar plot is the field direction. Otherwise, 0° is the x-axis.
            if is_monotonic[j]:
                ax2.plot(angle, (m_avg/data.m_avg_sat), color='red' if n == 1 else color, marker='^', fillstyle='none')
            else:
                N2 = math.floor(H_fields.size//2)
                ax2.plot(angle[:N2], (m_avg/data.m_avg_sat)[:N2], color='black' if n == 1 else color, marker='^', fillstyle='none', label='up' if j == 0 else None)
                ax2.plot(angle[N2:], (m_avg/data.m_avg_sat)[N2:], color='red' if n == 1 else color, marker='v', fillstyle='none', label='down' if j == 0 else None)
            if not FIELD_POLAR_ZERO: # Draw line of field
                all_same_angle = np.allclose([h.H_angle for h in hystereses], data.H_angle)
                if j == 0 or not all_same_angle: # Always draw the first one, but after that only draw if they are not all the same
                    for i in range(2): ax2.axvline(x=np.deg2rad(data.H_angle) + i*np.pi, ymin=0, ymax=1, color='black' if all_same_angle else color, lw=2, linestyle=':', zorder=0.75)
            
        if all(is_monotonic): ax1.set_xlim(left=0)
        if any(labels):
            ax1.legend(title=legend_title)
        if not all(is_monotonic):
            leg2 = ax2.legend(frameon=True, markerfirst=False, loc='upper right')
            for handle in leg2.legend_handles:
                if n > 1: handle.set_color('black')
        
        ## THRESHOLD STATES PLOT
        if SHOW_SNAPSHOTS:
            max_thresholds = max(len(data.threshold_indices) for data in hystereses)
            if _vertical:
                gs_sub_snapshots = gs_fig[1].subgridspec(n, max_thresholds, height_ratios=(1,)*n)
            else:
                gs_sub_snapshots = gs_fig[1].subgridspec(max_thresholds, n, width_ratios=(1,)*n)
            for j, data in enumerate(hystereses):
                for i, state in enumerate(data.threshold_states):
                    ax = fig.add_subplot(gs_sub_snapshots[j,i] if _vertical else gs_sub_snapshots[i,j])
                    hotspice.plottools.plot_simple_ax(ax, data.mm, state, mode='avg')
                    H = data.H_fields[data.threshold_indices[i]]
                    n_thresholds, c = len(data.threshold_indices), colors[j] # fontsize decreases if there are more plots
                    if _vertical:
                        fs = min(10, 80/max_thresholds)
                        separator = ' ' if n_thresholds < 11 else '\n'
                        ax.set_title(f"({i+1}){separator}{H/H_factor:.1f}{H_SIprefix}T", fontsize=fs, color=c)
                        if i == 0 and n > 1: ax.set_ylabel(labels[j], color=colors[j])
                    else:
                        fs = min(10, 40/max_thresholds)
                        separator = ' ' if n_thresholds < 5 else '\n'
                        ax.set_ylabel(f"({i+1}){separator}{H/H_factor:.1f}{H_SIprefix}T", rotation=270, labelpad=11, fontsize=fs, color=c)
                        ax.yaxis.set_label_position("right")
                        if i == 0 and n > 1: ax.set_title(labels[j], color=colors[j])
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_yticklabels([])
                    ax.set_yticks([])

        fig.tight_layout()
        if save: hotspice.plottools.save_plot(save, ext='pdf')
        if show:
            plt.show()
        else:
            plt.close(fig=fig)
        return fig


if __name__ == "__main__":
    """ KEY FINDINGS: 
        - M_PERP_FACTOR < 1 IS CRITICAL TO GET THE TWO-STEP BEHAVIOR BACK
        FOR ANGLE=30°:
            - E_B_STD 7% SEEMS BEST FIT WITH FIG. 4 FROM EXPERIMENTAL PAPER
            - E_B OF 60eV PLACES THE FIRST DROP AT THE SAME FIELD AS IN EXPERIMENT
            - M_PERP_FACTOR = 0.3 IS CRITICAL TO GET THE CORRECT FIELD MAGNITUDES FOR BOTH STEPS
        FOR ANGLE=-6°:
            - M_PERP_FACTOR = 0.3 NO LONGER WORKS: CRITICAL FIELD IS KIND OF OK, BUT POLAR PLOT IS NOT OK AT ALL (too wide, should be straight line)
    """
    save = False
    outdir_base = Path("./results/test_pinwheelReversal").absolute()
    
    def run_hysteresis_sweep(varname: Literal["angle", "E_B", "E_B_std", "m_perp_factor"], /,
                             angle: float = 30, E_B: float = 60, E_B_std: float = 0.07, m_perp_factor: float = 0.4, *,
                             half: bool = False, show_thresholds: bool = None, labels: list[str] = None):
        params = {"angle": angle, "E_B": E_B, "E_B_std": E_B_std, "m_perp_factor": m_perp_factor}
        values = params[varname]
        if not isinstance(values, Iterable):
            raise ValueError(f"Varname is '{varname}', yet its provided value ({values}) is not iterable.")
        
        ## Determine output directory name
        dirname = []
        if varname != "m_perp_factor": dirname.append(f"[⊥{m_perp_factor:.2g}]" if m_perp_factor else '[∥]')
        elif len(values) == 2 and 0 in values: dirname.append(f"[⊥{values[np.nonzero(values)[0][0]]}]") # If it is purely a comparison between perp and no perp, then add the ⊥ in the directory anyway
        if varname != "angle": dirname.append(f"{angle:.3g}°")
        dirname.append(f"E_B" if varname == "E_B" else f"{E_B:.4g}eV")
        if varname != "E_B_std": dirname[-1] += f"±{E_B_std*100:.2g}%" # Previous one is always present so can just add to it
        dirname = ",".join(dirname)
        if half: dirname += "_half"
        dirname += f"_{hotspice.utils.timestamp()}"
        outdir = outdir_base / varname / dirname
        print(f'Saving results in "{outdir}"' if save else 'Not saving results.')
        
        ## Determine relevant figure labels
        hystereses: list[Hysteresis] = []
        reversaltests: list[PinwheelReversalTest] = []
        legend_title = {"angle": f"Field angle",
                        "E_B": f"$E_B$ [eV]",
                        "E_B_std": f"$E_B$ std",
                        "m_perp_factor": f"$m_\perp$/$m_\parallel$"
                        }[varname] if labels is None else None
        labels = [(lambda x: {"angle": f"{x:.3g}°",
                              "E_B": f"{x:.4g}",
                              "E_B_std": f"{x*100:.2g}%",
                              "m_perp_factor": f"{x:.2g}"
                              }[varname])(value) for value in values] if labels is None else labels
        unit = {"angle": "°", "E_B": "eV", "E_B_std": "", "m_perp_factor": ""}[varname]
        
        ## Run the various hystereses
        for value in values:
            values_here = params | {varname: value}
            reversaltests.append(PinwheelReversalTest(E_B_std=values_here['E_B_std'], E_B=hotspice.utils.eV_to_J(values_here['E_B']), m_perp_factor=values_here['m_perp_factor']))
            hystereses.append(reversaltests[-1].hysteresis(H_angle=values_here['angle'], half=half, save=(outdir/f"{varname}={value:.2g}{unit}") if save else False))
        Hysteresis.plot(*hystereses, SHOW_SNAPSHOTS=show_thresholds, _vertical=show_thresholds, labels=labels, legend_title=legend_title, save=outdir/"summary.pdf" if save else False, show=(not save))
        return hystereses, reversaltests
    
    ## NEW VS. OLD BARRIER CALCULATION (PERP OR NO PERP) (for the closest reproduction of the experimental system as I can guess)
    # run_hysteresis_sweep("m_perp_factor", angle=-6, E_B_std=0.07, E_B=60, m_perp_factor=[0.4, 0], show_thresholds=True, labels=['With perp', 'No perp'])
    # run_hysteresis_sweep("m_perp_factor", angle=-6, E_B_std=0.07, E_B=60, m_perp_factor=[0.3, 0], show_thresholds=True, labels=['With perp', 'No perp'])
    # run_hysteresis_sweep("m_perp_factor", angle=30, E_B_std=0.07, E_B=60, m_perp_factor=[0.3, 0], show_thresholds=True, labels=['With perp', 'No perp'])
    # run_hysteresis_sweep("m_perp_factor", angle=30, E_B_std=0.05, E_B=71, m_perp_factor=[1, 0], show_thresholds=True, labels=['With perp', 'No perp']) # Original best guess before using m_perp_factor
    # run_hysteresis_sweep("m_perp_factor", angle=30, E_B_std=0.07, E_B=60, m_perp_factor=[0.4, 0], show_thresholds=True, labels=['With perp', 'No perp'])
    
    ## VARY FIELD ANGLE
    # run_hysteresis_sweep("angle", angle=np.linspace(0, 40, 5), E_B_std=0.07, E_B=60, m_perp_factor=0.4)

    ## VARY E_B_STD
    # run_hysteresis_sweep("E_B_std", E_B_std=np.linspace(0, 0.15, 3), angle=30, E_B=60, m_perp_factor=0.4)

    ## VARY E_B
    # run_hysteresis_sweep("E_B", E_B=np.linspace(1, 11, 6), angle=30, E_B_std=0.07, m_perp_factor=0.4)

    ## VARY M_PERP_FACTOR
    # run_hysteresis_sweep("m_perp_factor", m_perp_factor=[0.3, 0.5, 0.75, 1], angle=30, E_B=60, E_B_std=0.07)
    # run_hysteresis_sweep("m_perp_factor", m_perp_factor=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], angle=30, E_B=60, E_B_std=0.07)
    # run_hysteresis_sweep("m_perp_factor", m_perp_factor=[0.3, 0.5, 0.75, 1], angle=-6, E_B=60, E_B_std=0.07)
    # run_hysteresis_sweep("m_perp_factor", m_perp_factor=[0.3, 0.5, 0.75, 1], angle=0, E_B=60, E_B_std=0.07)

    # BEST FITS TO EXPERIMENT (zoomed in using 'half')
    # run_hysteresis_sweep("m_perp_factor", m_perp_factor=[0.3], half=True, angle=30, E_B=60, E_B_std=0.07)
