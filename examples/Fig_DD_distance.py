""" This file creates a plot comparing the point dipole, second-order dipole, and dumbbell models
    that can be used to calculate the magnetostatic interaction.
    The resulting figure is used in the Hotspice paper as Figure 2.
    NOTE: This script requires mumax³ to be installed and available in the system PATH.
          Download mumax³ at https://mumax.github.io/download.html
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os
import pandas as pd

from enum import Enum, auto
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import hotspice
import _example_plot_utils as epu


outdir = os.path.splitext(hotspice.utils.get_caller_script())[0] + '.out' # Where things get saved by default
class Types(Enum):
    OOP = auto()
    IP_PARALLEL = auto()
    IP_ANTIPARALLEL = auto()


def get_data_OOP(scale: bool = False, **kwargs):
    moment = (Msat := 1063242)*(t_lay := 1.4e-9)*(n_lay := 7)*np.pi*((d := 170e-9)/2)**2
    # Mumax
    tablefile = f"{outdir}/OOP.out/table.txt"
    if not os.path.exists(tablefile): create_and_run_mumax_file("OOP", mumax_file_OOP)
    mumax_csv = pd.read_csv(tablefile, sep="\t")
    data_mumax = mumax_csv["E_MC (J)"].to_numpy()
    distances = mumax_csv["Distance (d)"].to_numpy()*d
    # Point dipole approximation
    data_dipole = 1e-7*moment**2*(-1/distances**3)
    # Point dipole approximation with second-order correction
    I_ij = ((d/2)**2 + (d/2)**2)/4
    data_dipole_finite = data_dipole + 1e-7*moment**2*(3/2*I_ij)*(-3/distances**5)
    # Dumbbell approximation
    with np.errstate(divide='ignore'):
        dumbbell_d = t_lay*n_lay
        data_dumbbell = 1e-7*moment**2/dumbbell_d**2*(-2/distances + 2/np.sqrt(distances**2+dumbbell_d**2))
    
    factor = -1e-7*moment**2 if scale else 1
    return {"distances": distances/d, "dipole": data_dipole/factor, "dipole_finite": data_dipole_finite/factor, "dumbbell": data_dumbbell/factor, "mumax": data_mumax/factor}


def get_data_IP_parallel(scale: bool = False, dumbbell_ratio: float = 1, **kwargs):
    moment = (Msat := 1063242)*(t := 10e-9)*np.pi*(l := 220e-9)*(b := 80e-9)/4
    # Mumax
    tablefile = f"{outdir}/IP_parallel.out/table.txt"
    if not os.path.exists(tablefile): create_and_run_mumax_file("IP_parallel", mumax_file_IP_parallel)
    mumax_csv = pd.read_csv(tablefile, sep="\t")
    data_mumax = mumax_csv["E_MC (J)"].to_numpy()
    distances = mumax_csv["Distance (l)"].to_numpy()*l
    # Point dipole approximation
    data_dipole = 1e-7*moment**2*(-2/distances**3)
    # Point dipole approximation with second-order correction
    I_ij = ((l/2)**2 + (b/2)**2)/4
    data_dipole_finite = data_dipole + 1e-7*moment**2*(3/2*I_ij)*(-4/distances**5)
    # Dumbbell approximation
    dumbbell_d = l*dumbbell_ratio #! Use l*0.9 for a good correspondence to mumax curve.
    with np.errstate(divide='ignore'):
        data_dumbbell = 1e-7*moment**2/dumbbell_d**2*(2/distances - 1/(distances-dumbbell_d) - 1/(distances+dumbbell_d))
    
    factor = -1e-7*moment**2 if scale else -1
    return {"distances": distances/l, "dipole": data_dipole/factor, "dipole_finite": data_dipole_finite/factor, "dumbbell": data_dumbbell/factor, "mumax": data_mumax/factor}


def get_data_IP_antiparallel(scale: bool = False, dumbbell_ratio: float = 1, **kwargs):
    moment = (Msat := 1063242)*(t := 10e-9)*np.pi*(l := 220e-9)*(b := 80e-9)/4
    # Mumax
    tablefile = f"{outdir}/IP_antiparallel.out/table.txt"
    if not os.path.exists(tablefile): create_and_run_mumax_file("IP_antiparallel", mumax_file_IP_antiparallel)
    mumax_csv = pd.read_csv(tablefile, sep="\t")
    data_mumax = mumax_csv["E_MC (J)"].to_numpy()
    distances = mumax_csv["Distance (w)"].to_numpy()*b
    # Point dipole approximation
    data_dipole = 1e-7*moment**2*(-1/distances**3)
    # Point dipole approximation with second-order correction
    I_ij = ((l/2)**2 + (b/2)**2)/4
    data_dipole_finite = data_dipole + 1e-7*moment**2*(3/2*I_ij)*(-1/distances**5)
    # Dumbbell approximation
    dumbbell_d = l*dumbbell_ratio #! Use l*0.9 for a good correspondence to mumax curve.
    with np.errstate(divide='ignore'):
        data_dumbbell = 1e-7*moment**2/dumbbell_d**2*(2/np.sqrt(distances**2 + dumbbell_d**2) - 2/distances)
    
    factor = -1e-7*moment**2 if scale else -1
    return {"distances": distances/b, "dipole": data_dipole/factor, "dipole_finite": data_dipole_finite/factor, "dumbbell": data_dumbbell/factor, "mumax": data_mumax/factor}


def inset_ax(ax: plt.Axes, ASI_type: Types, fig: plt.Figure = None):
    if fig is None: fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height # Width and height of ax in inches
    inset_width = 0.95*width # Width of inset ax in inches
    inset_ax: plt.Axes = inset_axes(ax, width=inset_width, height=inset_width, loc='upper right',
                        bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, borderpad=0)
    inset_ax.axis('off')
    inset_ax.patch.set_alpha(0) # Transparent background
    
    def draw_ellipse(x, y, rx, ry=None, **kwargs):
        if ry is None: ry = rx
        inset_ax.add_patch(Ellipse((x, y), width=2*rx, height=2*ry, **kwargs))
    
    def annotate_distance(text, x1, y1, x2, y2, color='k', opposite_side: bool = False, text_pad=3, endlines=False):
        fancyarrowkwargs = dict(posA=(x1, y1), posB=(x2, y2), color=color, lw=1, linestyle='-', zorder=1, shrinkA=0, shrinkB=0)
        inset_ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='<|-|>', mutation_scale=8))
        if endlines: inset_ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='|-|', mutation_scale=2))
        
        ln_slope = 1 + int(np.sign((x2 - x1)*(y2 - y1))) # Determine slope direction of line to set text anchors correctly
        ha = ["right", "center", "left"][::(-1 if opposite_side else 1)][ln_slope]
        va = "baseline" if opposite_side else "top"
        if x1 == x2: ha, va = "left" if opposite_side else "right", "center" # Special case: vertical line
        text_offset = transforms.offset_copy(inset_ax.transData, x=[text_pad, 0, -text_pad][::(-1 if opposite_side else 1)][ln_slope],
                                             y=text_pad*(1 if opposite_side else -1)*(va != "center"), units="points", fig=fig)
        inset_ax.text(x=np.mean((x1, x2)), y=np.mean((y1, y2)), s=text, color=color, ha=ha, va=va, transform=text_offset)

    r, padding = 0.19, 0.05
    d = 0.9
    wl_ratio = 4/11 # w/l ratio for ellipses
    colors = "gray", "gray" # "blue", "red"
    match ASI_type:
        case Types.OOP:
            rx = ry = r
            text = "$2r$"
        case Types.IP_PARALLEL:
            rx = r
            ry = wl_ratio*r
            text = "$l$"
        case Types.IP_ANTIPARALLEL:
            rx = wl_ratio*r
            ry = r
            text = "$w$"
    y = 1 - padding - r
    x2 = 1 - padding - rx
    x1 = x2 - rx - padding - rx
    draw_ellipse(x1, y, rx=rx, ry=ry, color=colors[0], alpha=0.9)
    draw_ellipse(x2, y, rx=rx, ry=ry, color=colors[1], alpha=0.9)
    annotate_distance("$r_{ij}$", x1, y - ry - padding/2, x2, y - ry - padding/2, endlines=True)
    match ASI_type:
        # case Types.OOP:
        #     inset_ax.text(x=x1, y=y, s=r"$\otimes$", color='k', ha="center", va="center")
        #     inset_ax.text(x=x2, y=y, s=r"$\odot$", color='k', ha="center", va="center")
        case Types.IP_PARALLEL:
            draw_ellipse(x1 - d*rx, y, r*0.05, color="blue")
            draw_ellipse(x1 + d*rx, y, r*0.05, color="red")
            draw_ellipse(x2 - d*rx, y, r*0.05, color="blue")
            draw_ellipse(x2 + d*rx, y, r*0.05, color="red")
            annotate_distance("$d=0.9l$", x2 - d*rx, y, x2 + d*rx, y, opposite_side=True, text_pad=12, color="C2")
        case Types.IP_ANTIPARALLEL:
            draw_ellipse(x1, y - d*ry, r*0.05, color="red")
            draw_ellipse(x1, y + d*ry, r*0.05, color="blue")
            draw_ellipse(x2, y - d*ry, r*0.05, color="blue")
            draw_ellipse(x2, y + d*ry, r*0.05, color="red")
            # annotate_distance("$d=0.9l$", x2 - d*rx, y, x2 + d*rx, y, opposite_side=True, text_pad=6)
    annotate_distance(text, x1 - rx, y, x1 + rx, y, opposite_side=True, text_pad=12 if ASI_type == Types.IP_PARALLEL else 3, endlines=True)
            

def show_DD_distance_fig():
    """ Shows a comparison between the DD interaction in OOP_Square as a function of distance,
        calculated using various methods:
            - The point dipole approximation
            - The point dipole approximation with second-order correction
            - The dumbbell approximation
            - mumax³
        This figure is used in the Hotspice paper as Figure 2.
    """
    get_data_kwargs = {'scale': True, 'dumbbell_ratio': 0.9}
    try:
        data = {Types.OOP: get_data_OOP(**get_data_kwargs), Types.IP_PARALLEL: get_data_IP_parallel(**get_data_kwargs), Types.IP_ANTIPARALLEL: get_data_IP_antiparallel(**get_data_kwargs)}
        titles = {Types.OOP: "OOP", Types.IP_PARALLEL: r"IP $\rightarrow\rightarrow$", Types.IP_ANTIPARALLEL: r"IP $\uparrow\downarrow$"}
    except FileNotFoundError:
        raise FileNotFoundError("The mumax output tables could not be found.\nSee the `get_data_...()` functions for the expected file paths.")
    
    epu.init_style()
    fig, axes = plt.subplots(1, len(data), figsize=(epu.page_width, 2.7))
    for i, (ASI_type, data_i) in enumerate(data.items()):
        ax: plt.Axes = axes[i]
        scale_y = 1e20
        dumbbell_is_dipole = np.allclose(data_i['dipole'], data_i['dumbbell'], rtol=1e-2)
        ax.plot(data_i['distances'], data_i['dipole']/scale_y, label="Point dipole")
        ax.plot(data_i['distances'], data_i['dipole_finite']/scale_y, label="Finite dipole")
        ax.plot(data_i['distances'], data_i['dumbbell']/scale_y, label="Dumbbell" + (f" $d={get_data_kwargs['dumbbell_ratio']:.1f}l$" if get_data_kwargs['dumbbell_ratio'] != 1 else ""), linestyle=(0,(3,3)) if dumbbell_is_dipole else '-')
        ax.plot(data_i['distances'], data_i['mumax']/scale_y, label="mumax³", color='k')
        if i == 0: ax.set_ylabel(r"$|E_\mathrm{MS}|/M^2$ (a.u.)" if get_data_kwargs['scale'] else r"$E_{MC}$ (J)")
        ax.set_xlabel([r"$r_{ij}/2r$", r"$r_{ij}/l$", r"$r_{ij}/w$"][i])
        ax.axvline(1, linestyle=':', color='grey', linewidth=1)
        ax.set_xlim(right=data_i['distances'].max())
        ax.set_ylim(bottom=0, top=5)
        inset_ax(ax, ASI_type=ASI_type)
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="upper center")
    fig.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.06, right=0.98)
    hotspice.utils.save_results(data=data, figures={"DD_distance": fig}, timestamped=False)


#### CALCULATION OF MUMAX INTERACTION ####
def create_and_run_mumax_file(filename: str, filecontents: str = None): # The file is expected in the Fig_DD_distance.out directory.
    if not os.path.exists(outdir): os.mkdir(outdir)
    filepath = outdir + "/" + filename + ".mx3"
    if filecontents is not None:
        with open(filepath, "w") as mumaxFile:
            mumaxFile.writelines(filecontents)
    os.system(f'mumax3 "{filepath}"')
    
mumax_file_OOP = """/*
    Calculates the magnetic dipole-dipole energy for a pair of out-of-plane magnets.
*/
//// Shape
d := 170e-9
z := 7*1.4e-9
distance_step := 1/100

//// Grid
Nx := 400
Ny := 100 
Nz := ceil(z/4e-9)
SetGridSize(Nx, Ny, Nz)
SetCellSize(d*distance_step, d*distance_step, z/Nz)
print(sprintf("Cell size: %.2fnm x %.2fnm x %.2fnm", d*distance_step*1e9, d*distance_step*1e9, z/Nz*1e9))

//// Material parameters
Msat = 1063242
Aex = 13e-12
print("Exchange length:", Sqrt((2*Aex.GetRegion(0))/(mu0*Pow(Msat.GetRegion(0), 2)))) // Check if larger than cell size
print("Saturation magnetization:", Msat.GetRegion(0))

//// Track E_total, angle, shape
TableAdd(E_total)
TableAdd(E_demag)
distance := 1.
distance0 := distance
interaction := 0.
TableAddVar(distance, "Distance", "d") // Center-to-center distance in multiples of <d>
TableAddVar(interaction, "E_MC", "J") // Magnetostatic coupling energy between magnets


//// Determine demag energy of single magnet (and declare shape variables so we can change them in the loop)
magnet := Circle(d)
magnet1 := magnet.Transl(-Nx*d*distance_step/2+d/2, 0, 0)
magnet2 := magnet.Transl( distance*d/2, 0, 0)
geometry := magnet1.Add(magnet2)
SetGeom(magnet)
m.Set(Uniform(0, 0, 1))
E_self := E_demag.Get()

//// Sweep center-to-center distance
for distance=distance0; distance <= 3; distance = distance + distance_step {
    print(sprintf("Distance: %.2f*d", distance))

    //// Set shape
    magnet2 = magnet1.Transl(distance*d, 0, 0)
    DefRegion(1, magnet1)
    DefRegion(2, magnet2)
    geometry = magnet1.Add(magnet2)
    SetGeom(geometry)

    m.SetRegion(1, Uniform(0, 0,  1))
    m.SetRegion(2, Uniform(0, 0, -1))
    interaction = E_demag.get() - 2*E_self
    //SnapshotAs(m, sprintf("m_%.2fd.png", distance))
    
    TableSave()
}"""

mumax_file_IP_parallel = """/*
    Calculates the magnetic dipole-dipole energy for a pair of in-plane magnets aligned along their long axis.
*/
//// Shape
l := 220e-9
w := 80e-9
z := 10e-9
distance_step := 1/100

//// Grid
Nx := 400
Ny := 40 
Nz := ceil(z/4e-9)
SetGridSize(Nx, Ny, Nz)
SetCellSize(l*distance_step, l*distance_step, z/Nz)
print(sprintf("Cell size: %.2fnm x %.2fnm x %.2fnm", l*distance_step*1e9, l*distance_step*1e9, z/Nz*1e9))

//// Material parameters
Msat = 1063242
Aex = 13e-12
print("Exchange length:", Sqrt((2*Aex.GetRegion(0))/(mu0*Pow(Msat.GetRegion(0), 2)))) // Check if larger than cell size
print("Saturation magnetization:", Msat.GetRegion(0))

//// Track E_total, angle, shape
TableAdd(E_total)
TableAdd(E_demag)
distance := 1.
distance0 := distance
interaction := 0.
TableAddVar(distance, "Distance", "l") // Center-to-center distance in multiples of <d>
TableAddVar(interaction, "E_MC", "J") // Magnetostatic coupling energy between magnets

//// Determine demag energy of single magnet (and declare shape variables so we can change them in the loop)
magnet := Ellipse(l, w)
magnet1 := magnet.Transl(-Nx*l*distance_step/2+l/2, 0, 0)
magnet2 := magnet.Transl( distance*l/2, 0, 0)
geometry := magnet1.Add(magnet2)
SetGeom(magnet)
m.Set(Uniform(1, 0, 0))
E_self := E_demag.Get()

//// Sweep center-to-center distance
for distance=distance0; distance <= 3; distance = distance + distance_step {
    print(sprintf("Distance: %.2f*l", distance))

    //// Set shape
    magnet2 = magnet1.Transl(distance*l, 0, 0)
    geometry = magnet1.Add(magnet2)
    SetGeom(geometry)

    m.Set(Uniform(1, 0, 0))
    interaction = E_demag.get() - 2*E_self
    //SnapshotAs(m, sprintf("m_%.2fl.png", distance))
    
    TableSave()
}
"""

mumax_file_IP_antiparallel = """/*
    Calculates the magnetic dipole-dipole energy for a pair of in-plane magnets aligned along their short axis.
*/
//// Shape
l := 220e-9
w := 80e-9
z := 10e-9
distance_step := 1/100

//// Grid
Nx := 400
Ny := 280
Nz := ceil(z/4e-9)
SetGridSize(Nx, Ny, Nz)
SetCellSize(w*distance_step, w*distance_step, z/Nz)
print(sprintf("Cell size: %.2fnm x %.2fnm x %.2fnm", w*distance_step*1e9, w*distance_step*1e9, z/Nz*1e9))

//// Material parameters
Msat = 1063242
Aex = 13e-12
print("Exchange length:", Sqrt((2*Aex.GetRegion(0))/(mu0*Pow(Msat.GetRegion(0), 2)))) // Check if larger than cell size
print("Saturation magnetization:", Msat.GetRegion(0))

//// Track E_total, angle, shape
TableAdd(E_total)
TableAdd(E_demag)
distance := 1.
distance0 := distance
interaction := 0.
TableAddVar(distance, "Distance", "w") // Center-to-center distance in multiples of <d>
TableAddVar(interaction, "E_MC", "J") // Magnetostatic coupling energy between magnets

//// Determine demag energy of single magnet (and declare shape variables so we can change them in the loop)
magnet := Ellipse(w, l)
magnet1 := magnet.Transl(-Nx*w*distance_step/2+w/2, 0, 0)
DefRegion(1, magnet1)
magnet2 := magnet.Transl( distance*w/2, 0, 0)
geometry := magnet1.Add(magnet2)
SetGeom(magnet)
m.Set(Uniform(0, 1, 0))
E_self := E_demag.Get()

//// Sweep center-to-center distance
for distance=distance0; distance <= 3; distance = distance + distance_step {
    print(sprintf("Distance: %.2f*w", distance))

    //// Set shape
    magnet2 = magnet1.Transl(distance*w, 0, 0)
    DefRegion(2, magnet2)
    geometry = magnet1.Add(magnet2)
    SetGeom(geometry)

    m.SetRegion(1, Uniform(0,  1, 0))
    m.SetRegion(2, Uniform(0, -1, 0))
    interaction = E_demag.get() - 2*E_self
    //SnapshotAs(m, sprintf("m_%.2fw.png", distance))
    
    TableSave()
}"""


if __name__ == "__main__":
    """ EXPECTED RUNTIME: <1min, since no dynamics are simulated, only figures are plotted.
        The pre-calculation of mumax interactions is required, which typically takes less than a minute.
    """
    show_DD_distance_fig()