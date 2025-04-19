import inspect
import matplotlib.pyplot as plt

from matplotlib import cycler
from pathlib import Path

import hotspice

page_width = 6.32*1.45 # inches

class Cycle(object):
    def __init__(self, data): self.data = data
    def __getitem__(self, i): return self.data[i % len(self.data)]
    def __repr__(self): return self.data.__repr__()
marker_cycle = Cycle(['o', 's', 'D', 'P', 'X', 'p', '*', '^']) # Circle, square, diamond, plus, cross, pentagon, star, triangle up (and repeat enough times)

fs_small = 9
fs_medium = fs_small + 1
fs_large = fs_medium + 1

def init_style(style=None):
    hotspice.plottools.init_style(small=fs_small, medium=fs_medium, large=fs_large, style=style if style is not None else "default")
    if style is None: plt.rcParams['axes.prop_cycle'] = cycler(color=["dodgerblue", "tab:red", "tab:orange", "m", "c"])
    plt.rcParams["legend.fontsize"] = fs_medium
    plt.rcParams["mathtext.fontset"] = "dejavusans"

def replot_all(plot_function):
    """ Replots all the timestamp dirs  """
    script = Path(inspect.stack()[1].filename) # The caller script, i.e. the one where __name__ == "__main__"
    outdir = script.parent / (script.stem + '.out')
    for data_dir in outdir.iterdir():
        try:
            plot_function(data_dir)
        except Exception as e:
            print(e)
            pass

def label_ax(ax: plt.Axes, i: int = None, form: str = "(%s)", offset: tuple[float, float] = (0,0), fontsize: float = 11, **kwargs):
    """ To add a label to `ax`, pass either `i` or `form` (or both).
        If only `i` is passed, the label becomes "(a)", with the letter corresponding to index `i` (0=a, 1=b ...)
        If only `form` is passed, it is used as the complete label.
        If both `i` and `form` are passed, then the letter s corresponding to `i` is formatted using `form % s`.
        Examples:
            label_ax(ax, 1) --> "(b)"
            label_ax(ax, form="Some text.") --> "Some text."
            label_ax(ax, 3, "[%s]") --> "[d]"
        
        @param `ax` [plt.Axes]: The axis object for which the label should be drawn.
        @param `i` [int] (None): Index of the axis, which gets translated to a letter.
            If None, then `form` is assumed to be the complete string.
        @param `form` [str] ("(%s)"): The format used to represent the label with index `i`.
        @param `offset` [tuple(2)]: Tuple of two floats, determining x and y offset (in axis units).
        @param `fontsize` [float] (12): Font size of the label (default 12pt).
        Additional `**kwargs` get passed to the `ax.text()` call.
    """
    if isinstance(i, int):
        s = 'abcdefghijklmnopqrstuvwxyz'[i]
        form = form % s
    kwargs = dict(color='k', weight='bold', fontfamily='DejaVu Sans') | kwargs
    t = ax.text(0 + offset[0], 1 + offset[1], form, fontsize=fontsize,
                bbox=dict(boxstyle='square,pad=3', facecolor='none', edgecolor='none'),
                ha='left', va='bottom', transform=ax.transAxes, zorder=1000,
                **kwargs)

def get_last_outdir(subdir: str = None):
    script = Path(inspect.stack()[1].filename) # The caller script, i.e. the one where __name__ == "__main__"
    outdir_all = script.parent / (script.stem + '.out')
    if subdir is not None: outdir_all /= subdir
    timestamped_dirs = [d.absolute() for d in outdir_all.iterdir() if d.is_dir() and d.stem.isnumeric()]
    if len(timestamped_dirs) == 0:
        raise FileNotFoundError("No automatically generated output directory could be found.")
    else:
        return sorted(timestamped_dirs)[-1]
