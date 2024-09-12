import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from matplotlib import colormaps, colors
from matplotlib.patches import FancyArrowPatch

import hotspice
import _example_plot_utils as epu


def annotate_connection(ax: plt.Axes, text, x1, y1, x2, y2, color='k', opposite_side: bool = False, text_pad=3):
    fancyarrowkwargs = dict(posA=(x1, y1), posB=(x2, y2), color=color, lw=1, linestyle='-', zorder=1, shrinkA=0, shrinkB=0)
    ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='->', mutation_scale=8))
    
    ln_slope = 1 + int(np.sign((x2 - x1)*(y2 - y1))) # Determine slope direction of line to set text anchors correctly
    ha = ["right", "center", "left"][::(-1 if opposite_side else 1)][ln_slope]
    va = "baseline" if opposite_side else "top"
    if x1 == x2: ha, va = "left" if opposite_side else "right", "center" # Special case: vertical line
    offset_x = [text_pad, 0, -text_pad][::(-1 if opposite_side else 1)][ln_slope] if x1 != x2 else text_pad*(1 if opposite_side else -1)
    offset_y = text_pad*(1 if opposite_side else -1)*(va != "center")
    text_offset = transforms.offset_copy(ax.transData, x=offset_x, y=offset_y, units="points", fig=ax.get_figure())
    ax.text(x=np.mean((x1, x2)), y=np.mean((y1, y2)), s=text, color=color, ha=ha, va=va, transform=text_offset)

def plot_clocking(data, params, show_domains: bool = True):
    epu.init_style()
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(epu.page_width, epu.page_width/9*2))
    ax: plt.Axes = axes[0,0]
    
    OOP = params['ASI_type'] == "OOP_Square"
    ax.set_aspect('equal')
    ax.set_axis_off()
    if OOP:
        if show_domains:
            OOPcmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
            OOPcmap.set_bad(color='red')
        else:
            cmap = colormaps['hsv']
            r0, g0, b0, _ = cmap(.5) # Value at angle 'pi' (-1)
            r1, g1, b1, _ = cmap(0) # Value at angle '0' (1)
            cdict = {'red':   [[0.0, r0,  r0], # x, value_left, value_right
                                [0.5, 0.0, 0.0],
                                [1.0, r1,  r1]],
                        'green': [[0.0, g0,  g0],
                                [0.5, 0.0, 0.0],
                                [1.0, g1,  g1]],
                        'blue':  [[0.0, b0,  b0],
                                [0.5, 0.0, 0.0],
                                [1.0, b1,  b1]]}
            OOPcmap = colors.LinearSegmentedColormap('OOP_cmap', segmentdata=cdict, N=256)
            OOPcmap.set_bad(color='k')
    else:
        if show_domains:
            avg = hotspice.plottools.Average.SQUAREFOUR
        else:
            avg = hotspice.plottools.Average.POINT
    

    imfrac = 0.8 # How wide the imshows are w.r.t. their spacing
    x, y = -1, 0
    ax.set_xlim([-imfrac/2, imfrac/2])
    ax.set_ylim([-imfrac/2, imfrac/2])
    # Plot the states
    for i, state in enumerate(data['states']):
        nextrow = (data['values'][i] != data['values'][i-1]) if i > 1 else False
        dy = -1 if nextrow else 0
        dx = 0 if nextrow else -(y % 2)*2 + 1
        x, y = x + dx, y + dy
        ax.set_xlim(min(x-.5, ax.get_xlim()[0]), max(x+.5, ax.get_xlim()[1]))
        ax.set_ylim(min(y-.5, ax.get_ylim()[0]), max(y+.5, ax.get_ylim()[1]))
        if OOP:
            image = (1 - data['domains'][i]) if show_domains else (state + 1)/2
        else:
            mm = hotspice.ASI.IP_Pinwheel(a=1, n=params['size'])
            image = hotspice.plottools.get_rgb(mm, m=state, avg=avg, fill=True)
        ax.imshow(image, extent=[x-imfrac/2, x+imfrac/2, y-imfrac/2, y+imfrac/2],
                  vmin=0, vmax=1, cmap=OOPcmap if OOP else 'hsv', origin='lower')
        ax.add_patch(plt.Rectangle((x-imfrac/2, y-imfrac/2), imfrac, imfrac, fill=False, color="gray", linewidth=1))
        # Draw arrow from previous
        if i == 0: continue # No arrow drawn before the first image
        text = '$A$' if data['values'][i] else '$B$'
        if nextrow: annotate_connection(ax, text, x, y+1-imfrac/2, x, y+imfrac/2, opposite_side=y%2, text_pad=3)
        else: annotate_connection(ax, text, x+(-1+imfrac/2)*(1 if dx > 0 else -1), y, x+(-imfrac/2)*(1 if dx > 0 else -1), y)
    # Finish the axes
    epu.label_ax(ax, 1 if OOP else 0, offset=(-0.03, -0.14))
    fig.subplots_adjust(top=0.98, bottom=0.02)
    pos = ax.get_position()  # Get the original position of the axis
    left = 0.03
    new_pos = [left, pos.y0, pos.width, pos.height]  # Reduce width to shift to the left
    ax.set_position(new_pos)  # Set the new position
    return fig
