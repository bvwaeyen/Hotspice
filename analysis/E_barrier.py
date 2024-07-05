""" Shows the energy landscape (in the ideal case of perfectly sinusoidal energy components,
    which is true for DD and Zeeman, but not for shape anisotropy.
"""

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["HOTSPICE_USE_GPU"] = "False"
import hotspice


def plot(E_B=1):
    hotspice.plottools.init_style()
    delta_E_range = np.linspace(-E_B*4, E_B*4, 201)
    
    def method1(delta_E):
        E = -delta_E/2
        return np.maximum(delta_E, E_B - E)
    def method2(delta_E):
        E = -delta_E/2
        E_highest_state = np.abs(E)
        return np.where(E_B > E_highest_state, E_B - E, delta_E)
    
    thetas = np.linspace(0, 180, 181) # [deg]
    def E_landscape(delta_E=0, thetas=thetas): #! `thetas` in degrees! (bad design choice)
        return E_landscape_base(delta_E, thetas) + (1 - np.cos(2*np.deg2rad(thetas)))*E_B/2
    def E_landscape_base(delta_E=0, thetas=thetas): #! `thetas` in degrees! (bad design choice)
        return (1 - np.cos(np.deg2rad(thetas)))*delta_E/2
    
    E_barrier_method1 = method1(delta_E_range)
    E_barrier_method2 = method2(delta_E_range)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    artist_method1, = ax1.plot(delta_E_range, E_barrier_method1, label=r"Method 1: $\mathrm{max}(E_\perp, E_2) - E_1$")
    artist_method2, = ax1.plot(delta_E_range, E_barrier_method2, label=r"Method 2: $E_\perp$ if $E_\perp > \mathrm{max}(E_1, E_2)$ else $E_2$")
    artist_starline = ax1.scatter(delta_E_range, [E_landscape(delta_E)[np.argmax(E_landscape(delta_E))] for delta_E in delta_E_range], marker='*', color='C3', label=r"Real $E_\mathrm{barrier}$ for ideal sines")
    artist_parabolafit, = ax1.plot(delta_E_range, E_B*(delta_E_range/E_B/4+1)**2, color='red', label=r"Parabola fit to 'Real $E_\mathrm{barrier}$'")
    ax1.set_xlabel(r"$\Delta E$ between state $1 \rightarrow 2$")
    ax1.set_ylabel(r'$E_\mathrm{barrier}$')
    ax1.set_xlim([-4*E_B, 4*E_B])
    ax1.set_ylim([-4*E_B, 4*E_B])
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_frame_on(False)
    ax2.tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
    ax2.set_xlabel("Magnetization angle [°]")
    ax2.set_xlim([0, 180])
    ax2.set_ylim([-4*E_B, 4*E_B])
    fig.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(0.1) # Pause for DPI awareness
    background1 = fig.canvas.copy_from_bbox(ax1.bbox)
    background2 = fig.canvas.copy_from_bbox(ax2.bbox)

    artist_baselandscape, = ax2.plot(thetas, E_landscape_base(), color='black', lw=1, linestyle='--')
    artist_landscape, = ax2.plot(thetas, E_landscape(), color='C3')
    artist_top = ax2.scatter([0], [0], marker='*', label="Maximum", color='C3')
    artist_dots = ax2.scatter(dots_x := np.array([0,90,180]), E_landscape(0, thetas=dots_x), color='C1')
    
    def mouse_move(event=None):
        if event is None:
            delta_E = 0
        else:
            if event.inaxes is not ax1: delta_E = 0
            else: delta_E = event.xdata
        if delta_E is None: return
        
        E_landscape_now = E_landscape(delta_E)
        artist_landscape.set_ydata(E_landscape_now)
        artist_baselandscape.set_ydata(E_landscape_base(delta_E))
        topindex = np.argmax(E_landscape_now)
        artist_top.set_offsets([[thetas[topindex], E_landscape_now[topindex]]])
        artist_dots.set_offsets(np.asarray([dots_x, E_landscape(delta_E, thetas=dots_x)]).T)
        fig.canvas.restore_region(background1)
        fig.canvas.restore_region(background2)
        ax1.draw_artist(artist_method1)
        ax1.draw_artist(artist_method2)
        ax1.draw_artist(ax1.axvline(delta_E, color='black', lw=1, linestyle='dotted'))
        ax1.draw_artist(ax1.scatter([delta_E], [E_landscape_now[topindex]], marker='*', color='C3'))
        ax1.draw_artist(artist_starline)
        ax1.draw_artist(artist_parabolafit)
        ax2.draw_artist(artist_baselandscape)
        ax2.draw_artist(artist_landscape)
        ax2.draw_artist(artist_top)
        ax2.draw_artist(artist_dots)
        ax2.draw_artist(ax2.plot([.5,.5], [0, E_landscape_now[topindex]], color='C3')[0])
        ax2.draw_artist(ax2.plot([.5,180], [E_landscape_now[topindex], E_landscape_now[topindex]], color='C3', lw=1, linestyle=':')[0])
        ax2.draw_artist(ax2.plot([2,2], [0, method1(delta_E)], color='C0')[0])
        ax2.draw_artist(ax2.plot([2,180], [method1(delta_E), method1(delta_E)], color='C0', lw=1, linestyle=':')[0])
        ax2.draw_artist(ax2.plot([3.5,3.5], [0, method2(delta_E)], color='C1')[0])
        ax2.draw_artist(ax2.plot([3.5,180], [method2(delta_E), method2(delta_E)], color='C1', lw=1, linestyle=':')[0])
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)

    cid = fig.canvas.mpl_connect('motion_notify_event', mouse_move)

    mouse_move()
    plt.show()


def plot_ab(E_1, E_perp):
    # a, b = E_perp*np.sqrt(1 + (E_1/E_perp)**2), np.arctan2(E_1, E_perp)
    # plt.scatter([0, np.pi/2, np.pi, -np.pi/2, -np.pi], [E_1, E_perp, -E_1, -E_perp, -E_1], color="blue") # The known points
    # plt.plot(x := np.linspace(-np.pi, np.pi, 2001), a*np.sin(x+b))
    a, b = E_1*np.sqrt(1 + (E_perp/E_1)**2), np.arctan2(E_perp, E_1)
    plt.scatter([0, np.pi/2, np.pi, -np.pi/2, -np.pi], [E_1, E_perp, -E_1, -E_perp, -E_1], color="blue") # The known points
    plt.plot(x := np.linspace(-np.pi, np.pi, 2001), a*np.cos(x-b))
    plt.show()

if __name__ == "__main__":
    plot(E_B=2)