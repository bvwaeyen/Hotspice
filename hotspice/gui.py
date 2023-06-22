import customtkinter as ctk
import inspect
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk

from dataclasses import dataclass
from enum import auto, Enum
from inpoly import inpoly2
from matplotlib import cm, colorbar, colormaps, colors, image, quiver
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from textwrap import dedent
from tkinter import messagebox
from typing import Callable, Literal

from .core import Magnets, DipolarEnergy, ZeemanEnergy, ExchangeEnergy, SimParams
from .io import Inputter, OutputReader, BinaryDatastream, ScalarDatastream, IntegerDatastream
from .utils import appropriate_SIprefix, asnumpy, bresenham, J_to_eV, SIprefix_to_mul
from .plottools import get_rgb, Average, _get_averaged_extent, init_fonts
from . import config
if config.USE_GPU:
    import cupy as xp
else:
    import numpy as xp


class GUI(ctk.CTk):
    def __init__(self, mm: Magnets, inputter: Inputter = None, outputreader: OutputReader = None, 
                 custom_step: Callable[['GUI'], None] = None, custom_reset: Callable[['GUI'], None] = None,
                 editable: bool = True):
        """ <mm>, <inputter> and <outputreader> are typical Magnets, Inputter and OutputReader instances.
            <custom_step> and <custom_reset> are functions that support either 0 or 1 arguments.
                If they support 1 argument, that argument is supposed to be this GUI object.
                From there <mm>, <inputter> etc. (and everything else) can be accessed.
            If <editable> is False, no options will appear that can change <mm.m>.
        """
        ## Initialize self object and __init__-passed attributes
        super().__init__()
        self.mm = mm
        self.inputter = inputter
        self.outputreader = outputreader
        self.editable = editable
        
        self.custom_step, self.custom_reset = custom_step, custom_reset
        if self.custom_step is not None:
            if len(inspect.signature(custom_step).parameters) == 0: self.custom_step = lambda gui: custom_step()
        if self.custom_reset is not None:
            if len(inspect.signature(custom_reset).parameters) == 0: self.custom_reset = lambda gui: custom_reset()

        ## Configure layout of four main panels
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=300)
        self.grid_columnconfigure(1, weight=3, minsize=400)
        self.grid_columnconfigure(2, weight=1, minsize=400 if self.editable else 0)
        self.grid_rowconfigure(0, weight=3, minsize=300)
        self.grid_rowconfigure(1, weight=0, minsize=150)

        ## Set parameters for the entire window (color, scaling, size...)
        self.title("Hotspice GUI")
        ctk.set_appearance_mode("Light")  # Modes: system (default), light, dark
        ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
        if self.dark_mode: plt.style.use('dark_background') # Absolutely atrocious, but the possibility exists :)
        scaling = ctk.ScalingTracker.get_window_dpi_scaling(self)
        screen_width = self.winfo_screenwidth()/scaling
        screen_height = self.winfo_screenheight()/scaling
        min_width = sum([self.grid_columnconfigure(col).get('minsize', 0) for col in range(10)])/scaling # 10 cols just to be sure
        min_height = sum([self.grid_rowconfigure(row).get('minsize', 0) for row in range(10)])/scaling # 10 rows just to be sure
        self.minsize(min_width, min_height)
        width = (min_width + screen_width)/2
        height = (min_height + screen_height)/2
        self.geometry(f"{width:.0f}x{height:.0f}+{(screen_width-width)/2:.0f}+{(screen_height - 40 - height)/2:.0f}")

        ## 1.1) Magnetization view panel
        self.magnetization_view = MagnetizationView(self, gui=self)
        self.magnetization_view.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0), sticky=ctk.NSEW)

        ## 1.2) Realtime parameters info panel
        self.parameter_info = ParameterInfo(self, gui=self, width=300)
        self.parameter_info.grid(row=1, column=0, padx=10, pady=(10, 0), sticky=ctk.NSEW)

        ## 2) View-settings panel (quiver, averaging etc.)
        self.magnetization_view_settings = MagnetizationViewSettingsTabView(self, self.magnetization_view, height=150)
        self.magnetization_view_settings.grid_propagate(False)
        self.magnetization_view_settings.grid(row=1, column=1, padx=10, pady=(0, 10), sticky=ctk.NSEW)
        
        ## 3) Actions panel (step, progress, relax etc.)
        self.actions_panel = ActionsPanel(self, gui=self)
        if self.editable: self.actions_panel.grid(row=0, column=2, padx=10, pady=(10, 0), sticky=ctk.NSEW)
        self.actions_panel.grid_columnconfigure((0, 1), weight=1)
        self.actions_panel.grid_columnconfigure(2, weight=0)

        ## 4) ASI-settings panel (update scheme etc.)
        self.ASI_settings_frame = ASISettingsFrame(self, gui=self, fg_color="red")
        if self.editable: self.ASI_settings_frame.grid(row=1, column=2, padx=10, pady=(20, 10), sticky=ctk.NSEW)
        self.ASI_settings = self.ASI_settings_frame.settings

        if not self.editable:
            self.deactivate_widget(self.actions_panel)
            self.deactivate_widget(self.ASI_settings_frame)

        ## Focus on this window
        self.lift()
        self.attributes("-topmost", True)
        self.attributes("-topmost", False)
        self.after(1, lambda: self.focus_set())

    def show(self):
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                plt.close("all")
                self.destroy()
        self.protocol("WM_DELETE_WINDOW", on_closing)
        self.mainloop()
    
    def redraw(self, *args, **kwargs):
        self.magnetization_view.redraw(*args, **kwargs)
        self.parameter_info.update()

    @property
    def dark_mode(self) -> bool:
        return ctk.get_appearance_mode() == "Dark"
    
    def all_children(self, widget):
        return [widget] + [subchild
                           for child in widget.winfo_children()
                           for subchild in self.all_children(child)]
        
    def deactivate_widget(self, widget):
        for child in self.all_children(widget):
            try: child.configure(state=ctk.DISABLED)
            except: pass
    

class ActionsPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, gui: GUI, **kwargs):
        super().__init__(master, **kwargs)
        self.gui = gui
        self.mm = gui.mm
        cols = 3
        row = 0
        
        ## UPDATE
        # Widget definitions and default values
        self.action_step = ctk.CTkButton(self, text="Update", command=lambda:self.action('step', n=int(self.action_step_n.get())))
        self.action_step_n = ttk.Spinbox(self, from_=1, wrap=True, values=[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000], width=6, textvariable=tk.IntVar(value=1))
        action_step_n_label = ctk.CTkLabel(self, text="steps", anchor=ctk.W)
        self.action_step_n.insert(0, "1")
        # Grid geometry manager
        self.action_step.grid(row=row, column=0, sticky=ctk.EW, padx=10, pady=(10,0))
        self.action_step_n.grid(row=row, column=1, sticky=tk.E, padx=10, pady=(10,0))
        action_step_n_label.grid(row=row, column=2, sticky=ctk.W, padx=10, pady=(10,0))

        ## PROGRESS
        # Widget definitions and default values
        self.action_progress = ctk.CTkButton(self, text="Progress", command=lambda:self.action('progress', t_max=float(self.action_progress_tmax.get()), MCsteps_max=float(self.action_progress_MCstepsmax.get())))
        sig_progress = inspect.signature(self.mm.progress)
        action_progress_tmax_values = list(10.**np.arange(-10, 10)) + [1e100, np.inf]
        self.action_progress_tmax = ttk.Spinbox(self, from_=0, wrap=True, width=15, values=action_progress_tmax_values, textvariable=tk.DoubleVar(value=sig_progress.parameters['t_max'].default))
        self.action_progress_tmax.insert(0, "1")
        action_progress_tmax_label = ctk.CTkLabel(self, text="seconds", anchor=ctk.W)
        self.action_progress_MCstepsmax = ttk.Spinbox(self, from_=0, wrap=True, values=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 2, 4, 8, 16, 32, 1001, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000], width=6, textvariable=tk.DoubleVar(value=sig_progress.parameters['MCsteps_max'].default))
        self.action_progress_MCstepsmax.insert(0, "4")
        action_progress_MCstepsmax_label = ctk.CTkLabel(self, text="MC steps", anchor=ctk.W)
        # Grid geometry manager
        row += 1
        ttk.Separator(self).grid(row=row, columnspan=cols, sticky=ctk.EW, pady=5)
        row += 1
        self.action_progress.grid(row=row, column=0, rowspan=2, sticky=ctk.EW, padx=10)
        self.action_progress_tmax.grid(row=row, column=1, sticky=ctk.E, padx=10)
        action_progress_tmax_label.grid(row=row, column=2, sticky=ctk.W, padx=10)
        row += 1
        self.action_progress_MCstepsmax.grid(row=row, column=1, sticky=ctk.E, padx=10)
        action_progress_MCstepsmax_label.grid(row=row, column=2, sticky=ctk.W, padx=10)

        ## INITIALIZE
        # Widget definitions and default values
        # - Button
        def action_initialize_click(value): # Helper function that makes the SegmentedButton behave like many buttons at once (i.e. not persistent)
            self.action('initialize', pattern=value, angle=float(self.action_initialize_angle.get())/180*np.pi)
            self.action_initialize._unselect_button_by_value(value)
            self.action_initialize._current_value = ""
        self.action_initialize = ctk.CTkSegmentedButton(self, values=["Uniform", "AFM", "Vortex", "Random"], unselected_color=ctk.ThemeManager.theme["CTkSegmentedButton"]["selected_color"], command=action_initialize_click)
        action_initialize_angle_label_text = lambda angle=0: action_initialize_angle_label.configure(text=f"{float(angle):.0f}°")
        # - Slider
        if self.mm.in_plane:
            from_, to, number_of_steps = 0, 359, 359
        else:
            from_, to, number_of_steps = 0, 180, 1
        self.action_initialize_angle = ctk.CTkSlider(self, from_=from_, to=to, number_of_steps=number_of_steps, command=action_initialize_angle_label_text)
        self.action_initialize_angle.set(0)
        # - Label
        action_initialize_angle_label = ctk.CTkLabel(self, anchor=ctk.W, padx=10)
        action_initialize_angle_label_text()
        # Grid geometry manager
        row += 1
        ttk.Separator(self).grid(row=row, columnspan=cols, sticky=ctk.EW, pady=5)
        row += 1
        self.action_initialize.grid(row=row, column=0, columnspan=3, sticky=ctk.EW, pady=10, padx=10)
        row += 1
        self.action_initialize_angle.grid(row=row, column=0, columnspan=2, sticky=ctk.EW, padx=(50, 10))
        action_initialize_angle_label.grid(row=row, column=2, sticky=ctk.W, padx=10)

        ## INPUTTER
        # Widget definitions and default values
        self.action_inputter = ctk.CTkButton(self, text="Input value", command=lambda:self.action('inputter', value=self.action_inputter_value.get(), remove_stimulus=self.action_inputter_remove_stimulus.get()))
        self.action_inputter_random = ctk.CTkButton(self, text="Input random", fg_color="green", command=lambda:self.action('inputter', value=None, remove_stimulus=self.action_inputter_remove_stimulus.get()))
        datastream = self.gui.inputter.datastream if self.gui.inputter is not None else None
        if isinstance(datastream, BinaryDatastream):
            self.action_inputter_value = ctk.CTkSwitch(self, text="1", width=100)
        elif isinstance(datastream, ScalarDatastream):
            self.action_inputter_value = ctk.CTkSlider(self, from_=0, to=1)
            self.action_inputter_value.set(0)
        elif isinstance(datastream, IntegerDatastream):
            self.action_inputter_value = ttk.Spinbox(self, from_=0, to=int(2**datastream.bits_per_int)-1, wrap=False, width=6, textvariable=tk.IntVar(value=0))
            self.action_inputter_value.insert(0, "0")
            # self.action_inputter_value = ctk.CTkEntry(self, width=100, textvariable=tk.StringVar(value=""))
        self.action_inputter_remove_stimulus = ctk.CTkSwitch(self, text="Remove stimulus after input")
        self.action_inputter_remove_stimulus.deselect() # Set to 'off' by default
        # Grid geometry manager
        if self.gui.inputter is not None:
            row += 1
            ttk.Separator(self).grid(row=row, columnspan=cols, sticky=ctk.EW, pady=5)
            row += 1
            self.action_inputter_random.grid(row=row, column=0, columnspan=cols, sticky=ctk.EW, padx=40, pady=(10,0))
            row += 1
            self.action_inputter.grid(row=row, column=0, sticky=ctk.EW, padx=10, pady=(10,0))
            if isinstance(datastream, BinaryDatastream):
                ctk.CTkLabel(self, text="0").grid(row=row, column=1, sticky=ctk.E, padx=0, pady=(10,0))
                self.action_inputter_value.grid(row=row, column=2, sticky=ctk.W, padx=10, pady=(10,0))
            elif isinstance(datastream, ScalarDatastream):
                self.action_inputter_value.grid(row=row, column=1, columnspan=2, sticky=ctk.EW, padx=10, pady=(10,0))
            elif isinstance(datastream, IntegerDatastream):
                self.action_inputter_value.grid(row=row, column=1, columnspan=2, sticky=ctk.W, padx=10, pady=(10,0))
            row += 1
            self.action_inputter_remove_stimulus.grid(row=row, column=0, columnspan=cols, sticky=ctk.EW, padx=10, pady=(10,0))

        ## CUSTOM FUNCTION
        # Widget definitions and default values
        self.action_customstep = ctk.CTkButton(self, text="Custom step", fg_color="blue", command=lambda:self.action('custom_step', n=int(self.action_customstep_n.get())))
        self.action_customstep_n = ttk.Spinbox(self, from_=1, wrap=True, values=[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000], width=6, textvariable=tk.IntVar(value=1))
        action_customstep_n_label = ctk.CTkLabel(self, text="times", anchor=ctk.W)
        self.action_customstep_n.insert(0, "1")
        self.action_customreset = ctk.CTkButton(self, text="Custom reset", fg_color="blue", command=lambda:self.action('custom_reset'))
        # Grid geometry manager
        if self.gui.custom_step is not None or self.gui.custom_reset is not None:
            row += 1
            ttk.Separator(self).grid(row=row, columnspan=cols, sticky=ctk.EW, pady=5)
        if self.gui.custom_step is not None:
            row += 1
            self.action_customstep.grid(row=row, column=0, sticky=ctk.EW, padx=10, pady=(10,0))
            self.action_customstep_n.grid(row=row, column=1, sticky=tk.E, padx=10, pady=(10,0))
            action_customstep_n_label.grid(row=row, column=2, sticky=ctk.W, padx=10, pady=(10,0))
        if self.gui.custom_reset is not None:
            row += 1
            columnspan = 1 if self.gui.custom_step is not None else cols
            self.action_customreset.grid(row=row, column=0, columnspan=columnspan, sticky=ctk.EW, padx=10, pady=(10,0))

        # TODO: add actions (minimize(all), relax, initialize with a menu showing options 'uniform', 'AFM', 'random', 'vortex' with additional angle value next to it)

    def action(self, action: str, **kwargs):
        """ An 'action' is an event that changes the magnetization state, which then needs to be redrawn. """
        match action:
            case 'step' | 'update':
                self._action_step(**kwargs)
            case 'progress':
                self._action_progress(**kwargs)
            case 'initialize':
                self._action_initialize(**kwargs)
            case 'inputter':
                self._action_inputter(**kwargs)
            case 'custom_step':
                self._action_custom_step(**kwargs)
            case 'custom_reset':
                self._action_custom_reset(**kwargs)
        self.gui.redraw(settings_changed=False)
    
    def _action_step(self, n=1, **kwargs):
        match self.mm.params.UPDATE_SCHEME:
            case "Néel":
                kwargs.setdefault("t_max", self.gui.ASI_settings.t_max)
                kwargs.setdefault("attempt_freq", self.gui.ASI_settings.attempt_freq)
            case "Glauber":
                kwargs.setdefault("Q", self.gui.ASI_settings.Q)
        for _ in range(n):
            self.mm.update(**kwargs)
    
    def _action_progress(self, t_max=1., MCsteps_max=4., **kwargs):
        self.mm.progress(t_max=t_max, MCsteps_max=MCsteps_max)
    
    def _action_initialize(self, pattern='random', **kwargs):
        self.mm.initialize_m(pattern, **kwargs)
    
    def _action_inputter(self, value=None, **kwargs):
        if value is not None:
            try: value = float(value)
            except ValueError: print('Hotspice GUI error: Inputter did not receive a valid scalar value.') # Don't throw a real error, we want to keep the GUI running
        value = self.gui.inputter.input(self.mm, values=value, **kwargs)
        print(f"Applied input value {value}")
    
    def _action_custom_step(self, n=1, **kwargs):
        if self.gui.custom_step is not None:
            for _ in range(n): self.gui.custom_step(self.gui)

    def _action_custom_reset(self, **kwargs):
        if self.gui.custom_reset is not None:
            self.gui.custom_reset(self.gui)


class ParameterInfo(ctk.CTkFrame):
    def __init__(self, master, gui: GUI, **kwargs):
        super().__init__(master, **kwargs)
        self.gui = gui
        self.mm = gui.mm
        # We will have three columns: <name> <value> <unit>
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        # 1) The time.
        self.info_t = tk.StringVar()
        self.info_t_label = ctk.CTkLabel(self, textvariable=self.info_t)
        ctk.CTkLabel(self, text="Time elapsed:").grid(row=0, column=0, sticky=ctk.E)
        self.info_t_label.grid(row=0, column=1, padx=10, sticky=ctk.W)
        # 2) (Attempted) switches overall.
        self.info_switches = tk.StringVar()
        self.info_switches_label = ctk.CTkLabel(self, textvariable=self.info_switches)
        ctk.CTkLabel(self, text="Switches:").grid(row=1, column=0, sticky=ctk.E)
        self.info_switches_label.grid(row=1, column=1, padx=10, sticky=ctk.W)
        # 3) MCsteps (Glauber)
        self.info_MCsteps = tk.StringVar()
        self.info_MCsteps_label = ctk.CTkLabel(self, textvariable=self.info_MCsteps)
        ctk.CTkLabel(self, text="Monte Carlo steps:").grid(row=2, column=0, sticky=ctk.E)
        self.info_MCsteps_label.grid(row=2, column=1, padx=10, sticky=ctk.W)
        self.info_attempted_switches = tk.StringVar()
        self.info_attempted_switches_label = ctk.CTkLabel(self, font=ctk.CTkFont(size=10), anchor=ctk.N, textvariable=self.info_attempted_switches)
        self.info_attempted_switches_label.grid(row=3, column=1, padx=10, sticky=ctk.W)
        # 4) Switches in the last iteration.
        self.reset_button = ctk.CTkButton(self, text="Reset all values to zero", command=self.reset)
        self.reset_button.grid(row=4, column=0, columnspan=2, padx=10, pady=0, sticky=ctk.EW)
        # Possibly also T_avg etc. in another section if there is room.
        self.update() # To initialize all StringVars in their correct formatting
    
    def update(self):
        t, tp = appropriate_SIprefix(self.mm.t)
        self.info_t.set(f"{t:.3f} {tp}s")
        self.info_switches.set(f"{self.mm.switches:d}")
        self.info_MCsteps.set(f"{self.mm.MCsteps:.2f}")
        self.info_attempted_switches.set(f"({self.mm.attempted_switches:d} sw. attempts)")
    
    def reset(self): # TODO: make this a method of the ASI object itself, now it is quite hacky
        self.mm.t = 0
        self.mm.switches = 0
        self.mm.attempted_switches = 0 # This sets MCsteps automatically
        self.update()


class MagnetizationView(ctk.CTkFrame):
    class DisplayMode(Enum):
        MAGNETIZATION = auto()
        QUIVER = auto()
        DOMAINS = auto()
        ENERGY = auto()
        FIELD = auto()
        READOUT = auto() # Not entirely clear how to do this consistently, so not implemented for now

    @dataclass
    class ViewSettings:
        avg: Average|str = Average.POINT
        fill: bool = True
        color_quiver: bool = True
        subtract_barrier: bool = True
        energy_component: Literal['Total', 'E_barrier']|str = 'Total'
        in_eV: bool = True
        field_type: Literal['E_B', 'T', 'moment'] = 'E_B'

        def __post_init__(self):
            self.avg = Average.resolve(self.avg)
            self.fill = bool(self.fill)
            self.color_quiver = bool(self.color_quiver)
            self.subtract_barrier = bool(self.subtract_barrier)
            self.energy_component = str(self.energy_component)
            self.in_eV = bool(self.in_eV)
            self.field_type = str(self.field_type)
            if self.field_type not in self.available_field_types: raise ValueError(f"<field_type> can only be any of {self.available_field_types}.")
        
        @property
        def available_field_types(self): return ['E_B', 'T', 'moment']

    def __init__(self, master, gui: GUI, outputreader: OutputReader = None, **kwargs):
        super().__init__(master, **kwargs)
        self.gui = gui
        self.mm = gui.mm
        self.outputreader = outputreader

        ## Create the basic MPL figure inside this Tkinter frame
        self.figure = Figure(dpi=100, tight_layout=True) # Tight_layout true at the start to get a nice layout to begin with
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=ctk.BOTTOM, fill=ctk.X)
        self.canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)
        self.infobutton = NavigationToolbar2Tk._Button(self.toolbar, " 🛈 ", None, False, command=self.show_info)
        self.infobutton.configure(fg='blue',font=('Calibri',14,'bold'))
        self.infobutton.pack(side=ctk.RIGHT, fill=ctk.Y)
        self.canvas.draw()
        # TODO: add some sidebar or subbar or something where t, MCsteps, (attempted_)switches are shown
        # TODO: possibly also show realtime quantities like correlation_length, m_avg, T_avg, E_B_avg, moment_avg, E_tot etc.

        ## Draw DisplayMode-independent elements, like axes, axes labels, etc.
        self.settings = MagnetizationView.ViewSettings()
        self.unit_axes = 'µ'
        self.unit_axes_factor = SIprefix_to_mul(self.unit_axes)
        self.full_extent = np.array([self.mm.x_min - self.mm.dx/2,
                                     self.mm.x_max + self.mm.dx/2,
                                     self.mm.y_min - self.mm.dy/2,
                                     self.mm.y_max + self.mm.dy/2]
                                    )/self.unit_axes_factor
        self.ax: Axes = self.figure.add_subplot(111)
        self.ax_aspect = 'auto' if self.mm.nx == 1 or self.mm.ny == 1 else 'equal'
        self.ax.set_aspect(self.ax_aspect)
        self.ax.set_xlabel(f"x [{self.unit_axes}m]")
        self.ax.set_ylabel(f"y [{self.unit_axes}m]")
        self.ax.set_xlim(self.full_extent[:2])
        self.ax.set_ylim(self.full_extent[2:])
        self.content: quiver.Quiver | image.AxesImage = None
        self.colorbar: colorbar.Colorbar = None
        self.figparams = { # This should not be changed dynamically, so this can stay as a dictionary
            'fontsize_colorbar' : 10,
            'fontsize_axes': 10,
            'text_averaging': True
        }

        self.cmap_hsv = colormaps['hsv']
        if not self.mm.in_plane: # Determine OOP cmap from IP hsv cmap at angles 0 and pi
            r0, g0, b0, _ = self.cmap_hsv(.5) # Value at angle 'pi' (-1)
            r1, g1, b1, _ = self.cmap_hsv(0) # Value at angle '0' (1)
            cdict = {'red': [[0.0, r0,  r0], # x, value_left, value_right
                            [0.5, 0.0, 0.0],
                            [1.0, r1,  r1]],
                    'green':[[0.0, g0,  g0],
                            [0.5, 0.0, 0.0],
                            [1.0, g1,  g1]],
                    'blue': [[0.0, b0,  b0],
                            [0.5, 0.0, 0.0],
                            [1.0, b1,  b1]]}
            self.cmap_hsv = colors.LinearSegmentedColormap('OOP_cmap', segmentdata=cdict, N=256)
        
        ## Draw everything specific to a DisplayMode, default to MAGNETIZATION
        self.change_mode(self.DisplayMode.MAGNETIZATION) # This initializes the contents of the matplotlib plot
        self.figure.set_tight_layout(False) # To prevent figure widgets changing size when changing DisplayMode

        ## MATPLOTLIB EVENT LISTENERS
        def on_figure_resize(event): # Make sure the matplotlib plot stays nicely laid out when resizing the screen (no text gets cut off at bottom etc.)
            self.figure.set_tight_layout(True) # Activates a layout engine in self.figure
            self.figure.get_layout_engine().execute(self.figure) # That layout engine recalculates the layout of self.figure
            self.figure.set_tight_layout(False) # To prevent figure widgets changing size when changing DisplayMode
        sid = self.figure.canvas.mpl_connect('resize_event', on_figure_resize)

        ## MOUSE AND KEYBOARD CONTROLS
        toolbar_is_in_use = lambda: any([button.var.get() for button in self.toolbar._buttons.values() if isinstance(button, tk.Checkbutton)]) # Only 'Pan' and 'Zoom' buttons are Checkbutton
        def get_idx(mouse_event: MouseEvent):
            """ Returns: (x, y) integer indices of self.mm.m array where MouseEvent took place.
                         (None, None) if invalid event or somehow else not intending to change mm.m state.
            """
            if mouse_event.inaxes is not self.ax: return None, None # Only respond to mouse events inside the main drawing area (not e.g. the colorbar)
            if toolbar_is_in_use(): return None, None # Do not change magnets when the user wants to Pan or Zoom
            idx_x = int(xp.argmin(xp.abs(mouse_event.xdata*self.unit_axes_factor - self.mm.xx[0,:].reshape(-1))))
            idx_y = int(xp.argmin(xp.abs(mouse_event.ydata*self.unit_axes_factor - self.mm.yy[:,0].reshape(-1))))
            return idx_x, idx_y
        
        self.last_toggle_index = (0, 0) # (y, x)
        def toggle(indices_x: int|xp.ndarray = None, indices_y: int|xp.ndarray = None, mouse_event: MouseEvent = None):
            """ Toggles the magnet(s) at self.mm.m[indices_y, indices_x] (and updates all necessary variables along with it).
                If <mouse_event> is passed as an argument, it is used to determine a more specific action, using filter_indices().
            """
            if indices_x is None or indices_y is None:
                if mouse_event is None: return # Then no information was provided for the toggle, so we can't do anything
                indices_x, indices_y = get_idx(mouse_event) # Get the indices information from mouse_event
            if mouse_event is not None:
                indices_x, indices_y = filter_indices(mouse_event.key, xp.asarray(indices_x).reshape(-1), xp.asarray(indices_y).reshape(-1))
            if indices_x.size == 0: return # Efficiency
            self.mm.m[indices_y, indices_x] *= -1
            self.mm.update_energy(index=xp.asarray([indices_y, indices_x]).reshape(2, -1))
            self.last_toggle_index = (int(indices_y[-1]), int(indices_x[-1]))
            self.redraw(settings_changed=False)

        def filter_indices(key: str, indices_x, indices_y): # KEY is a keyboard key, NOT MOUSE
            """ Only retains the indices of magnets that should be flipped according to <key>. """
            if indices_x.size == 0 or key is None: return indices_x, indices_y # No lookup needed
            match key:
                case 'left' | 'right': # Arrow keys left-right
                    if not self.mm.in_plane: return indices_x, indices_y
                    direction = -1 if key == 'left' else 1
                    mx = self.mm.orientation[indices_y,indices_x,0]*self.mm.m[indices_y,indices_x]
                    update_indices = xp.where(mx*direction < 0)
                case 'up' | 'down': # Arrow keys up-down
                    direction = -1 if key == 'down' else 1
                    oy = self.mm.orientation[indices_y,indices_x,1] if self.mm.in_plane else 1
                    update_indices = xp.where(oy*self.mm.m[indices_y,indices_x]*direction < 0)
                case 'u' | 'd': # If U (D) pressed, set m to +1 (-1)
                    update_indices = xp.where(self.mm.m[indices_y, indices_x] != (+1 if key == 'u' else -1))
                case _:
                    return indices_x, indices_y # Unknown key, so everything touched by the mouse may just flip
            return indices_x[update_indices], indices_y[update_indices]

        self.polygon = [] # list of (y, x) tuples
        def onclick(event):
            tup = get_idx(event)[::-1]
            if tup == (None, None): return
            match event.button:
                case MouseButton.LEFT: # If left mouse button clicked: change state of clicked magnet
                    toggle(mouse_event=event)
                case MouseButton.RIGHT: # If right mouse button clicked: start building self.polygon
                    self.polygon.append(tup)

        def update_polygonplotted():
            polygon = np.asarray(self.polygon).reshape(-1, 2)
            if polygon.size > 0:
                x = self.mm.xx[0, polygon.T[1]]/self.unit_axes_factor
                y = self.mm.yy[polygon.T[0], 0]/self.unit_axes_factor
                self.polygonplotted.set_xy(np.asarray([asnumpy(x), asnumpy(y)]).T)
            else:
                self.polygonplotted.set_xy([(0, 0)])
            self.ax.draw_artist(self.polygonplotted)
            self.figure.canvas.draw()
        self.polygonplotted = Polygon([(0, 0)], facecolor='#88888888', edgecolor='#555')
        self.ax.add_artist(self.polygonplotted)
        update_polygonplotted()

        def onmotion(event): 
            tup = get_idx(event)[::-1]
            if tup == (None, None): return
            match event.button:
                case MouseButton.LEFT: # If left mouse button dragged: change state of all magnets the mouse passes
                    if self.last_toggle_index != (None, None):
                        indices_y, indices_x = xp.asarray(bresenham(self.last_toggle_index, tup))[1:,:].T
                    else:
                        indices_y, indices_x = tup
                    toggle(indices_x, indices_y, mouse_event=event)
                case MouseButton.RIGHT: # If right mouse button dragged: enlarge polygon behind-the-scenes
                    if len(self.polygon) != 0:
                        if self.polygon[-1] == tup: return # Don't add identical successive vertices to self.polygon
                    self.polygon.append(tup)
                    update_polygonplotted()

        def onrelease(event):
            self.last_toggle_index = (None, None)
            match event.button:
                case MouseButton.RIGHT: # If right mouse button released: draw polygon
                    if len(self.polygon) == 0: return
                    polygon = np.asarray(self.polygon).reshape(-1, 2)
                    xx, yy = np.meshgrid(np.arange(np.min(polygon[:,1]), np.max(polygon[:,1]) + 1), np.arange(np.min(polygon[:,0]), np.max(polygon[:,0]) + 1)) # Tight index-grid surely containing the polygon
                    possible_x, possible_y = xx.reshape(-1), yy.reshape(-1)
                    in_polygon, on_edge = inpoly2(np.asarray([possible_y, possible_x]).T, polygon)
                    in_polygon = np.logical_or(in_polygon, on_edge)
                    toggle(possible_x[in_polygon], possible_y[in_polygon], mouse_event=event)
                    self.polygon = []
                    update_polygonplotted()

        if self.gui.editable:
            cid = self.figure.canvas.mpl_connect('button_press_event', onclick)
            mid = self.figure.canvas.mpl_connect('motion_notify_event', onmotion)
            rid = self.figure.canvas.mpl_connect('button_release_event', onrelease)

    def show_info(self): # Could also use third-party CTkMessagebox package for this
        information = """
            By clicking/dragging the mouse over the magnetization viewport, 
            the state of the magnets can be switched by the user in real-time.
                - Click left: toggles the state of the single magnet clicked.
                - Drag left: toggles all magnets along the path of the mouse.
                - Drag right: draws a polygon, toggling all magnets inside of it.

            Holding down keys while clicking/dragging can modify the 'toggling':
                - Arrow keys: toggles the magnets such that they point towards
                    the arrow key. For OOP ASI, up/down is identical to 'U'/'D'.
                - 'U'/'D' keys: set the magnetization to +1 or -1, respectively.
                    Makes most sense for OOP ASI, though IP also supports this.
        """
        messagebox.showinfo("Keyboard and mouse controls", dedent(information[1:]))

    def change_setting(self, setting, value, _update=True):
        match setting: # Some cleaning of some values that are possibly passed as the wrong class, e.g. str instead of Average
            case 'avg':
                value = Average.resolve(value)
        if not hasattr(self.settings, setting): # If the setting does not exist, then don't set it
            raise ValueError(f"The setting '{setting}' does not exist.")
        setattr(self.settings, setting, value)
        if _update: self.redraw(settings_changed=True)
    
    def change_settings(self, settings: dict|ViewSettings):
        if isinstance(settings, MagnetizationView.ViewSettings):
            self.settings = settings
        elif isinstance(settings, dict):
            for setting, value in settings:
                self.change_setting(setting, value, _update=False)
        else: raise TypeError('Parameter <settings> must be of type <dict> or <MagnetizationView.Settings>.')
        self.redraw(settings_changed=True)
    
    def get_settings(self, *settings):
        result = tuple(getattr(self.settings, setting) for setting in settings)
        return result if len(settings) > 1 else result[0]


    def change_mode(self, new_mode: DisplayMode|str):
        if isinstance(new_mode, str): new_mode = self.DisplayMode[new_mode.upper()]
        if not isinstance(new_mode, self.DisplayMode): raise ValueError(f"Argument <new_mode> must be type <str> or <MagnetizationView().DisplayMode>, but is <{type(new_mode)}>.")
        self.mode = new_mode
        # if self.colorbar is not None: self.colorbar.remove() #! MUST remove self.colorbar BEFORE removing self.content
        if self.content is not None: self.content.remove()
        colorbar_ax_kwargs = {'ax': self.ax} if self.colorbar is None else {'cax': self.colorbar.ax} # Use 'cax' if colorbar already exists

        ## HERE, WE DO SOME COMMANDS THAT ARE NEEDED ONLY WHEN SWITCHING BETWEEN DISPLAY MODES, LIKE TITLES AND TEXT, CHECKS ETC.
        im_placeholder = np.ones((1,1))
        match self.mode:
            case self.DisplayMode.MAGNETIZATION:
                self.ax.set_title(r"Magnetization $\overrightarrow{m}$")
                IP = self.mm.in_plane
                self.content = self.ax.imshow(im_placeholder, cmap=self.cmap_hsv, origin='lower', vmin=0 if IP else -1, vmax=2*np.pi if IP else 1, interpolation='antialiased', interpolation_stage='rgba', aspect=self.ax_aspect) # extent doesnt work perfectly with triangle or kagome but is still ok
                self.colorbar = plt.colorbar(self.content, **colorbar_ax_kwargs)
                self.colorbar.ax.get_yaxis().labelpad = 10 + 2*self.figparams['fontsize_colorbar']
            case self.DisplayMode.QUIVER:
                if not self.mm.in_plane: raise ValueError("Can only use DisplayMode.QUIVER for in-plane ASI.")
                self.ax.set_title(r"Magnetization $\overrightarrow{m}$")
                self.content = self.ax.quiver(asnumpy(self.mm.xx[self.mm.nonzero])/self.unit_axes_factor, asnumpy(self.mm.yy[self.mm.nonzero])/self.unit_axes_factor,
                                           np.ones(self.mm.n), np.ones(self.mm.n), color=self.cmap_hsv(np.ones(self.mm.n)), pivot='mid',
                                           scale=1.1/self.mm._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7, units='xy') # units='xy' makes arrows scale correctly when zooming
                self.colorbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=2*np.pi), cmap=self.cmap_hsv), **colorbar_ax_kwargs) # Quiver is not a Mappable, so we create our own ScalarMappable.
                self.colorbar.ax.get_yaxis().labelpad = 15
                self.colorbar.ax.set_ylabel(f"Magnetization angle [rad]", rotation=270, fontsize=self.figparams['fontsize_colorbar'])
            case self.DisplayMode.DOMAINS:
                if not hasattr(self.mm, 'get_domains'): raise ValueError("Can only use DisplayMode.DOMAINS for ASI that supports the .get_domains() method.")
                self.ax.set_title(r"Domains")
                cmap = colormaps['Greys'].copy()
                cmap.set_bad(color='#FFAAAA')
                self.content = self.ax.imshow(im_placeholder, cmap=cmap, origin='lower', extent=self.full_extent, interpolation='antialiased', interpolation_stage='rgba', aspect=self.ax_aspect)
                self.colorbar = plt.colorbar(self.content, **colorbar_ax_kwargs)
                self.colorbar.ax.get_yaxis().labelpad = 10 + 2*self.figparams['fontsize_colorbar']
                self.colorbar.ax.set_ylabel("Domain number", rotation=270, fontsize=self.figparams['fontsize_colorbar']) # TODO: use the number of domains when that has been implemented in .ASI module
            case self.DisplayMode.ENERGY:
                self.ax.set_title(r"Local energy $E_{int}$")
                self.content = self.ax.imshow(im_placeholder, origin='lower', extent=self.full_extent, interpolation='antialiased', interpolation_stage='rgba', aspect=self.ax_aspect)
                self.colorbar = plt.colorbar(self.content, **colorbar_ax_kwargs)
                self.colorbar.ax.get_yaxis().labelpad = 15
            case self.DisplayMode.FIELD:
                self.content = self.ax.imshow(im_placeholder, cmap=colormaps['inferno'], origin='lower', extent=self.full_extent, interpolation='antialiased', interpolation_stage='rgba', aspect=self.ax_aspect)
                self.colorbar = plt.colorbar(self.content, **colorbar_ax_kwargs)
                self.colorbar.ax.get_yaxis().labelpad = 15
            case self.DisplayMode.READOUT:
                if self.outputreader is None: raise AttributeError("Can not use DisplayMode.READOUT because no OutputReader was provided.")
                self.ax.set_title(r"Values of readout nodes")
        self.redraw(settings_changed=True)


    def redraw(self, settings_changed: bool = False):
        match self.mode:
            case self.DisplayMode.MAGNETIZATION:
                self._redraw_magnetization(settings_changed)
            case self.DisplayMode.QUIVER:
                self._redraw_quiver(settings_changed)
            case self.DisplayMode.DOMAINS:
                self._redraw_domains(settings_changed)
            case self.DisplayMode.ENERGY:
                self._redraw_energy(settings_changed)
            case self.DisplayMode.FIELD:
                self._redraw_field(settings_changed)
            case self.DisplayMode.READOUT:
                self._redraw_readout(settings_changed)
        self.canvas.draw()
        self.canvas.flush_events()

    def _redraw_magnetization(self, settings_changed: bool = False):
        avg, fill = self.settings.avg, self.settings.fill # Could also use self.get_settings, but then we don't get the type hints
        if settings_changed: # If averaging method changes, we need to update extent and colorbar text
            averaged_extent = _get_averaged_extent(self.mm, avg)/self.unit_axes_factor # List comp to convert to micrometre
            self.content.set_extent(averaged_extent)
            if avg is Average.POINT: # Then there is effectively no averaging taking place
                text = "Magnetization angle [rad]" if self.mm.in_plane else "Magnetization"
            else:
                text = "Averaged magnetization angle [rad]" if self.mm.in_plane else "Averaged magnetization"
                if self.figparams['text_averaging']: text += f"\n('{avg.name.lower()}' average{', PBC' if self.mm.PBC else ''})"
            self.colorbar.ax.set_ylabel(text, rotation=270, fontsize=self.figparams['fontsize_colorbar'])
        self.content.set_data(get_rgb(self.mm, m=self.mm.m, avg=avg, fill=fill))

    def _redraw_quiver(self, settings_changed: bool = False):
        color_quiver = self.settings.color_quiver
        mx, my = asnumpy(xp.multiply(self.mm.m, self.mm.orientation[:,:,0])[self.mm.nonzero]), asnumpy(xp.multiply(self.mm.m, self.mm.orientation[:,:,1])[self.mm.nonzero])
        self.content.set_UVC(mx/self.unit_axes_factor, my/self.unit_axes_factor)
        if settings_changed:
            self.colorbar.ax.set_visible(color_quiver)
        colorfield = self.cmap_hsv((np.arctan2(my, mx)/2/np.pi) % 1) if color_quiver else 'black'
        self.content.set_color(colorfield)

    def _redraw_domains(self, settings_changed: bool = False):
        domains = self.mm.get_domains()
        domains = xp.where(self.mm.occupation == 0, xp.nan, domains)
        lims = self.content.get_clim()
        self.content.set_data(asnumpy(domains))
        self.content.set_clim(vmin=min(lims[0], xp.nanmin(domains)), vmax=max(lims[1], xp.nanmax(domains)))

    def _redraw_energy(self, settings_changed: bool = False):
        energy_component, in_eV = self.settings.energy_component, self.settings.in_eV
        if settings_changed:
            # Update colormap
            cmap = 'seismic' if energy_component == 'E_barrier' else 'inferno' # TODO: select good colormaps and limits for this, perhaps scaled by kBT?
            self.content.set_cmap(cmap)
            # Change axes units
            E_unit = 'eV' if in_eV else 'J'
            colorbar_name = {
                'Total': "Total energy",
                'E_barrier': "Effective energy barrier",
                'DipolarEnergy': "Dipolar energy",
                'ZeemanEnergy': "Zeeman energy",
                'ExchangeEnergy': "Exchange energy"
            }.get(energy_component, "Energy")
            self.colorbar.ax.set_ylabel(f"{colorbar_name} [{E_unit}]", rotation=270, fontsize=self.figparams['fontsize_colorbar'])
            match energy_component:
                case 'Total': self.E = lambda: self.mm.E
                case 'E_barrier': self.E = lambda: self.mm.E_B - self.mm.E # Crude approximation of the effective energy barrier for each magnet
                case _:
                    self._displayed_energy = self.mm.get_energy(energy_component)
                    self.E = lambda: self._displayed_energy.E
        E = xp.where(self.mm.m != 0, self.E(), xp.nan) # Change empty spots to NaN
        if in_eV: E = J_to_eV(E)
        self.content.set_data(asnumpy(E))
        lims = self.content.get_clim()
        if energy_component == 'E_barrier':
            maxabs = max(abs(xp.nanmin(E)), abs(xp.nanmax(E)))
            vmin, vmax = -maxabs, maxabs
        else:
            vmin, vmax = xp.nanmin(E), xp.nanmax(E)
        if settings_changed: # Then we dont have a 'previous' colorscale to adjust
            self.content.set_clim(vmin=vmin, vmax=vmax) # For some weird reason, this can not be done in one go
            self.content.set_clim(vmin=vmin, vmax=vmax) # (weird bug where only one of the limits gets updated)
        else:
            self.content.set_clim(vmin=min(lims[0], vmin), vmax=max(lims[1], vmax))
        # TODO: add an option to scale the colorbar to kBT

    def _redraw_field(self, settings_changed: bool = False):
        field_type, in_eV = self.settings.field_type, self.settings.in_eV
        field = {
            'T': self.mm.T,
            'E_B': J_to_eV(self.mm.E_B) if in_eV else self.mm.E_B,
            'moment': self.mm.moment
        }[field_type]
        if settings_changed:
            title = {
                'T': "Temperature profile",
                'E_B': "Intrinsic energy barrier profile",
                'moment': "Magnetic moment profile"
            }[field_type]
            colorbar_label = {
                'T': "Temperature [K]",
                'E_B': f"Energy barrier [{'eV' if in_eV else 'J'}]",
                'moment': "Magnetic moment [Am²]"
            }[field_type]
            self.ax.set_title(title)
            self.colorbar.ax.set_ylabel(colorbar_label, rotation=270, fontsize=self.figparams['fontsize_colorbar'])
            self.content.set_clim(vmin=xp.nanmin(field), vmax=xp.nanmax(field))
        field = xp.where(self.mm.m != 0, field, xp.nan)
        self.content.set_data(asnumpy(field))
        lims = self.content.get_clim()
        self.content.set_clim(vmin=min(lims[0], xp.nanmin(field)), vmax=max(lims[1], xp.nanmax(field)))

    def _redraw_readout(self, settings_changed: bool = False):
        raise NotImplementedError("DisplayMode.READOUT is not yet available, as it is unclear how to consistently do this for various OutputReaders.")


class MagnetizationViewSettingsTabView(ctk.CTkTabview):
    """ This thing does not concern itself with the ASI. It only cares about the MagnetizationView. """
    def __init__(self, master, magnetization_view_widget: MagnetizationView, **kwargs):
        self.name_to_mode = {
            "Magnetization": MagnetizationView.DisplayMode.MAGNETIZATION,
            "Quiver": MagnetizationView.DisplayMode.QUIVER,
            "Domains": MagnetizationView.DisplayMode.DOMAINS,
            "Energy": MagnetizationView.DisplayMode.ENERGY,
            "Scalar field": MagnetizationView.DisplayMode.FIELD,
            "Readout": MagnetizationView.DisplayMode.READOUT
            }
        self.mode_to_name = {v: k for k, v in self.name_to_mode.items()}
        super().__init__(master, command=lambda: self.magviewwidget.change_mode(self.name_to_mode[self._current_name]), **kwargs)
        self.magviewwidget = magnetization_view_widget
        self.mm = self.magviewwidget.mm

        ## Tab 1: MAGNETIZATION
        self.tab_MAGNETIZATION = self.add(self.mode_to_name[MagnetizationView.DisplayMode.MAGNETIZATION])
        available_averages = {avg: avg.name.capitalize() for avg in Average}
        self.option_avg = ctk.CTkOptionMenu(self.tab_MAGNETIZATION, values=list(available_averages.values()), command=self.init_settings_magview)
        self.option_avg.pack(pady=10, padx=10)
        self.option_avg.set(available_averages[Average.resolve(self.mm._get_appropriate_avg())])
        self.option_fill = ctk.CTkSwitch(self.tab_MAGNETIZATION, text="Fill", command=self.init_settings_magview)
        self.option_fill.pack(pady=10, padx=10)
        self.option_fill.select() # Set to 'on' by default

        ## Tab 2: QUIVER
        if self.mm.in_plane: # Only add the quiver tab if we have an in-plane system
            self.tab_QUIVER = self.add(self.mode_to_name[MagnetizationView.DisplayMode.QUIVER])
        else:
            self.tab_QUIVER = ctk.CTkFrame(self.master) # Dummy frame, so it does not show up in the tabs
        self.option_color_quiver = ctk.CTkSwitch(self.tab_QUIVER, text="Color", command=self.init_settings_magview)
        self.option_color_quiver.pack(pady=10, padx=10)
        self.option_color_quiver.select() # Set to 'on' by default

        ## Tab 3: DOMAINS
        if hasattr(self.mm, 'get_domains'):
            self.tab_DOMAINS = self.add(self.mode_to_name[MagnetizationView.DisplayMode.DOMAINS])
        else:
            self.tab_DOMAINS = ctk.CTkFrame(self.master) # Dummy frame, so it does not show up in the tabs

        ## Tab 4: ENERGY
        self.tab_ENERGY = self.add(self.mode_to_name[MagnetizationView.DisplayMode.ENERGY])
        self.name_to_energycomponent = {"Effective barrier": 'E_barrier', "Total energy": 'Total'}
        valid_components = list(self.name_to_energycomponent.keys()) + [e.__class__.__name__ for e in self.mm._energies]
        self.option_energy_component = ctk.CTkOptionMenu(self.tab_ENERGY, values=valid_components, command=self.init_settings_magview)
        self.option_energy_component.pack(pady=10, padx=10)
        self.option_energy_component.set('Total') # Set to 'Total' by default
        self.option_in_eV = ctk.CTkSwitch(self.tab_ENERGY, text="Energy in eV", command=self.init_settings_magview)
        self.option_in_eV.pack(pady=10, padx=10)
        self.option_in_eV.select() # Set to 'on' by default
        # TODO: change option_subtract_barrier to multibutton with options: E: Zeeman, E: dipolar, (E: exchange), Total interaction energy, Effective barrier

        ## Tab 5: FIELD
        self.tab_FIELD = self.add(self.mode_to_name[MagnetizationView.DisplayMode.FIELD])
        self.option_field_type = ctk.CTkOptionMenu(self.tab_FIELD, values=self.magviewwidget.settings.available_field_types, command=self.init_settings_magview)
        self.option_field_type.pack(pady=10, padx=10)
        self.option_field_type.set(self.magviewwidget.settings.available_field_types[0]) # Set a default field type

        ## Tab 6: READOUT # Not implemented (yet?)
        # self.tab_READOUT = self.add(self.mode_to_name(MagnetizationView.DisplayMode.READOUT))

        self.init_settings_magview()
    
    def init_settings_magview(self, *args, **kwargs): # Need *args to consistently use this in 'command' parameter of CTk widgets
        """ Call this once at the start, to synchronize the self.magviewwidget.settings with all the buttons in this tabview. """
        option_energy_component = self.option_energy_component.get()
        settings = MagnetizationView.ViewSettings(
            avg=self.option_avg.get(),
            fill=self.option_fill.get(),
            color_quiver=self.option_color_quiver.get(),
            energy_component=self.name_to_energycomponent.get(option_energy_component, option_energy_component),
            in_eV=self.option_in_eV.get(),
            field_type=self.option_field_type.get()
        )
        self.magviewwidget.change_settings(settings)


class ASISettingsFrame(ctk.CTkFrame):
    @dataclass
    class ASISettings:
        UPDATE_SCHEME: Literal['Néel', 'Glauber'] = 'Néel'
        t_max: float = 1.
        attempt_freq: float = 1e10
        Q: float = 0.05
        # TODO: choose grid selection method in Glauber

        def __post_init__(self):
            self.UPDATE_SCHEME = str(self.UPDATE_SCHEME)
            if self.UPDATE_SCHEME not in self.available_update_modes: raise ValueError(f"<UPDATE_MODE> can only be any of {self.available_update_modes}.")
            self.t_max = float(self.t_max)
            self.attempt_freq = float(self.attempt_freq)
            self.Q = float(self.Q) # Range 0-inf, but logical range 0-1, and goes logarithmically

        @property
        def available_update_modes(self): return ['Néel', 'Glauber']

    def __init__(self, master, gui: GUI, **kwargs):
        super().__init__(master, **kwargs)
        self.gui = gui
        self.mm = gui.mm
        self.settings = ASISettingsFrame.ASISettings(UPDATE_SCHEME=self.mm.params.UPDATE_SCHEME)

        def change_updatemode():
            self.settings.UPDATE_SCHEME = self.updatescheme_tabview._current_name
            self.apply_settings_to_mm()

        ## Update scheme
        self.updatescheme_tabview = ctk.CTkTabview(self, command=change_updatemode, height=120)
        self.updatescheme_tabview.pack(pady=(0,10), padx=10)

        self.Néel = self.updatescheme_tabview.add("Néel")
        self.Néel.grid_columnconfigure(0, weight=2)
        self.Néel.grid_columnconfigure(1, weight=1)
        Néel_tmax_values = list(10.**np.arange(-10, 10)) + [1e100, np.inf]
        self.Néel_t_max_var = tk.DoubleVar()
        self.Néel_t_max = ttk.Spinbox(self.Néel, width=15, command=lambda: self.change_setting('t_max', float(self.Néel_t_max.get())), values=Néel_tmax_values, textvariable=self.Néel_t_max_var)
        self.Néel_t_max.grid(row=0, column=1, padx=10, sticky=ctk.W)
        self.Néel_t_max_text = ctk.CTkLabel(self.Néel, text="t_max [s]: ", anchor=ctk.E)
        self.Néel_t_max_text.grid(row=0, column=0, sticky=ctk.E)
        Néel_attemptfreq_values = [1., 1.5, 2, 3, 5, 7.5, 10., 15., 20., 30., 50., 75., 100.]
        self.Néel_attemptfreq_var = tk.DoubleVar()
        self.Néel_attempt_freq = ttk.Spinbox(self.Néel, width=15, command=lambda: self.change_setting('attempt_freq', 1e9*float(self.Néel_attempt_freq.get())), values=Néel_attemptfreq_values, textvariable=self.Néel_attemptfreq_var)
        self.Néel_attempt_freq.grid(row=1, column=1, padx=10, sticky=ctk.W)
        self.Néel_attempt_freq_text = ctk.CTkLabel(self.Néel, text="Attempt freq. [GHz]: ", anchor=ctk.E)
        self.Néel_attempt_freq_text.grid(row=1, column=0, sticky=ctk.E, padx=(10,0))

        self.Glauber = self.updatescheme_tabview.add("Glauber")
        self.Glauber.grid_columnconfigure(0, weight=2)
        self.Glauber.grid_columnconfigure(1, weight=1)
        Glauber_Q_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, np.inf]
        self.Glauber_Q_var = tk.DoubleVar()
        self.Glauber_Q = ttk.Spinbox(self.Glauber, width=7, command=lambda: self.change_setting('Q', float(self.Glauber_Q.get())), values=Glauber_Q_values, textvariable=self.Glauber_Q_var)
        self.Glauber_Q.grid(row=0, column=1, padx=10, sticky=ctk.W)
        self.Glauber_Q_text = ctk.CTkLabel(self.Glauber, text="Q: ", anchor=ctk.E)
        self.Glauber_Q_text.grid(row=0, column=0, sticky=ctk.E)

        self.get_settings_from_mm()
    
    def apply_settings_to_mm(self):
        self.mm.params.UPDATE_SCHEME = self.settings.UPDATE_SCHEME
    
    def get_settings_from_mm(self):
        self.settings.UPDATE_SCHEME = self.mm.params.UPDATE_SCHEME
        self.updatescheme_tabview.set(self.settings.UPDATE_SCHEME)
        sig_Néel = inspect.signature(self.mm._update_Néel)
        self.Néel_t_max_var.set(float(sig_Néel.parameters['t_max'].default))
        self.Néel_attemptfreq_var.set(float(sig_Néel.parameters['attempt_freq'].default)*1e-9)
        sig_Glauber = inspect.signature(self.mm._update_Glauber)
        self.Glauber_Q_var.set(sig_Glauber.parameters['Q'].default)

    def change_setting(self, setting, value, _redraw=True):
        if not hasattr(self.settings, setting): # If the setting does not exist, then don't set it
            raise ValueError(f"The setting '{setting}' does not exist.")
        setattr(self.settings, setting, value)
        self.apply_settings_to_mm()
        if _redraw: self.gui.magnetization_view.redraw(settings_changed=True)
    
    def change_settings(self, settings: dict|ASISettings):
        if isinstance(settings, ASISettingsFrame.ASISettings):
            self.settings = settings
        elif isinstance(settings, dict):
            for setting, value in settings:
                self.change_setting(setting, value, _redraw=False)
        else: raise TypeError('Parameter <settings> must be of type <dict> or <ASISettingsFrame.Settings>.')
        self.gui.magnetization_view.redraw(settings_changed=True)
    
    def get_settings(self, *settings):
        result = tuple(getattr(self.settings, setting) for setting in settings)
        return result if len(settings) > 1 else result[0]







# TODO: INPUT: applying a single input value through the Inputter()
# TODO: VIEW: showing an OutputReader() readout (if an OutputReader was provided, and each node has coordinates)
#       Could do this using scatter plot, with size determined by nearest output nodes
#       Other option is to use voronoi diagram, but then I don't know how to color the regions that extend to infinity, plus is probably quite slow to draw 
# TODO: INPUT: simple console that can eval() code in the namespace where GUI was created.
#       Will have to use inspect module in __init__ probably, to determine locals there and save them to self._locals
#       This can be done with a popup window where one can run some (possibly multi-line) <command>,
#       which behind-the-scenes can be executed upon pressing a 'run' button using something like 
#        IPython.core.interactiveshell.InteractiveShell().run_cell("x += 5")
#       but then with an InteractiveShell() that has been given the right scope.
#       After each 'run' button press (i.e. each command), close the popup and redraw the MagnetizationView.
#       If the return value is not None, display it in a new popup? (has to be big enough to scroll through array output etc.)
# TODO: VIEW: show the current time, switches, MCsteps etc.
#       Have a button to reset them to zero
# TODO: MISC: allow recording what happens to a video
#       Can have a switch, to toggle adding a new frame every time the plot is updated
#       Also a manual button

def show(mm, **kwargs):
    gui = GUI(mm, **kwargs)
    gui.show()

def test(in_plane=False, **kwargs):
    from .ASI import OOP_Square, IP_Pinwheel
    if in_plane:
        mm = IP_Pinwheel(230e-9, 40, T=300, E_B=0, params=SimParams(UPDATE_SCHEME="Glauber"))
    else:
        mm = OOP_Square(230e-9, 200, T=300, E_B=0, params=SimParams(UPDATE_SCHEME="Néel"))
    show(mm, **kwargs)
