# src/main_gui.py
"""
Main graphical user interface for PCC-Explorer.
Built with tkinter/ttkbootstrap, provides file selection, parameter
configuration, and analysis execution with progress feedback.
"""

import sys as _sys


def _harden_console_encoding():
    """
    Never let a printed character break a code path.

    A Windows console runs cp1252, so printing 'v' or an arrow raises
    UnicodeEncodeError. Where such a print sits inside a try block the damage is
    far worse than a missing character: read_broadcast_file parsed a whole
    11 MB navigation file, printed a check mark on the last line of its try, and
    the resulting UnicodeEncodeError was swallowed by its own `except Exception`
    - so every parsed record was thrown away and the caller was told the orbit
    could not be loaded. Reconfiguring the streams removes the entire class.
    """
    for stream in (_sys.stdout, _sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass    # frozen builds may hand us a stream without reconfigure()


_harden_console_encoding()


def _ensure_project_root_on_path():
    """Make `python src/main_gui.py` work as well as `python -m src.main_gui`.

    Every import below is absolute (`from src....`), so the PROJECT ROOT has to
    be importable. `python -m src.main_gui`, which launch_gui.bat uses, puts the
    working directory on sys.path and is fine. Running the file directly does
    not: Python puts the SCRIPT's own folder, src/, on sys.path instead, and the
    very first import dies with "No module named 'src'".

    That is a confusing failure, because the working directory looks right and
    the venv is active. Adding the
    root here makes both invocations behave the same. A frozen build already
    resolves its own imports, so it is skipped there.
    """
    if getattr(_sys, "frozen", False):
        return
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if root not in _sys.path:
        _sys.path.insert(0, root)


_ensure_project_root_on_path()

from src.main_cli import main as run_cli_mode
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cartopy.crs as ccrs
import re
from PIL import Image, ImageTk, ImageEnhance

# Shared opt-in dialog for the IfE software mailing list. Both import forms are
# tried because PyInstaller runs main_gui as __main__ in the frozen build.
# Never let a missing copy stop the program from starting; canonical in shared/.
try:
    from src import mailing_list
except Exception:
    try:
        import mailing_list
    except Exception:
        mailing_list = None

# Version and release date, from src/tool_version.py. The same pair sits in the
# README tag that the PCC-Suite launcher reads, so the launcher does not have
# to guess a release from a file timestamp. Both import forms are tried because
# PyInstaller runs main_gui as __main__ in the frozen build.
try:
    from src import tool_version, version_info
    VERSION_TEXT = version_info.about_line(tool_version)
except Exception:
    try:
        import tool_version
        import version_info
        VERSION_TEXT = version_info.about_line(tool_version)
    except Exception:
        VERSION_TEXT = "Version: unknown"

# --- Project Root Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Display name mapping for parameter labels ---
PARAM_DISPLAY_NAMES = {
    'Tropo': 'Tropospheric Parameter',
    'Tropo_Gn': 'Tropospheric Parameter (North Gradient)',
    'Tropo_Ge': 'Tropospheric Parameter (East Gradient)',
    'Clock_GPS': 'GPS Clock',
    'Clock_Galileo': 'Galileo Clock',
    'Clock_GLONASS': 'GLONASS Clock',
    'Clock_BDS/BeiDou': 'BDS Clock',
    'Clock_QZSS': 'QZSS Clock',
}

# For global/regional plot titles and colorbars (full names + position 'component' suffix)
GLOBAL_DISPLAY_NAMES = {
    **PARAM_DISPLAY_NAMES,
    'North': 'North component',
    'East': 'East component',
    'Up': 'Up component',
}

# --- Try importing necessary modules ---
try:
    from src.data_io import read_antex_file, read_config_file, save_timeline_results_to_txt
    from src.processing_core import (
        run_processing_pipeline, 
        run_timeline_pipeline, 
        run_folder_pipeline,
        run_global_pipeline
    )
    from src.geodesy import ell_to_ecef, ecef_to_ell
    from src.plotting import plot_skyplot, plot_world_map, plot_results_bar_chart, plot_timeline
    from src.gui_progress_handler import run_with_progress
    from src.input_validation import ConfigValidator, ValidationError
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- PATH HELPER FOR EXE ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__) 
        # If running as script, remove 'src' from relative path if it's there twice
        # because dirname(__file__) is already inside src/
        if relative_path.startswith("src"):
             # In dev mode: dirname is .../src. relative is src/assets. 
             # We want .../src/assets. 
             # So we strip the 'src' from relative path or adjust logic.
             # Simplest approach for dev structure (src/main_gui.py and src/assets):
             relative_path = relative_path.replace("src\\", "").replace("src/", "")

    return os.path.join(base_path, relative_path)

class CreateToolTip(object):
    """
    create a tooltip for a given widget
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 250   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

        if tw:
            tw.destroy()

class ScrollableFrame(ttk.Frame):
    """
    A scrollable frame that can contain any widgets (Radiobuttons, Checkbuttons, etc.)
    """
    def __init__(self, container, height=150, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.scrollable_frame.bind("<Destroy>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_mousewheel(self, event):
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

class App(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.analysis_results = None
        self.antex_data_1 = None
        self.antex_data_2 = None
        self.antex_data_1 = None
        self.antex_data_2 = None
        self.azimuth_masks_list = [] 
        
        # Signal Selection Variables
        self.selected_signal_var = tk.StringVar()  # For Single Mode (Radio)
        self.signal_vars = {}        # For Multi Mode ({signal: BooleanVar})
        self.current_signals = []    # List of currently available signals
        
        self.logo_img = None 
        self.luh_logo_img = None 
        self.last_pos_mode = "Ellipsoidal"

        self._setup_window()
        self._create_widgets()
        
        self.start_date_var.trace_add("write", self._update_insights)
        self.orbit_type_var.trace_add("write", self._update_insights)
        # The orbit note depends on the constellation as well as the
        # date, so it has to be recomputed when the signal selection changes.
        self.selected_signal_var.trace_add("write", self._update_insights)
        self.multifreq_var.trace_add("write", self._update_insights)
        self.pos_type.trace_add("write", self._on_pos_mode_change)
        
        # Auto-refresh when file paths change
        self.antex1_path.trace_add("write", self._on_file1_path_changed)
        self.antex2_path.trace_add("write", self._on_file2_path_changed)
        
        # Trigger validation when Area mode fields change
        self.lat_min_var.trace_add("write", self._validate_inputs)
        self.lat_max_var.trace_add("write", self._validate_inputs)
        self.lon_min_var.trace_add("write", self._validate_inputs)
        self.lon_max_var.trace_add("write", self._validate_inputs)
        self.grid_step_var.trace_add("write", self._validate_inputs)
        self.pos_type.trace_add("write", self._validate_inputs)  # Also re-validate on mode change
        
        self._update_insights()
        self.update_antex_mode()
        self._update_pos_labels_only()
        self.update_mask_widgets()
        self.update_frequency_mode()
        self.update_position_mode()
        self._validate_inputs()  # Initial validation
        
        # Add close confirmation dialog (N2)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_window(self):
        self.title("PCC-Explorer")
        self.geometry("1200x1000") 

    def _on_closing(self):
        """Handle window close event with optional confirmation dialog."""
        if messagebox.askokcancel("Quit", "Do you want to close PCC-Explorer?"):
            self.destroy() 

    def _create_widgets(self):
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(fill="x")

        # 1. Left: IfE Logo
        left_frame = ttk.Frame(top_bar)
        left_frame.pack(side="left")
        
        # Logo path loading
        ife_logo_path = resource_path(os.path.join('src', 'assets', 'ife_logo.png'))
        
        if os.path.exists(ife_logo_path):
            try:
                pil_image = Image.open(ife_logo_path)
                aspect_ratio = pil_image.width / pil_image.height
                target_height = 50
                target_width = int(target_height * aspect_ratio)
                resized_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(resized_image)
                ttk.Label(left_frame, image=self.logo_img).pack(side="left", padx=(0, 15))
            except Exception: pass
        
        text_frame = ttk.Frame(left_frame)
        text_frame.pack(side="left")
        ttk.Label(text_frame, text="PCC-Explorer", font=("Helvetica", 20, "bold")).pack(anchor="w")
        ttk.Label(text_frame, text="Institut für Erdmessung (IfE)", font=("Helvetica", 12), bootstyle="light").pack(anchor="w")

        # 2. Far Right: LUH Logo
        luh_frame = ttk.Frame(top_bar)
        luh_frame.pack(side="right", padx=(20, 0)) 
        
        # Logo path loading
        luh_logo_path = resource_path(os.path.join('src', 'assets', 'luh_logo.png'))

        if os.path.exists(luh_logo_path):
            try:
                pil_luh = Image.open(luh_logo_path)
                target_h = 60
                aspect = pil_luh.width / pil_luh.height
                target_w = int(target_h * aspect)
                resized_luh = pil_luh.resize((target_w, target_h), Image.Resampling.LANCZOS)
                enhancer = ImageEnhance.Sharpness(resized_luh)
                final_luh = enhancer.enhance(1.5)
                self.luh_logo_img = ImageTk.PhotoImage(final_luh)
                ttk.Label(luh_frame, image=self.luh_logo_img).pack(side="right")
            except Exception as e: 
                print(f"Warning loading LUH logo: {e}")

        # 3. Middle-Right: Info Buttons
        btn_frame = ttk.Frame(top_bar)
        btn_frame.pack(side="right", padx=20)
        ttk.Button(btn_frame, text="Contact", command=self._show_contact_info, bootstyle="outline-info").pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Information", command=self._show_about_info, bootstyle="outline-secondary").pack(side="left", padx=5)

        ttk.Separator(self, orient='horizontal').pack(fill='x', pady=(0, 10))

        # --- Main Content ---
        proc_file_frame = ttk.LabelFrame(self, text="Processing File Options", padding=10)
        proc_file_frame.pack(padx=20, pady=(0, 10), fill="x")
        btn_container = ttk.Frame(proc_file_frame)
        btn_container.pack(fill="x", expand=True)
        ttk.Button(btn_container, text="Load Existing Configuration File", command=self._load_processing_file, bootstyle="info-outline").pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(btn_container, text="Create New Configuration File", command=self._show_create_config_dialog, bootstyle="success-outline").pack(side="left", fill="x", expand=True, padx=(5, 0))



        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(main_container, highlightthickness=0, bg='#222222')
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            if sys.platform.startswith('win'): delta = int(-1 * (event.delta / 120))
            elif sys.platform.startswith('darwin'): delta = event.delta
            else: 
                delta = 0
                if event.num == 4: delta = -1
                elif event.num == 5: delta = 1
            canvas.yview_scroll(delta, "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        if sys.platform.startswith('linux'):
             canvas.bind_all("<Button-4>", _on_mousewheel)
             canvas.bind_all("<Button-5>", _on_mousewheel)

        main_frame = ttk.Frame(scrollable_frame, padding=10)
        main_frame.pack(fill="both", expand=True)
        left_column = ttk.Frame(main_frame)
        left_column.pack(side="left", fill="y", anchor="n", padx=(0, 10), ipadx=10, expand=False)
        right_column = ttk.Frame(main_frame)
        right_column.pack(side="left", fill="both", anchor="n", padx=(10, 0), ipadx=10, expand=True)

        self._create_file_selection_widgets(left_column)
        self._create_position_widgets(left_column)
        self._create_time_widgets(left_column)
        
        # --- Input Validation Feedback Box ---
        validation_frame = ttk.LabelFrame(left_column, text="Input Validation", padding=10)
        validation_frame.pack(pady=10, fill="x")
        self.validation_text = tk.StringVar(value="✅ All inputs look valid.")
        self.validation_label = ttk.Label(validation_frame, textvariable=self.validation_text, wraplength=300, justify="left")
        self.validation_label.pack(fill="x")
        
        # Add trace callbacks for real-time validation
        self.samplerate_var.trace_add('write', self._validate_inputs)
        self.coord1_var.trace_add('write', self._validate_inputs)
        self.coord2_var.trace_add('write', self._validate_inputs)
        self.coord3_var.trace_add('write', self._validate_inputs)
        self.start_date_var.trace_add('write', self._validate_inputs)
        self.end_date_var.trace_add('write', self._validate_inputs)
        self.start_time_var.trace_add('write', self._validate_inputs)
        self.end_time_var.trace_add('write', self._validate_inputs)
        self.interval_var.trace_add('write', self._validate_inputs)
        self.antex1_path.trace_add('write', self._validate_inputs)
        self.antex2_path.trace_add('write', self._validate_inputs)
        
        insights_frame = ttk.LabelFrame(right_column, text="Analysis Insights", padding=10)
        insights_frame.pack(pady=10, fill="x")
        self.insights_text = tk.StringVar()
        self.insights_label = ttk.Label(insights_frame, textvariable=self.insights_text, wraplength=450, justify="left")
        self.insights_label.pack(fill="x")

        self._create_model_selection_widgets(right_column)
        self._create_mask_widgets(right_column)
        

        # --- Plot Settings ---
        self.color_scheme_var = tk.StringVar(value="Distinct")
        self.color_schemes = {
            "Distinct": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],  # 7 distinct colors
            "Paper Grey": ['#666666', '#666666', '#666666', '#666666', '#666666', '#666666', '#666666'],  # Uniform grey for academic papers
            "Grayscale": ['#333333', '#444444', '#555555', '#666666', '#777777', '#999999', '#bbbbbb'],
            "Warm": ['#e74c3c', '#e67e22', '#f1c40f', '#d35400', '#c0392b', '#e74c3c', '#d35400'],
            "Cool": ['#3498db', '#1abc9c', '#2ecc71', '#9b59b6', '#34495e', '#2980b9', '#27ae60']
        }
        self.y_axis_mode_var = tk.StringVar(value="Auto")
        self.y_axis_min_var = tk.StringVar(value="-5")
        self.y_axis_max_var = tk.StringVar(value="5")
        
        # Colorbar scaling for Global/Regional analysis
        self.colorbar_mode_var = tk.StringVar(value="Auto")
        self.colorbar_min_var = tk.StringVar(value="-2")
        self.colorbar_max_var = tk.StringVar(value="2")

        button_frame = ttk.Frame(self)
        button_frame.pack(side="bottom", pady=20, fill="x")
        
        # Plot settings row
        settings_row = ttk.Frame(button_frame)
        ttk.Label(settings_row, text="Color Scheme:").pack(side="left", padx=(0, 5))
        ttk.Combobox(settings_row, textvariable=self.color_scheme_var, values=list(self.color_schemes.keys()), state="readonly", width=10).pack(side="left", padx=(0, 15))
        ttk.Label(settings_row, text="Y-Axis:").pack(side="left", padx=(0, 5))
        self.y_axis_combo = ttk.Combobox(settings_row, textvariable=self.y_axis_mode_var, values=["Auto", "Symmetric", "Shared", "Custom"], state="readonly", width=10)
        self.y_axis_combo.pack(side="left", padx=(0, 5))
        self.y_axis_combo.bind("<<ComboboxSelected>>", self._on_y_axis_mode_change)
        
        # Custom Y-axis min/max fields (initially hidden)
        self.y_axis_custom_frame = ttk.Frame(settings_row)
        ttk.Label(self.y_axis_custom_frame, text="Min:").pack(side="left", padx=(5, 2))
        self.y_axis_min_entry = ttk.Entry(self.y_axis_custom_frame, textvariable=self.y_axis_min_var, width=5)
        self.y_axis_min_entry.pack(side="left", padx=(0, 5))
        ttk.Label(self.y_axis_custom_frame, text="Max:").pack(side="left", padx=(0, 2))
        self.y_axis_max_entry = ttk.Entry(self.y_axis_custom_frame, textvariable=self.y_axis_max_var, width=5)
        self.y_axis_max_entry.pack(side="left")
        # Don't pack the frame yet - it will be shown when "Custom" is selected
        
        # Colorbar scaling for Global/Regional analysis
        ttk.Separator(settings_row, orient='vertical').pack(side="left", fill="y", padx=10)
        ttk.Label(settings_row, text="Colorbar (Global):").pack(side="left", padx=(0, 5))
        self.colorbar_combo = ttk.Combobox(settings_row, textvariable=self.colorbar_mode_var, values=["Auto", "Custom"], state="readonly", width=8)
        self.colorbar_combo.pack(side="left", padx=(0, 5))
        self.colorbar_combo.bind("<<ComboboxSelected>>", self._on_colorbar_mode_change)
        
        # Custom colorbar min/max fields (initially hidden)
        self.colorbar_custom_frame = ttk.Frame(settings_row)
        ttk.Label(self.colorbar_custom_frame, text="Min:").pack(side="left", padx=(5, 2))
        self.colorbar_min_entry = ttk.Entry(self.colorbar_custom_frame, textvariable=self.colorbar_min_var, width=5)
        self.colorbar_min_entry.pack(side="left", padx=(0, 5))
        ttk.Label(self.colorbar_custom_frame, text="Max:").pack(side="left", padx=(0, 2))
        self.colorbar_max_entry = ttk.Entry(self.colorbar_custom_frame, textvariable=self.colorbar_max_var, width=5)
        self.colorbar_max_entry.pack(side="left")
        # Don't pack the frame yet - it will be shown when "Custom" is selected
        
        settings_row.pack(pady=(0, 10))
        
        btn_container = ttk.Frame(button_frame)
        ttk.Button(btn_container, text="Run Single Analysis", command=self.run_single_analysis, bootstyle="primary").pack(side="left", padx=5)
        ttk.Button(btn_container, text="Run Time Series Analysis", command=self.run_timeline_analysis, bootstyle="success").pack(side="left", padx=5)
        ttk.Button(btn_container, text="Run Global/Regional Analysis", command=self.run_global_analysis, bootstyle="success").pack(side="left", padx=5) 
        self.skyplot_button = ttk.Button(btn_container, text="Show Skyplot", command=self.show_skyplot, bootstyle="secondary")
        self.skyplot_button.pack(side="left", padx=5)
        ttk.Button(btn_container, text="Close All Figure Windows", command=self._close_all_figures, bootstyle="warning-outline").pack(side="left", padx=5)
        ttk.Button(btn_container, text="Quit App", command=self.destroy, bootstyle="danger-outline").pack(side="left", padx=5)
        btn_container.pack()

    def _show_contact_info(self):
        """Custom Contact dialog with clickable email and web links."""
        import webbrowser
        
        dialog = tk.Toplevel(self)
        dialog.title("Contact Information")
        dialog.geometry("480x320")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(frame, text="Institut für Erdmessung (IfE)", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Leibniz Universität Hannover").pack(anchor="w")
        ttk.Label(frame, text="Schneiderberg 50").pack(anchor="w")
        ttk.Label(frame, text="D-30167 Hannover").pack(anchor="w", pady=(0, 15))
        
        ttk.Label(frame, text="Developer: Dr.-Ing. Johannes Kröger", font=("Helvetica", 10, "bold")).pack(anchor="w")
        
        # Email link
        email_frame = ttk.Frame(frame)
        email_frame.pack(anchor="w", pady=2)
        ttk.Label(email_frame, text="Email: ").pack(side="left")
        email_link = ttk.Label(email_frame, text="kroeger@ife.uni-hannover.de", foreground="#4da6ff", cursor="hand2")
        email_link.pack(side="left")
        email_link.bind("<Button-1>", lambda e: webbrowser.open("mailto:kroeger@ife.uni-hannover.de"))
        
        # Web link
        web_frame = ttk.Frame(frame)
        web_frame.pack(anchor="w", pady=2)
        ttk.Label(web_frame, text="Web: ").pack(side="left")
        web_link = ttk.Label(web_frame, text="www.ife.uni-hannover.de", foreground="#4da6ff", cursor="hand2")
        web_link.pack(side="left")
        web_link.bind("<Button-1>", lambda e: webbrowser.open("https://www.ife.uni-hannover.de"))
        
        def _open_mailing_list():
            # Close first. Two stacked modal dialogs each take the grab, and the
            # inner one releasing it would leave this window unresponsive.
            dialog.destroy()
            self._show_mailing_list()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(20, 0))
        ttk.Button(buttons, text="Mailing list", command=_open_mailing_list,
                   bootstyle="outline-info").pack(side="left", padx=4)
        ttk.Button(buttons, text="Close", command=dialog.destroy,
                   bootstyle="secondary").pack(side="left", padx=4)

        # Size to the content, with 460x300 as the minimum: a fixed size would cut
        # the buttons off on systems with larger fonts or display scaling.
        dialog.update_idletasks()
        dialog.geometry("%dx%d" % (max(460, dialog.winfo_reqwidth()),
                                   max(300, dialog.winfo_reqheight())))

    def _show_about_info(self):
        msg = ("PCC-Explorer\n\n"
               "Software for analyzing the impact of GNSS antenna\n"
               "Phase Center Corrections on geodetic parameters.\n\n"
               + VERSION_TEXT + "\n\n"
               "Citation:\n"
               "Kr\u00f6ger, J., Kersten, T. & Sch\u00f6n, S. PCC-Explorer:\n"
               "An open-source software tool to assess the impact of\n"
               "GNSS antenna phase center corrections on geodetic\n"
               "parameters. GPS Solut 30, 93 (2026).\n"
               "https://doi.org/10.1007/s10291-026-02056-2\n\n"
               "Acknowledgement:\n"
               "The authors thank Mareike Brekenkamp, M.Sc., who developed\n"
               "the software tool's graphical user interface as part of\n"
               "her bachelor's thesis, and Amr Fawzy, M.Sc., who\n"
               "software-engineered the tool from MATLAB to Python,\n"
               "designed and developed the graphical user interface, and\n"
               "implemented several additional features.\n\n"
               "We also thank the Center for Orbit Determination in Europe\n"
               "(CODE) for providing the high-quality GNSS orbits that are\n"
               "primarily used within the software.\n\n"
               "License:\n"
               "This project is licensed under the GNU General Public\n"
               "License v3.0 or later.\n"
               "See LICENSE.txt and LICENSE")
        self._show_scrollable_info("About / Information", msg)

    def _show_scrollable_info(self, title, text):
        """
        Show a long message with the OK button always reachable.

        messagebox.showinfo() grows to fit its content and cannot scroll, and
        this message is 30 lines. With larger system text scaling the OK button
        would land below the bottom of the screen and the dialog could only be
        closed with Escape, so the dialog measures its content and never exceeds
        the screen.
        """
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=15)
        frame.pack(fill="both", expand=True)

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)

        lines = text.splitlines()
        widget = tk.Text(body, wrap="word", relief="flat", padx=10, pady=10,
                         width=max(48, min(72, max(len(l) for l in lines) + 2)),
                         height=min(26, len(lines) + 1))
        scroll = ttk.Scrollbar(body, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        widget.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Follow the app theme, so this does not sit in a dark window as a
        # white rectangle. Guarded: a missing style must not cost the dialog.
        try:
            colors = self.style.colors
            widget.configure(background=colors.bg, foreground=colors.fg,
                             insertbackground=colors.fg)
        except Exception:
            pass

        ttk.Button(frame, text="OK", command=dialog.destroy,
                   bootstyle="secondary").pack(pady=(12, 0))
        dialog.bind("<Escape>", lambda e: dialog.destroy())

        dialog.update_idletasks()
        dialog.geometry("%dx%d" % (
            dialog.winfo_reqwidth(),
            min(dialog.winfo_reqheight(),
                int(dialog.winfo_screenheight() * 0.85))))

    def _show_mailing_list(self):
        """The mailing list dialog, on request, whatever was answered before."""
        if mailing_list is None:
            messagebox.showwarning(
                "IfE software mailing list",
                "This installation is missing mailing_list.py, so the "
                "subscription dialog cannot be shown. Please report it to "
                "the IfE, the Contact button has the address.")
            return
        mailing_list.show_on_request(self, "PCC-Explorer")

    def _launch_tool(self, folder_name, script_name):
        """Launch a sibling PCC tool as a subprocess."""
        import subprocess
        # Resolve path: <project root>/../<folder_name>/<script_name>
        hiwi_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tool_script = os.path.join(hiwi_dir, folder_name, script_name)
        if not os.path.isfile(tool_script):
            messagebox.showerror("Tool Not Found",
                                 f"Cannot find {folder_name}/{script_name}\n\n"
                                 f"Expected at:\n{tool_script}")
            return
        try:
            subprocess.Popen([sys.executable, tool_script],
                             cwd=os.path.dirname(tool_script))
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch {folder_name}:\n{e}")

    def _close_all_figures(self):
        """Close all matplotlib figure windows without closing the main application."""
        import matplotlib.pyplot as plt
        num_figures = len(plt.get_fignums())
        if num_figures == 0:
            messagebox.showinfo("Close Figures", "No figure windows are currently open.")
        else:
            plt.close('all')
            messagebox.showinfo("Close Figures", f"Closed {num_figures} figure window(s).")

    def _on_y_axis_mode_change(self, event=None):
        """Show/hide custom Y-axis min/max fields based on mode selection."""
        mode = self.y_axis_mode_var.get()
        if mode == "Custom":
            self.y_axis_custom_frame.pack(side="left", padx=(5, 0))
        else:
            self.y_axis_custom_frame.pack_forget()
    
    def _on_colorbar_mode_change(self, event=None):
        """Show/hide custom colorbar min/max fields based on mode selection."""
        mode = self.colorbar_mode_var.get()
        if mode == "Custom":
            self.colorbar_custom_frame.pack(side="left", padx=(5, 0))
        else:
            self.colorbar_custom_frame.pack_forget()

    def _create_file_selection_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Selection of the ANTEX-Files", padding=10)
        frame.pack(pady=10, fill="x")
        self.antex_mode = tk.StringVar(value="two_files")
        ttk.Radiobutton(frame, text="Compare two files", variable=self.antex_mode, value="two_files", command=self.update_antex_mode).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(frame, text="Single file (absolute PCC)", variable=self.antex_mode, value="single_file", command=self.update_antex_mode).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(frame, text="Whole folder", variable=self.antex_mode, value="whole_folder", command=self.update_antex_mode).grid(row=2, column=0, columnspan=3, sticky="w")
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)

        self.antex1_label = ttk.Label(frame, text="ANTEX File:")
        self.antex1_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.antex1_path = tk.StringVar()
        self.antex1_entry = ttk.Entry(frame, textvariable=self.antex1_path, width=40)
        self.antex1_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.antex1_entry.bind("<FocusOut>", lambda e: self._on_file1_path_changed())
        self.antex1_entry.bind("<Return>", lambda e: self._on_file1_path_changed())
        self.antex1_browse = ttk.Button(frame, text="Browse...", command=lambda: self.browse_file(self.antex1_path, 1))
        self.antex1_browse.grid(row=4, column=2, padx=5, pady=5)

        self.antex2_label = ttk.Label(frame, text="File 2:")
        self.antex2_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.antex2_path = tk.StringVar()
        self.antex2_entry = ttk.Entry(frame, textvariable=self.antex2_path, width=40)
        self.antex2_entry.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        self.antex2_entry.bind("<FocusOut>", lambda e: self._on_file2_path_changed())
        self.antex2_entry.bind("<Return>", lambda e: self._on_file2_path_changed())
        self.antex2_browse = ttk.Button(frame, text="Browse...", command=lambda: self.browse_file(self.antex2_path, 2))
        self.antex2_browse.grid(row=5, column=2, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)

    def _create_position_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Selection of the user position", padding=10)
        frame.pack(pady=10, fill="x")
        # Kept so line-of-sight mode can relabel it and grey its entries.
        self.position_frame = frame
        self.pos_type = tk.StringVar(value="Ellipsoidal")
        ttk.Radiobutton(frame, text="ECEF Coordinates", variable=self.pos_type, value="ECEF").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frame, text="Ellipsoidal Coordinates", variable=self.pos_type, value="Ellipsoidal").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frame, text="Choose on Map", variable=self.pos_type, value="Map").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(frame, text="Area-based calculation (Global)", variable=self.pos_type, value="Area").grid(row=3, column=0, columnspan=2, sticky="w")

        self.coord1_label = ttk.Label(frame, text="Lat [deg]:")
        self.coord1_label.grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.coord1_var = tk.StringVar(value="52.385") 
        self.coord1_entry = ttk.Entry(frame, textvariable=self.coord1_var, width=15)
        self.coord1_entry.grid(row=4, column=1, sticky="ew")
        self.coord2_label = ttk.Label(frame, text="Lon [deg]:")
        self.coord2_label.grid(row=5, column=0, sticky="e", padx=5, pady=2)
        self.coord2_var = tk.StringVar(value="9.713") 
        self.coord2_entry = ttk.Entry(frame, textvariable=self.coord2_var, width=15)
        self.coord2_entry.grid(row=5, column=1, sticky="ew")
        self.coord3_label = ttk.Label(frame, text="H [m]:")
        self.coord3_label.grid(row=6, column=0, sticky="e", padx=5, pady=2)
        self.coord3_var = tk.StringVar(value="65.0") 
        self.coord3_entry = ttk.Entry(frame, textvariable=self.coord3_var, width=15)
        self.coord3_entry.grid(row=6, column=1, sticky="ew")
        
        ttk.Separator(frame, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky='ew', pady=5)
        area_label = ttk.Label(frame, text="Global and Regional Analysis Settings", font=("Helvetica", 9))
        area_label.grid(row=8, column=0, columnspan=2, sticky="w", pady=(5,2))
        
        ttk.Label(frame, text="Lat Min/Max:").grid(row=9, column=0, sticky="e", padx=5, pady=2)
        lat_frame = ttk.Frame(frame); lat_frame.grid(row=9, column=1, sticky="w")
        self.lat_min_var = tk.StringVar(value="-90"); self.lat_min_entry = ttk.Entry(lat_frame, textvariable=self.lat_min_var, width=5)
        self.lat_min_entry.pack(side="left", padx=(0,2))
        ttk.Label(lat_frame, text="/").pack(side="left")
        self.lat_max_var = tk.StringVar(value="90"); self.lat_max_entry = ttk.Entry(lat_frame, textvariable=self.lat_max_var, width=5)
        self.lat_max_entry.pack(side="left", padx=(2,0))

        ttk.Label(frame, text="Lon Min/Max:").grid(row=10, column=0, sticky="e", padx=5, pady=2)
        lon_frame = ttk.Frame(frame); lon_frame.grid(row=10, column=1, sticky="w")
        self.lon_min_var = tk.StringVar(value="-180"); self.lon_min_entry = ttk.Entry(lon_frame, textvariable=self.lon_min_var, width=5)
        self.lon_min_entry.pack(side="left", padx=(0,2))
        ttk.Label(lon_frame, text="/").pack(side="left")
        self.lon_max_var = tk.StringVar(value="180"); self.lon_max_entry = ttk.Entry(lon_frame, textvariable=self.lon_max_var, width=5)
        self.lon_max_entry.pack(side="left", padx=(2,0))

        ttk.Label(frame, text="Resolution [deg]:").grid(row=11, column=0, sticky="e", padx=5, pady=2)
        self.grid_step_var = tk.StringVar(value="15.0")
        self.grid_step_entry = ttk.Entry(frame, textvariable=self.grid_step_var, width=10)
        self.grid_step_entry.grid(row=11, column=1, sticky="w", padx=2)
        frame.columnconfigure(1, weight=1)

    def _create_time_widgets(self, parent):
        self.time_frame = ttk.LabelFrame(parent, text="Selection of the GPS-observation time", padding=10)
        self.time_frame.pack(pady=10, fill="x")
        ttk.Label(self.time_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="w", pady=2)
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        self.start_date_entry = ttk.Entry(self.time_frame, textvariable=self.start_date_var)
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(self.time_frame, text="Start Time (HH:MM):").grid(row=1, column=0, sticky="w", pady=2)
        self.start_time_var = tk.StringVar(value="00:00")
        self.start_time_entry = ttk.Entry(self.time_frame, textvariable=self.start_time_var)
        self.start_time_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(self.time_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w", pady=2)
        self.end_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        self.end_date_entry = ttk.Entry(self.time_frame, textvariable=self.end_date_var)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(self.time_frame, text="End Time (HH:MM):").grid(row=3, column=0, sticky="w", pady=2)
        self.end_time_var = tk.StringVar(value="23:59")
        self.end_time_entry = ttk.Entry(self.time_frame, textvariable=self.end_time_var)
        self.end_time_entry.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(self.time_frame, text="Sample Rate [sec]:").grid(row=4, column=0, sticky="w", pady=2)
        self.samplerate_var = tk.StringVar(value="300")
        self.samplerate_entry = ttk.Entry(self.time_frame, textvariable=self.samplerate_var)
        self.samplerate_entry.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(self.time_frame, text="Interval [HH:MM] (sub-daily):").grid(row=5, column=0, sticky="w", pady=2)
        self.interval_var = tk.StringVar(value="24:00") 
        self.interval_entry = ttk.Entry(self.time_frame, textvariable=self.interval_var)
        self.interval_entry.grid(row=5, column=1, padx=5, pady=2, sticky="ew")
        # Hint for Time Series multi-day behavior
        hint_label = ttk.Label(self.time_frame, text="\u2139\ufe0f Time Series: Start Time applies to the first day only, "
                               "End Time to the last day only. Intermediate days are processed continuously (00:00\u201323:59).",
                               wraplength=300, font=("Helvetica", 8), foreground="gray")
        hint_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.time_frame.columnconfigure(1, weight=1)
        # Store all time entries for mode switching
        self.time_entries = [self.start_date_entry, self.start_time_entry,
                             self.end_date_entry, self.end_time_entry,
                             self.samplerate_entry, self.interval_entry]

    def _create_model_selection_widgets(self, parent):
        antenna_frame = ttk.LabelFrame(parent, text="Antenna Type", padding=10)
        antenna_frame.pack(pady=10, fill="x")
        
        # File 1 Antenna Selection (Combobox for multi-antenna files like igs20.atx)
        file1_row = ttk.Frame(antenna_frame)
        file1_row.pack(fill="x", pady=2)
        ttk.Label(file1_row, text="File 1:", width=8).pack(side="left", padx=5)
        self.antenna_type_var = tk.StringVar()
        self.antenna_combobox = ttk.Combobox(file1_row, textvariable=self.antenna_type_var, state="disabled", width=35)
        self.antenna_combobox.pack(side="left", padx=5, fill="x", expand=True)
        self.antenna_combobox.bind("<<ComboboxSelected>>", self._on_antenna_selected)
        
        # File 2 Antenna Selection (Independent Combobox - allows ANY antenna selection)
        self.file2_row = ttk.Frame(antenna_frame)
        self.file2_row.pack(fill="x", pady=2)
        ttk.Label(self.file2_row, text="File 2:", width=8).pack(side="left", padx=5)
        self.antenna_type_var_file2 = tk.StringVar()
        self.antenna_combobox_file2 = ttk.Combobox(self.file2_row, textvariable=self.antenna_type_var_file2, state="disabled", width=35)
        self.antenna_combobox_file2.pack(side="left", padx=5, fill="x", expand=True)
        self.antenna_combobox_file2.bind("<<ComboboxSelected>>", self._on_antenna2_selected)
        
        # Match Status Indicator
        match_row = ttk.Frame(antenna_frame)
        match_row.pack(fill="x", pady=(5, 2))
        ttk.Label(match_row, text="Match:", width=8).pack(side="left", padx=5)
        self.antenna_match_label = ttk.Label(match_row, text="--", foreground="gray")
        self.antenna_match_label.pack(side="left", padx=5)

        self.orbit_frame = ttk.LabelFrame(parent, text="Orbit Type", padding=10)
        self.orbit_frame.pack(pady=10, fill="x")
        self.orbit_type_var = tk.StringVar(value="final") 
        self.orbit_rb_broadcast = ttk.Radiobutton(self.orbit_frame, text="Broadcast Ephemeris", variable=self.orbit_type_var, value="broadcast")
        self.orbit_rb_broadcast.pack(anchor="w")
        self.orbit_rb_final = ttk.Radiobutton(self.orbit_frame, text="Final Orbits (Precise)", variable=self.orbit_type_var, value="final")
        self.orbit_rb_final.pack(anchor="w")

        # --- Computation Mode (v1.1: Line-of-Sight) ---
        comp_frame = ttk.LabelFrame(parent, text="Computation Mode", padding=10)
        comp_frame.pack(pady=10, fill="x")
        self.computation_mode_var = tk.StringVar(value="grid")
        # Set only when a RINEX header turned out to carry no APPROX POSITION,
        # which is the one case where LoS mode still needs the position boxes.
        self.los_needs_manual_position = False
        ttk.Radiobutton(comp_frame, text="Grid-based (default)", variable=self.computation_mode_var,
                        value="grid", command=self._on_computation_mode_change).pack(anchor="w")
        ttk.Label(comp_frame, text="   Simulative analysis using full-sky grid and orbit data",
                  foreground="gray", font=("Helvetica", 8)).pack(anchor="w")
        ttk.Radiobutton(comp_frame, text="Line-of-Sight (real observations)", variable=self.computation_mode_var,
                        value="los", command=self._on_computation_mode_change).pack(anchor="w")
        ttk.Label(comp_frame, text="   Analysis using actual satellite observations (CSV/ATT/KIN)",
                  foreground="gray", font=("Helvetica", 8)).pack(anchor="w")

        # LoS-specific widgets (initially hidden)
        self.los_widgets_frame = ttk.Frame(comp_frame)

        # Observation file
        obs_row = ttk.Frame(self.los_widgets_frame)
        obs_row.pack(fill="x", pady=(5, 2))
        # Name the format in the label. A RINEX file picked here is not
        # imported, and without the extension it is not obvious why: the RINEX
        # row is the one below.
        ttk.Label(obs_row, text="Observation File (.CSV):").pack(side="left", padx=(0, 5))
        self.los_obs_path = tk.StringVar()
        self.los_obs_entry = ttk.Entry(obs_row, textvariable=self.los_obs_path, width=30)
        self.los_obs_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(obs_row, text="Browse...", command=self._browse_los_obs_file).pack(side="left")

        # --- Attitude File (.ATT) — optional ---
        att_row = ttk.Frame(self.los_widgets_frame)
        att_row.pack(fill="x", pady=2)
        ttk.Label(att_row, text="Attitude File (.ATT):").pack(side="left", padx=(0, 5))
        self.att_file_path = tk.StringVar()
        self.att_file_path.trace_add("write", self._on_att_file_changed)
        self.att_file_entry = ttk.Entry(att_row, textvariable=self.att_file_path, width=30)
        self.att_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(att_row, text="Browse...", command=self._browse_att_file).pack(side="left")

        # --- Trajectory File (.KIN) — optional ---
        kin_row = ttk.Frame(self.los_widgets_frame)
        kin_row.pack(fill="x", pady=2)
        ttk.Label(kin_row, text="Trajectory File (.KIN):").pack(side="left", padx=(0, 5))
        self.kin_file_path = tk.StringVar()
        self.kin_file_entry = ttk.Entry(kin_row, textvariable=self.kin_file_path, width=30)
        self.kin_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(kin_row, text="Browse...", command=self._browse_kin_file).pack(side="left")

        # --- Euler Angles (manual entry, disabled when ATT loaded) ---
        euler_frame = ttk.Frame(self.los_widgets_frame)
        euler_frame.pack(fill="x", pady=2)
        ttk.Label(euler_frame, text="Euler Angles:").pack(side="left", padx=(0, 5))

        self.heading_deg_var = tk.StringVar(value="0.0")
        self.pitch_deg_var = tk.StringVar(value="0.0")
        self.roll_deg_var = tk.StringVar(value="0.0")

        ttk.Label(euler_frame, text="Yaw [°]:").pack(side="left", padx=(5, 2))
        self.yaw_entry = ttk.Entry(euler_frame, textvariable=self.heading_deg_var, width=7)
        self.yaw_entry.pack(side="left")

        ttk.Label(euler_frame, text="Pitch [°]:").pack(side="left", padx=(5, 2))
        self.pitch_entry = ttk.Entry(euler_frame, textvariable=self.pitch_deg_var, width=7)
        self.pitch_entry.pack(side="left")

        ttk.Label(euler_frame, text="Roll [°]:").pack(side="left", padx=(5, 2))
        self.roll_entry = ttk.Entry(euler_frame, textvariable=self.roll_deg_var, width=7)
        self.roll_entry.pack(side="left")

        CreateToolTip(euler_frame, "Manual Euler angles (degrees). Disabled when ATT file is loaded.\nYaw = heading (clockwise from North). Pitch/Roll default to 0°.")

        # --- RINEX Observation File (optional — converts to CSV via RINEX-Masker bridge) ---
        rinex_sep = ttk.Separator(self.los_widgets_frame, orient="horizontal")
        rinex_sep.pack(fill="x", pady=(8, 4))
        ttk.Label(self.los_widgets_frame,
                  text="Or import from RINEX (requires RINEX-Masker):",
                  foreground="gray", font=("Helvetica", 8)).pack(anchor="w", pady=(0, 2))

        rinex_row = ttk.Frame(self.los_widgets_frame)
        rinex_row.pack(fill="x", pady=2)
        ttk.Label(rinex_row, text="RINEX Obs File:").pack(side="left", padx=(0, 5))
        self.rinex_obs_path = tk.StringVar()
        self.rinex_obs_entry = ttk.Entry(rinex_row, textvariable=self.rinex_obs_path, width=25)
        self.rinex_obs_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(rinex_row, text="Browse...", command=self._browse_rinex_obs_file).pack(side="left", padx=(0, 3))
        self.rinex_convert_btn = ttk.Button(rinex_row, text="Convert → CSV",
                                            command=self._convert_rinex_to_csv, bootstyle="info-outline")
        self.rinex_convert_btn.pack(side="left")

        # Where RINEX-Masker lives. Auto-detection only knows the developer's
        # source-tree layout, so the user must be able to point at their own copy.
        masker_row = ttk.Frame(self.los_widgets_frame)
        masker_row.pack(fill="x", pady=(2, 2))
        ttk.Button(masker_row, text="RINEX-Masker folder...",
                   command=self._browse_masker_folder,
                   bootstyle="secondary-outline").pack(side="left", padx=(0, 5))
        self.masker_folder_label = ttk.Label(masker_row, text="(auto-detect)",
                                             foreground="gray", font=("Helvetica", 8))
        self.masker_folder_label.pack(side="left", fill="x", expand=True)
        self._refresh_masker_folder_label()

        # --- Resample interval (optional — for 10 Hz data processed at 1 Hz, etc.) ---
        resample_row = ttk.Frame(self.los_widgets_frame)
        resample_row.pack(fill="x", pady=(5, 2))
        ttk.Label(resample_row, text="Resample interval [s]:").pack(side="left", padx=(0, 5))
        self.los_resample_var = tk.StringVar(value="")
        self.los_resample_entry = ttk.Entry(resample_row, textvariable=self.los_resample_var, width=8)
        self.los_resample_entry.pack(side="left", padx=(0, 5))
        ttk.Label(resample_row, text="(empty = use all epochs)",
                  foreground="gray", font=("Helvetica", 8)).pack(side="left")
        CreateToolTip(resample_row, "Optional: thin high-rate data to a lower rate.\nE.g. enter '1' to process 10 Hz data at 1 Hz.")

        # Info label
        ttk.Label(self.los_widgets_frame,
                  text="CSV format: epoch_utc,prn,elevation_deg,azimuth_deg[,heading_deg]",
                  foreground="gray", font=("Helvetica", 8)).pack(anchor="w", pady=(2, 0))
        # Don't pack los_widgets_frame yet — it's shown only when LoS is selected

        freq_frame = ttk.LabelFrame(parent, text="Frequency & Signal", padding=10)
        freq_frame.pack(pady=10, fill="x")
        # Header with Switch
        header_frame = ttk.Frame(freq_frame)
        header_frame.pack(fill="x", pady=(0, 5))
        
        self.multifreq_var = tk.BooleanVar(value=False)
        self.multifreq_check = ttk.Checkbutton(header_frame, text="Switch to Multi-frequency Mode", variable=self.multifreq_var, command=self.update_frequency_mode, bootstyle="round-toggle")
        self.multifreq_check.pack(side="left")
        CreateToolTip(self.multifreq_check, "Toggle between Single Signal (Radio buttons) and Multi-signal (Checkboxes) selection modes.")
        
        self.mode_label = ttk.Label(header_frame, text="(Single Select)", foreground="gray", font=("Helvetica", 9, "italic"))
        self.mode_label.pack(side="left", padx=10)

        # Scrollable Frame for Signals (Replacing Listbox)
        self.signal_scroll_frame = ScrollableFrame(freq_frame, height=120)
        self.signal_scroll_frame.pack(fill="x", expand=True)
        self.signal_container = self.signal_scroll_frame.scrollable_frame
        
        # Explanatory text for filtering behavior and signal types
        self.signal_info_label = ttk.Label(freq_frame, text="Legend: G=GPS, R=GLONASS, E=Galileo, C=BDS. Showing available signals.", foreground="gray", font=("Helvetica", 8), wraplength=500)
        self.signal_info_label.pack(anchor="w", pady=(5,0))

        tropo_frame = ttk.LabelFrame(parent, text="Troposphere Mapping Function", padding=10)
        tropo_frame.pack(pady=10, fill="x")
        self.tropo_mapping_var = tk.StringVar()
        tropo_options = ["1/sin(Elevation)", "GMF", "VMF", "None"]
        self.tropo_combobox = ttk.Combobox(tropo_frame, textvariable=self.tropo_mapping_var, values=tropo_options, state="readonly", width=25)
        self.tropo_combobox.pack(padx=5, pady=(5,0), anchor="w")
        self.tropo_combobox.set("1/sin(Elevation)")
        self.gradient_var = tk.BooleanVar(value=False)
        self.gradient_check = ttk.Checkbutton(tropo_frame, text="Estimate Troposphere Gradients (Simple Model)", variable=self.gradient_var)
        self.gradient_check.pack(anchor="w", pady=(5,0))

        weight_frame = ttk.LabelFrame(parent, text="Observation Weighting Model", padding=10)
        weight_frame.pack(pady=10, fill="x")
        self.weight_var = tk.StringVar()
        weight_options = ["sin", "sin2", "unit"]
        self.weight_combobox = ttk.Combobox(weight_frame, textvariable=self.weight_var, values=weight_options, state="readonly", width=25)
        self.weight_combobox.pack(padx=5, pady=5)
        self.weight_combobox.set("sin") 

    def _create_mask_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Obstruction Masks", padding=10)
        frame.pack(pady=10, fill="x")
        el_frame = ttk.Frame(frame)
        el_frame.pack(fill="x", pady=(0, 10))
        self.elevation_mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(el_frame, text="Elevation mask <", variable=self.elevation_mask_var, command=self.update_mask_widgets).pack(side="left", padx=5)
        self.elevation_angle_var = tk.StringVar(value="5.0")
        self.elevation_angle_entry = ttk.Entry(el_frame, textvariable=self.elevation_angle_var, width=8)
        self.elevation_angle_entry.pack(side="left")
        ttk.Label(el_frame, text="[deg]").pack(side="left", padx=(2, 0))
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=5)
        az_frame = ttk.Frame(frame)
        az_frame.pack(fill="x")
        self.azimuth_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(az_frame, text="Azimuth mask(s) active", variable=self.azimuth_mask_var, command=self.update_mask_widgets).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5))
        input_frame = ttk.Frame(az_frame)
        input_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        ttk.Label(input_frame, text="From:").pack(side="left", padx=(5, 2))
        self.azimuth_from_var = tk.StringVar(value="0")
        self.azimuth_from_entry = ttk.Entry(input_frame, textvariable=self.azimuth_from_var, width=5, state="disabled")
        self.azimuth_from_entry.pack(side="left")
        ttk.Label(input_frame, text="To:").pack(side="left", padx=(10, 2))
        self.azimuth_to_var = tk.StringVar(value="90")
        self.azimuth_to_entry = ttk.Entry(input_frame, textvariable=self.azimuth_to_var, width=5, state="disabled")
        self.azimuth_to_entry.pack(side="left")
        ttk.Label(input_frame, text="[deg], Elev <").pack(side="left", padx=(10, 2))
        self.azimuth_el_var = tk.StringVar(value="30")
        self.azimuth_el_entry = ttk.Entry(input_frame, textvariable=self.azimuth_el_var, width=5, state="disabled")
        self.azimuth_el_entry.pack(side="left")
        ttk.Label(input_frame, text="[deg]").pack(side="left", padx=(2, 0))
        button_frame = ttk.Frame(az_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=5)
        self.add_az_mask_button = ttk.Button(button_frame, text="Add Mask", command=self._add_azimuth_mask, state="disabled")
        self.add_az_mask_button.pack(side="left", padx=5)
        self.remove_az_mask_button = ttk.Button(button_frame, text="Remove Selected", command=self._remove_azimuth_mask, state="disabled", bootstyle="warning")
        self.remove_az_mask_button.pack(side="left", padx=5)
        list_frame = ttk.Frame(az_frame)
        list_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=(5,0))
        self.azimuth_mask_listbox = tk.Listbox(list_frame, height=4, exportselection=False, state="disabled")
        az_list_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.azimuth_mask_listbox.yview)
        self.azimuth_mask_listbox.configure(yscrollcommand=az_list_scrollbar.set)
        az_list_scrollbar.pack(side="right", fill="y")
        self.azimuth_mask_listbox.pack(side="left", fill="both", expand=True)
        az_frame.rowconfigure(3, weight=1)
        az_frame.columnconfigure(0, weight=1)

        # --- Horizon profile imported from RINEX-Masker -------------------
        # An azimuth-dependent horizon: the real skyline around the station,
        # rather than a single cut-off angle.
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=5)
        obs_frame = ttk.Frame(frame)
        obs_frame.pack(fill="x")
        self.obstruction_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(obs_frame, text="Obstruction mask from RINEX-Masker",
                        variable=self.obstruction_mask_var,
                        command=self.update_mask_widgets).pack(anchor="w", padx=5, pady=(0, 5))
        obs_row = ttk.Frame(obs_frame)
        obs_row.pack(fill="x", padx=5)
        self.obstruction_mask_path = tk.StringVar()
        self.obstruction_mask_entry = ttk.Entry(obs_row, textvariable=self.obstruction_mask_path,
                                                state="disabled")
        self.obstruction_mask_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.obstruction_mask_button = ttk.Button(obs_row, text="Browse...",
                                                  command=self._browse_obstruction_mask,
                                                  state="disabled")
        self.obstruction_mask_button.pack(side="left")
        self.obstruction_mask_info = ttk.Label(obs_frame, text="", foreground="gray",
                                               font=("Helvetica", 8))
        self.obstruction_mask_info.pack(anchor="w", padx=5, pady=(3, 0))
        CreateToolTip(obs_row,
                      "Horizon profile exported by RINEX-Masker "
                      "(pcc-mask-v1 JSON or the plain-text mask).\n"
                      "Directions below the skyline are excluded from the analysis.\n"
                      "Tip: set the elevation mask to 0 to see the mask's own effect - "
                      "a cut-off angle hides most of it.")

    def _browse_obstruction_mask(self):
        """Select a horizon mask exported by RINEX-Masker and report what it contains."""
        filepath = filedialog.askopenfilename(
            title="Select an obstruction mask from RINEX-Masker",
            filetypes=[
                ("Mask files", "*.json *.txt"),
                ("pcc-mask-v1 JSON", "*.json"),
                ("Text mask", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if not filepath:
            return
        self.obstruction_mask_path.set(filepath)

        # Load it straight away: a mask that fails at run time after a long
        # analysis is far more annoying than one rejected here.
        try:
            try:
                from src.obstruction_mask import load_obstruction_mask
            except ImportError:
                from obstruction_mask import load_obstruction_mask
            mask = load_obstruction_mask(filepath)
            self.obstruction_mask_info.config(text=f"Loaded: {mask.summary()}")
        except Exception as e:
            self.obstruction_mask_info.config(text="Could not read this mask.")
            messagebox.showerror(
                "Invalid Obstruction Mask",
                f"Could not read the mask file:\n\n{e}\n\n"
                "Expected a 'pcc-mask-v1' JSON file or a RINEX-Masker text mask."
            )

    def _selected_signal_codes(self):
        """
        The signals currently ticked, as processing codes ('G01', 'IF_G01_G02').

        Extracted from the configuration builder so the Input
        Validation panel can ask the same question the run will ask, instead of
        a second copy of this mapping drifting away from it. Returns [] when the
        widgets do not exist yet (this runs once during __init__) or when
        nothing is selected - callers decide what an empty selection means.
        """
        raw_signals = []
        try:
            if self.multifreq_var.get():
                # Multi-Mode: Check tracking dictionary
                for sig, var in self.signal_vars.items():
                    if var.get():
                        raw_signals.append(sig)
            else:
                # Single-Mode: Check StringVar
                sel = self.selected_signal_var.get()
                if sel: raw_signals.append(sel)
        except (AttributeError, tk.TclError):
            return []

        parsed_signals = []
        for sig_str in raw_signals:
            if "GPS IF-LC L1/L2" in sig_str: parsed_signals.append("IF_G01_G02")
            elif "GPS IF-LC L1/L5" in sig_str: parsed_signals.append("IF_G01_G05")

            elif "GLONASS IF-LC R1/R2" in sig_str: parsed_signals.append("IF_R01_R02")
            elif "GLONASS IF-LC G1/G2" in sig_str: parsed_signals.append("IF_R01_R02") # Alias
            elif "GLONASS IF-LC G1/G3" in sig_str: parsed_signals.append("IF_R01_R03")

            elif "Galileo IF-LC E1/E5a" in sig_str: parsed_signals.append("IF_E01_E05")
            elif "Galileo IF-LC E1/E5b" in sig_str: parsed_signals.append("IF_E01_E07")
            elif "Galileo IF-LC E1/E5" in sig_str: parsed_signals.append("IF_E01_E08")
            elif "Galileo IF-LC E1/E6" in sig_str: parsed_signals.append("IF_E01_E06")

            elif "Beidou IF-LC B1/B2" in sig_str: parsed_signals.append("IF_C02_C07") # B1I/B2I
            elif "Beidou IF-LC B1/B3" in sig_str: parsed_signals.append("IF_C02_C06") # B1I/B3I

            else:
                parts = sig_str.split()
                if len(parts) >= 2:
                    # Old format: "GPS G01"
                    potential_code = parts[1]
                    if re.match(r'^[GRECJSC]\d{2}$', potential_code): parsed_signals.append(potential_code)
                else:
                    # Direct code format: "G01"
                    potential_code = parts[0]
                    if re.match(r'^[GRECJSC]\d{2}$', potential_code): parsed_signals.append(potential_code)
                    elif "IF-LC" in sig_str:
                         # Handles manual entry fallback
                         pass

        # Additional Validation for Empty Signals
        if not parsed_signals and raw_signals:
            # Parsing failed on a non-empty selection: fall back to the raw
            # labels (dangerous but better than empty).
            valid_raw = [s for s in raw_signals if s.strip()]
            if valid_raw: parsed_signals = valid_raw

        return parsed_signals

    def _update_insights(self, *args):
        try:
            orbit_type = self.orbit_type_var.get()
            date_str = self.start_date_var.get()
            selected_date = datetime.strptime(date_str, "%Y-%m-%d")
            message = ""; style = "default"
            
            # Check sampling rate for unusual values
            try:
                sample_rate = int(self.samplerate_var.get())
                if sample_rate < 30:
                    message = f"⚠️ WARNING: Very fast sampling rate ({sample_rate}s) may cause long processing times.\n\n"
                    style = "warning"
                elif sample_rate > 900:
                    message = f"⚠️ NOTE: Large sampling rate ({sample_rate}s = {sample_rate//60}min) may miss satellite passes.\n\n"
                    style = "warning"
            except (ValueError, AttributeError):
                pass
            
            if orbit_type == 'final':
                cutoff_days = ConfigValidator.FINAL_ORBIT_DELAY_DAYS
                time_delta = datetime.now() - selected_date
                if time_delta.days < cutoff_days:
                    cutoff_date = (datetime.now() - timedelta(days=cutoff_days)).strftime('%Y-%m-%d')
                    # Not a dead end: the run falls back to broadcast orbits
                    # automatically, so the note says that rather than asking
                    # the user to pick another date.
                    message += (f"ℹ️ NOTE: Precise orbits appear about {cutoff_days} days after the day, "
                                f"so they may not exist yet for this date (older than {cutoff_date} is safe).\n\n")
                    # ... but only for a GPS-only run. The broadcast
                    # reader skips every non-GPS record, so promising the
                    # substitution here contradicted the refusal the pipeline
                    # then raised ("this run needs E") - and the panel had said
                    # "All inputs look valid" a moment earlier.
                    selected = self._selected_signal_codes()
                    if not selected or ConfigValidator.broadcast_can_serve(
                            {'signals': selected}):
                        message += (f"Broadcast ephemeris will be used automatically if the precise product "
                                    f"is missing, satellite geometry is unaffected.")
                        if style != "warning": style = "info"
                    else:
                        message += (f"Broadcast ephemeris cannot stand in here: it supports GPS only, and "
                                    f"this run needs another constellation. Pick a date older than "
                                    f"{cutoff_date}, or a GPS signal.")
                        style = "warning"
                else:
                    message += f"✅ Final Orbits selected. Date is suitable for precise data."
                    if style != "warning": style = "success"
            elif orbit_type == 'broadcast':
                cutoff_days = ConfigValidator.BROADCAST_DELAY_DAYS
                time_delta = datetime.now() - selected_date
                if time_delta.days < cutoff_days:
                    message += (f"ℹ️ INFO: Broadcast files for 'today' or 'yesterday' might not be published yet.")
                    if style != "warning": style = "info"
                else:
                    message += "✅ Broadcast Ephemeris is suitable for recent and historical data."
                    if style != "warning": style = "success"
            self.insights_text.set(message)
            self.insights_label.config(bootstyle=style)
        except Exception:
            self.insights_text.set("Enter a valid date in YYYY-MM-DD format.")
            self.insights_label.config(bootstyle="secondary")

    def _validate_inputs(self, *args):
        """Real-time validation of user inputs with feedback."""
        messages = []
        style = "success"
        
        # --- ANTEX Files ---
        if self.antex_mode.get() in ('two_files', 'Datei'):
            if not self.antex1_path.get().strip():
                messages.append("ℹ️ Select first ANTEX file.")
                if style == "success": style = "info"
            if not self.antex2_path.get().strip():
                messages.append("ℹ️ Select second ANTEX file.")
                if style == "success": style = "info"
        
        # --- Sampling Rate ---
        try:
            sample_rate = self.samplerate_var.get().strip()
            if not sample_rate:
                messages.append("ℹ️ Enter a sampling rate.")
                if style == "success": style = "info"
            else:
                rate = int(sample_rate)
                if rate <= 0:
                    messages.append("❌ Sampling rate must be positive.")
                    style = "danger"
                elif rate < 30:
                    messages.append(f"⚠️ Fast sampling ({rate}s) - long processing.")
                    if style not in ("danger",): style = "warning"
                elif rate > 900:
                    messages.append(f"⚠️ Large sampling ({rate}s) may miss satellites.")
                    if style not in ("danger",): style = "warning"
        except ValueError:
            if self.samplerate_var.get().strip():
                messages.append("❌ Sampling rate must be a number.")
                style = "danger"
        
        # --- Dates ---
        for name, var in [("Start date", self.start_date_var), ("End date", self.end_date_var)]:
            val = var.get().strip()
            if val:
                try:
                    datetime.strptime(val, "%Y-%m-%d")
                except ValueError:
                    messages.append(f"❌ {name}: use YYYY-MM-DD format.")
                    style = "danger"
        
        # Check end >= start
        try:
            start_dt = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            if end_dt < start_dt:
                messages.append("❌ End date before start date.")
                style = "danger"
        except ValueError:
            pass
        
        # --- Times ---
        for name, var in [("Start time", self.start_time_var), ("End time", self.end_time_var)]:
            val = var.get().strip()
            if val:
                try:
                    parts = val.split(":")
                    if len(parts) != 2:
                        raise ValueError()
                    h, m = int(parts[0]), int(parts[1])
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        messages.append(f"⚠️ {name}: check HH:MM range.")
                        if style not in ("danger",): style = "warning"
                except ValueError:
                    messages.append(f"❌ {name}: use HH:MM format.")
                    style = "danger"
        
        # --- Interval ---
        try:
            interval_str = self.interval_var.get().strip()
            if interval_str:
                parts = interval_str.split(":")
                if len(parts) != 2:
                    raise ValueError()
                h, m = int(parts[0]), int(parts[1])
                if h < 0 or m < 0 or m > 59:
                    messages.append("❌ Interval: invalid HH:MM.")
                    style = "danger"
                elif h == 0 and m == 0:
                    messages.append("❌ Interval cannot be 0:00.")
                    style = "danger"
        except ValueError:
            if self.interval_var.get().strip():
                messages.append("❌ Interval: use HH:MM format.")
                style = "danger"
        
        # --- Elevation Mask ---
        if self.elevation_mask_var.get():
            try:
                elev = float(self.elevation_angle_var.get())
                if elev < 0 or elev > 90:
                    messages.append("❌ Elevation mask: 0-90°.")
                    style = "danger"
            except ValueError:
                messages.append("❌ Elevation mask: enter a number.")
                style = "danger"
        
        # --- Coordinates ---
        pos_mode = self.pos_type.get()
        if pos_mode in ('Ellipsoidal', 'Map'):
            try:
                lat_str = self.coord1_var.get().strip()
                lon_str = self.coord2_var.get().strip()
                height_str = self.coord3_var.get().strip() if hasattr(self, 'coord3_var') else ''
                if not lat_str or not lon_str:
                    messages.append("❌ Latitude and Longitude are required.")
                    style = "danger"
                elif not height_str:
                    messages.append("❌ Height is required.")
                    style = "danger"
                else:
                    lat = float(lat_str)
                    lon = float(lon_str)
                    if not (-90 <= lat <= 90):
                        messages.append("❌ Latitude: -90° to 90°.")
                        style = "danger"
                    if not (-180 <= lon <= 180):
                        messages.append("❌ Longitude: -180° to 180°.")
                        style = "danger"
            except ValueError:
                messages.append("❌ Coordinates must be numbers.")
                style = "danger"
        elif pos_mode == 'Area':
             try:
                 lat_min = float(self.lat_min_var.get()); lat_max = float(self.lat_max_var.get())
                 lon_min = float(self.lon_min_var.get()); lon_max = float(self.lon_max_var.get())
                 step = float(self.grid_step_var.get())
                 
                 if step <= 0: messages.append("❌ Step must be > 0."); style = "danger"
                 elif lat_min >= lat_max: messages.append("❌ Lat Min must be < Max."); style = "danger"
                 else:
                     lat_range = lat_max - lat_min
                     lon_range = lon_max - lon_min
                     # Check divisibility with tolerance
                     if abs((lat_range / step) - round(lat_range / step)) > 1e-4:
                         messages.append(f"⚠️ Lat range ({lat_range:.1f}°) not divisible by {step}°.")
                         messages.append(f"   → Grid will stop at last point within range.")
                         if style != "danger": style = "warning"
                     if abs((lon_range / step) - round(lon_range / step)) > 1e-4:
                         messages.append(f"⚠️ Lon range ({lon_range:.1f}°) not divisible by {step}°.")
                         messages.append(f"   → Grid will stop at last point within range.")
                         if style != "danger": style = "warning"
             except ValueError:
                 messages.append("❌ Area settings must be numbers.")
                 style = "danger"
        elif pos_mode == 'ECEF':
            for var, name in [(self.coord1_var, 'X'), (self.coord2_var, 'Y'), (self.coord3_var, 'Z')]:
                try:
                    val = float(var.get()) if var.get().strip() else None
                    if val is not None and abs(val) > 10000000:
                        messages.append(f"⚠️ {name}: check magnitude.")
                        if style not in ("danger",): style = "warning"
                except ValueError:
                    pass
        
        # --- Signals ---
        if self.multifreq_var.get():
             has_signal_selection = any(v.get() for v in self.signal_vars.values())
        else:
             has_signal_selection = bool(self.selected_signal_var.get())
        
        if not has_signal_selection and self.current_signals:
             messages.append("ℹ️ Select at least one signal.")
             if style == "success": style = "info"
        
        # Set the message
        if messages:
            self.validation_text.set("\n".join(messages[:4]))  # Limit to 4 messages
            self.validation_label.config(bootstyle=style)
        else:
            self.validation_text.set("✅ All inputs look valid.")
            self.validation_label.config(bootstyle="success")

    def update_mask_widgets(self):
        state = "normal" if self.elevation_mask_var.get() else "disabled"
        self.elevation_angle_entry.config(state=state)
        az_state = "normal" if self.azimuth_mask_var.get() else "disabled"
        self.azimuth_from_entry.config(state=az_state)
        self.azimuth_to_entry.config(state=az_state)
        self.azimuth_el_entry.config(state=az_state)
        self.add_az_mask_button.config(state=az_state)
        self.remove_az_mask_button.config(state=az_state)
        self.azimuth_mask_listbox.config(state=az_state)
        obs_state = "normal" if self.obstruction_mask_var.get() else "disabled"
        self.obstruction_mask_entry.config(state=obs_state)
        self.obstruction_mask_button.config(state=obs_state)

    def _current_station_xyz(self):
        """
        The user position as ECEF [X, Y, Z], or None if the boxes are unusable.

        Only a FALLBACK. In line-of-sight mode the angles come from the
        observation file, and a RINEX conversion takes the position from the
        file's own APPROX POSITION XYZ; this is offered to rinex_bridge only for
        the case where the header carries none.
        """
        mode = self.pos_type.get()
        if mode == 'Area':
            return None
        try:
            c1 = float(self.coord1_var.get())
            c2 = float(self.coord2_var.get())
            c3 = float(self.coord3_var.get())
        except (ValueError, AttributeError):
            return None
        if mode == 'ECEF':
            return [c1, c2, c3]
        return list(ell_to_ecef(np.deg2rad(c1), np.deg2rad(c2), c3))

    def _set_position_fields_enabled(self, enabled: bool, reason: str = ""):
        """
        Grey out the user-position boxes, or bring them back.

        In line-of-sight mode these boxes are read by nothing, because the
        observation file already carries azimuth and elevation per epoch.
        Leaving them editable suggests they are being used, so they are
        disabled and the frame says why.
        ⚠ They come BACK if a RINEX header turns out to have no APPROX POSITION,
        which is the one case where the user has to supply a position.
        """
        state = "normal" if enabled else "disabled"
        for entry in (self.coord1_entry, self.coord2_entry, self.coord3_entry):
            entry.config(state=state)
        self.position_frame.config(
            text="Selection of the user position" + (f" ({reason})" if reason else ""))

    def _on_computation_mode_change(self):
        """Show/hide LoS-specific widgets and disable/enable grid-only fields."""
        is_los = self.computation_mode_var.get() == 'los'
        # The position boxes are unused in LoS mode, unless a header-less RINEX
        # has already asked the user for one.
        if is_los and not getattr(self, 'los_needs_manual_position', False):
            self._set_position_fields_enabled(
                False, "taken from the observation file in LoS mode")
        else:
            self._set_position_fields_enabled(True)
            if not is_los:
                self.update_position_mode()
        if is_los:
            self.los_widgets_frame.pack(fill="x", pady=(5, 0))
            # Disable grid-only widgets (time period, orbit type)
            for entry in self.time_entries:
                entry.config(state="disabled")
            self.orbit_rb_broadcast.config(state="disabled")
            self.orbit_rb_final.config(state="disabled")
            self.time_frame.config(text="Selection of the GPS-observation time (not used in LoS mode)")
            self.orbit_frame.config(text="Orbit Type (not used in LoS mode)")
        else:
            self.los_widgets_frame.pack_forget()
            # Re-enable grid-only widgets
            for entry in self.time_entries:
                entry.config(state="normal")
            self.orbit_rb_broadcast.config(state="normal")
            self.orbit_rb_final.config(state="normal")
            self.time_frame.config(text="Selection of the GPS-observation time")
            self.orbit_frame.config(text="Orbit Type")

    def _browse_los_obs_file(self):
        """Open file dialog for satellite observation CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select Satellite Observation File (.CSV)",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.los_obs_path.set(filepath)

    def _browse_rinex_obs_file(self):
        """Open file dialog for RINEX observation file."""
        # A different file may well have a header position, so stop treating
        # the position boxes as required until this one proves otherwise.
        self.los_needs_manual_position = False
        filepath = filedialog.askopenfilename(
            title="Select RINEX Observation File",
            filetypes=[
                ("RINEX files", "*.RNX *.rnx *.obs *.OBS"),
                ("RINEX v2", "*.*O *.??o"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.rinex_obs_path.set(filepath)

    def _draw_pending_plots(self, config):
        """
        Draw figures the analysis queued while it ran in the background.

        The analysis runs in a worker thread. Matplotlib figures must not be
        created there: the figure manager ends up owned by a thread that dies,
        and the next plt.show() on the main thread raises "main thread is not in
        main loop", which crashes the application.
        processing_core therefore queues the request instead of drawing it, and
        this method — called from the completion callback, on the main thread —
        does the drawing.
        """
        if not isinstance(config, dict):
            return
        pending = config.pop('_pending_plots', None)
        if not pending:
            return
        for request in pending:
            try:
                if request.get('kind') == 'trajectory_map':
                    try:
                        from src.plotting import plot_trajectory_map
                    except ImportError:
                        from plotting import plot_trajectory_map
                    plot_trajectory_map(
                        request['kin_data'], request['epoch_results'],
                        request['param_names'],
                        antenna_name=request.get('antenna_name', '')
                    )
                else:
                    print(f"  Warning: unknown queued plot '{request.get('kind')}'.")
            except Exception as e:
                # A failed extra figure must never sink the results the user
                # actually asked for.
                print(f"  Warning: could not draw the queued "
                      f"{request.get('kind', 'figure')}: {e}")

    def _import_rinex_bridge(self):
        """
        Import the bridge module, absolute first.

        Same reason as in _convert_rinex_to_csv: a relative import has no parent
        package once PyInstaller runs main_gui as __main__.
        """
        try:
            from src import rinex_bridge
        except ImportError:
            import rinex_bridge
        return rinex_bridge

    def _refresh_masker_folder_label(self):
        """Show which RINEX-Masker folder will be used, if one has been chosen."""
        saved = None
        try:
            saved = self._import_rinex_bridge().load_saved_masker_path()
        except Exception:
            pass  # Never let a label block the GUI from building
        self.masker_folder_label.config(text=saved if saved else "(auto-detect)")

    def _browse_masker_folder(self):
        """Let the user point at their RINEX-Masker installation."""
        folder = filedialog.askdirectory(
            title="Select the RINEX-Masker folder (the one holding RINEX-Masker.exe)"
        )
        if not folder:
            return
        try:
            bridge = self._import_rinex_bridge()
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to import RINEX bridge:\n{e}")
            return

        if bridge.set_masker_path(folder):
            self._refresh_masker_folder_label()
            # Show where the modules were actually found. Picking the
            # installation folder resolves to its 'core' sub-folder, and the user
            # should see that rather than wonder whether the choice took effect.
            resolved = bridge.get_masker_path() or bridge.resolve_masker_module_dir(folder) or folder
            extra = "" if os.path.normpath(resolved) == os.path.normpath(folder) else \
                    f"\n(modules found in: {resolved})"
            messagebox.showinfo(
                "RINEX-Masker Folder Set",
                f"RINEX-Masker will be loaded from:\n\n{folder}{extra}\n\n"
                "This choice is remembered for future sessions."
            )
        else:
            messagebox.showerror(
                "Not a RINEX-Masker Folder",
                f"RINEX-Masker was not found in:\n\n{folder}\n\n"
                + bridge.describe_masker_folder_requirement()
            )

    def _convert_rinex_to_csv(self):
        """Convert a RINEX observation file to LoS CSV using the RINEX-Masker bridge."""
        rinex_path = self.rinex_obs_path.get().strip()
        if not rinex_path:
            messagebox.showwarning("No RINEX File", "Please select a RINEX observation file first.")
            return
        if not os.path.exists(rinex_path):
            messagebox.showerror("File Not Found", f"RINEX file not found:\n{rinex_path}")
            return

        try:
            # Absolute import, matching every other import in this module. A relative
            # import fails in the frozen build: PyInstaller runs main_gui as __main__,
            # which has no parent package ("attempted relative import with no known
            # parent package"). It only ever worked when run from source.
            try:
                from src.rinex_bridge import rinex_to_los_observations, check_rinex_masker_available
            except ImportError:
                from rinex_bridge import rinex_to_los_observations, check_rinex_masker_available
            if not check_rinex_masker_available():
                try:
                    from src.rinex_bridge import describe_masker_folder_requirement
                except ImportError:
                    from rinex_bridge import describe_masker_folder_requirement
                messagebox.showerror(
                    "RINEX-Masker Not Found",
                    "Could not find RINEX-Masker.\n\n"
                    + describe_masker_folder_requirement() +
                    "\n\nUse 'RINEX-Masker folder...' below to select it. "
                    "The folder is remembered for future sessions."
                )
                return
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to import RINEX bridge:\n{e}")
            return

        # The conversion runs on the shared BackgroundTask with a progress
        # dialog, like every other long operation here. It must not run
        # inline: it takes about 22 s for a 30 MB daily file, which freezes
        # the window, and Windows greys a non-responding GUI so it looks
        # like a crash. Safe to thread because, unlike the analysis, this
        # path draws nothing.
        self.rinex_convert_btn.config(state="disabled", text="Converting...")
        self.update_idletasks()

        # If a trajectory is loaded, the angles must be computed at the rover's
        # position at each epoch rather than at one static point.
        # The .KIN box is right above this button, so use what is
        # in it; an empty box means a static station.
        kin_for_conversion = self.kin_file_path.get().strip() or None
        if kin_for_conversion and not os.path.exists(kin_for_conversion):
            messagebox.showwarning(
                "Trajectory Not Found",
                f"The .KIN file could not be found:\n{kin_for_conversion}\n\n"
                "The conversion will use the static station position instead."
            )
            kin_for_conversion = None

        # The position boxes are offered ONLY as a fallback: the bridge uses the
        # RINEX header's APPROX POSITION XYZ whenever the file has one, which is
        # which is the intended behaviour. This is what makes
        # a header-less file (common for smartphone and low-cost receivers)
        # convertible at all, instead of dead-ending on an error the GUI could
        # not answer.
        fallback_xyz = self._current_station_xyz()

        def task(progress_cb):
            # The bridge reports (percent, message); BackgroundTask wants
            # (current, total, message).
            return rinex_to_los_observations(
                rinex_path=rinex_path,
                sampling_interval=30.0,
                progress_callback=lambda pct, msg: progress_cb(pct, 100, msg),
                kin_path=kin_for_conversion,
                fallback_station_xyz=fallback_xyz,
            )

        def done(result):
            self.rinex_convert_btn.config(state="normal", text="Convert → CSV")
            if not result:
                messagebox.showerror("Conversion Error", "No result returned.")
                return
            csv_path = result['csv_path']
            stats = result['stats']
            self.los_obs_path.set(csv_path)      # auto-fill the obs file path

            # Say which position the angles were computed at. Silence here would
            # hide the difference between a kinematic and a static conversion.
            if stats.get('position_mode') == 'per-epoch trajectory':
                pos_lines = (
                    f"Position: per epoch from the trajectory "
                    f"({stats.get('trajectory_span_km', 0):.2f} km)\n"
                    f"Epochs positioned at the rover: "
                    f"{stats.get('epochs_from_trajectory', 0)}\n"
                )
                if stats.get('epochs_outside_trajectory', 0):
                    pos_lines += (
                        f"Outside the trajectory, static position used: "
                        f"{stats.get('epochs_outside_trajectory')}\n"
                    )
            else:
                pos_lines = ("Position: static (no .KIN trajectory loaded)\n"
                             f"Position source: {stats.get('position_source', 'RINEX header')}\n")

            messagebox.showinfo(
                "Conversion Complete",
                f"RINEX converted successfully!\n\n"
                f"Station: {stats.get('station', 'N/A')}\n"
                f"Date: {stats.get('date', 'N/A')}\n"
                f"Epochs: {stats.get('total_epochs', 0)}\n"
                f"Observations: {stats.get('total_observations', 0)}\n"
                f"Mean sats/epoch: {stats.get('mean_sats_per_epoch', 0):.1f}\n"
                + pos_lines +
                f"Time: {stats.get('processing_time_sec', 0)}s\n\n"
                f"CSV: {csv_path}"
            )

        def failed(exc):
            self.rinex_convert_btn.config(state="normal", text="Convert → CSV")
            # The one failure the user can actually fix from this window: the
            # file carries no APPROX POSITION XYZ and no usable position was
            # typed. Bring the greyed-out position boxes back and say so,
            # instead of reporting a dead end.
            if "APPROX POSITION" in str(exc):
                self.los_needs_manual_position = True
                self._set_position_fields_enabled(
                    True, "required: this RINEX file has no position")
                messagebox.showwarning(
                    "Station Position Needed",
                    "This RINEX file's header has no APPROX POSITION XYZ, so the "
                    "satellite angles cannot be computed without knowing where "
                    "the antenna was.\n\n"
                    "The user position fields have been enabled. Enter the "
                    "approximate station position, then press Convert again.\n\n"
                    "An approximate position is enough: a few hundred metres of "
                    "error moves the elevation angles by well under a degree."
                )
                return
            messagebox.showerror("Conversion Error", f"RINEX conversion failed:\n\n{exc}")

        run_with_progress(self, task, done, failed,
                          dialog_title="Converting RINEX to CSV...", can_cancel=False)

    def _browse_att_file(self):
        """Open file dialog for attitude (.ATT) file."""
        filepath = filedialog.askopenfilename(
            title="Select Attitude File (.ATT)",
            filetypes=[("ATT files", "*.ATT"), ("All files", "*.*")]
        )
        if filepath:
            self.att_file_path.set(filepath)

    def _browse_kin_file(self):
        """Open file dialog for trajectory (.KIN) file."""
        filepath = filedialog.askopenfilename(
            title="Select Trajectory File (.KIN)",
            filetypes=[("KIN files", "*.KIN"), ("All files", "*.*")]
        )
        if filepath:
            self.kin_file_path.set(filepath)

    def _on_att_file_changed(self, *args):
        """Disable manual Euler angle inputs when an ATT file is loaded."""
        has_att = bool(self.att_file_path.get().strip())
        state = "disabled" if has_att else "normal"
        self.yaw_entry.config(state=state)
        self.pitch_entry.config(state=state)
        self.roll_entry.config(state=state)

    def _add_azimuth_mask(self):
        try:
            az_from = float(self.azimuth_from_var.get())
            az_to = float(self.azimuth_to_var.get())
            el_limit = float(self.azimuth_el_var.get())
            if not (0 <= az_from <= 360 and 0 <= az_to <= 360): raise ValueError("Azimuth must be between 0 and 360.")
            if not (0 <= el_limit <= 90): raise ValueError("Elevation limit must be between 0 and 90.")
            self.azimuth_masks_list.append((az_from, az_to, el_limit))
            self.azimuth_mask_listbox.insert(tk.END, f"Az: {az_from:.1f}° to {az_to:.1f}°, El < {el_limit:.1f}°")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid mask: {e}")

    def _remove_azimuth_mask(self):
        selected_indices = self.azimuth_mask_listbox.curselection()
        if not selected_indices: return
        for i in sorted(selected_indices, reverse=True):
            self.azimuth_mask_listbox.delete(i)
            if 0 <= i < len(self.azimuth_masks_list):
                del self.azimuth_masks_list[i]

    def _update_antenna_selector(self, file_num, filename):
        if not filename or not os.path.exists(filename): return
        try:
            antex_data = read_antex_file(filename)
            if not antex_data or not antex_data.get('pco'):
                print(f"Warning: Failed to load ANTEX data from {filename}")
                return
                
            if file_num == 1: 
                self.antex_data_1 = antex_data
                file1_keys = sorted(self.antex_data_1['metadata'].keys())
                self.antenna_combobox['values'] = file1_keys
                self.antenna_combobox.config(state="readonly")
                if file1_keys:
                    self.antenna_combobox.set(file1_keys[0])
                self._update_antenna_match()
                self._on_antenna_selected()
                
            elif file_num == 2: 
                self.antex_data_2 = antex_data
                file2_keys = sorted(self.antex_data_2['metadata'].keys())
                if file2_keys:
                    ant_key = file2_keys[0]
                    self.antenna2_display.config(text=ant_key, foreground="white")
                self._update_antenna_match()
                # Use after() to defer the signal refresh - ensures UI is ready
                self.after(100, self._on_antenna_selected)
                
        except Exception as e:
            print(f"Error in _update_antenna_selector: {e}")
            import traceback
            traceback.print_exc()

    def _on_file1_path_changed(self, *args):
        """Called when File 1 path changes. Updates antenna combobox and match indicator."""
        filepath = self.antex1_path.get()
        
        if not filepath or not os.path.exists(filepath):
            # Path was cleared - reset combobox
            self.antex_data_1 = None
            self.antenna_combobox['values'] = []
            self.antenna_combobox.set('')
            self.antenna_combobox.config(state="disabled")
            self._update_antenna_match()
            self._update_signal_list()
            return
        
        # Load the file
        try:
            antex_data = read_antex_file(filepath)
            if antex_data and antex_data.get('pco'):
                self.antex_data_1 = antex_data
                antenna_keys = sorted(list(antex_data['pco'].keys()))
                self.antenna_combobox['values'] = antenna_keys
                self.antenna_combobox.config(state="readonly")
                if antenna_keys:
                    self.antenna_combobox.set(antenna_keys[0])
                    self._update_antenna_match()
                    self.after(100, self._update_signal_list)
        except Exception as e:
            print(f"Error loading File 1: {e}")

    def _on_file2_path_changed(self, *args):
        """Called when File 2 path changes. Updates antenna combobox and match indicator."""
        filepath = self.antex2_path.get()
        
        if not filepath or not os.path.exists(filepath):
            # Path was cleared - reset combobox
            self.antex_data_2 = None
            self.antenna_combobox_file2.config(state="disabled")
            self.antenna_combobox_file2.set("")
            self.antenna_combobox_file2['values'] = []
            self._update_antenna_match()
            self._on_antenna_selected()  # Refresh signal list
            return
        
        # Load the file
        try:
            antex_data = read_antex_file(filepath)
            if antex_data and antex_data.get('pco'):
                self.antex_data_2 = antex_data
                antenna_keys = list(antex_data['pco'].keys())
                if antenna_keys:
                    # Populate File 2 combobox with all antennas
                    self.antenna_combobox_file2['values'] = antenna_keys
                    self.antenna_combobox_file2.config(state="readonly")
                    # Auto-select first antenna or try to match File 1 type
                    file1_key = self.antenna_type_var.get()
                    if file1_key and self.antex_data_1:
                        meta1 = self.antex_data_1['metadata'].get(file1_key, {})
                        type1 = meta1.get('type', '').strip()
                        # Try to find matching type in File 2
                        matched = None
                        for key2 in antenna_keys:
                            meta2 = antex_data['metadata'].get(key2, {})
                            if meta2.get('type', '').strip() == type1:
                                matched = key2
                                break
                        if matched:
                            self.antenna_combobox_file2.set(matched)
                        else:
                            self.antenna_combobox_file2.set(antenna_keys[0])
                    else:
                        self.antenna_combobox_file2.set(antenna_keys[0])
                    self._update_antenna_match()
                    # In folder mode, populate signals from reference file (File 2)
                    if self.antex_mode.get() == "whole_folder":
                        self._populate_signals_from_file2()
                    else:
                        self.after(100, self._on_antenna_selected)  # Refresh signal list
        except Exception as e:
            print(f"Error loading File 2: {e}")
            self.antenna_combobox_file2.config(state="disabled")
            self.antenna_combobox_file2.set(f"(error: {e})")

    def _update_antenna_match(self):
        """Updates the match indicator showing if both antennas are compatible."""
        if not self.antex_data_1 or not self.antex_data_2:
            self.antenna_match_label.config(text="--", foreground="gray")
            return
        
        # Get SELECTED antenna from File 1
        selected_key_file1 = self.antenna_type_var.get()
        if not selected_key_file1:
            self.antenna_match_label.config(text="--", foreground="gray")
            return
        
        # Get SELECTED antenna from File 2 (independent selection)
        selected_key_file2 = self.antenna_type_var_file2.get()
        if not selected_key_file2:
            self.antenna_match_label.config(text="--", foreground="gray")
            return
            
        meta1 = self.antex_data_1.get('metadata', {})
        meta2 = self.antex_data_2.get('metadata', {})
        
        if selected_key_file1 not in meta1 or selected_key_file2 not in meta2:
            self.antenna_match_label.config(text="--", foreground="gray")
            return
        
        type1 = meta1.get(selected_key_file1, {}).get('type', '').strip()
        serial1 = meta1.get(selected_key_file1, {}).get('serial', '')
        type2 = meta2.get(selected_key_file2, {}).get('type', '').strip()
        serial2 = meta2.get(selected_key_file2, {}).get('serial', '')
        
        if type1 == type2:
            if serial1 == serial2:
                self.antenna_match_label.config(text="[OK] Same TYPE and serial number", foreground="green")
            else:
                self.antenna_match_label.config(text="[OK] Same TYPE (different serial number)", foreground="orange")
        else:
            # Different types - warn but allow (this is now a valid use case)
            self.antenna_match_label.config(text="[!] Different antenna types (comparing A vs B)", foreground="orange")

    def _update_signal_list(self, signals=None):
        """Updates the signal list based on loaded ANTEX data.
        Populates scrollable frame with Radiobuttons/Checkbuttons."""
        # Delegate to the existing signal list logic
        self._on_antenna_selected()

    def _on_antenna2_selected(self, event=None):
        """Called when File 2 antenna selection changes - refresh signals and match."""
        self._update_antenna_match()
        self._on_antenna_selected()

    # Handler for antenna selection - populates signal list with intersection support.
    def _on_antenna_selected(self, event=None):
        selected_key_file1 = self.antenna_type_var.get()
        if not selected_key_file1 or not self.antex_data_1:
            self._populate_signal_widgets([])
            return

        def get_signals(antex_data, key):
            signals = set()
            if not antex_data or key not in antex_data.get('pco', {}): return signals
            ant_pco_data = antex_data['pco'][key]
            for system_data in ant_pco_data.values():
                signals.update(system_data.keys())
            return signals

        signals1 = get_signals(self.antex_data_1, selected_key_file1)
        
        # --- Handle different analysis modes ---
        analysis_mode = self.antex_mode.get()
        
        if analysis_mode == "single_file":
            self.antenna_match_label.config(text="[Single File Mode]", foreground="cyan")
            signals_to_display = signals1
        elif not self.antex_data_2:
            # Two-file mode but File 2 not loaded yet
            self._populate_signal_widgets(["(Load ANTEX File 2 to see common signals)"])
            return
        else:
            selected_key_file2 = self.antenna_type_var_file2.get()
            
            if not selected_key_file2 or selected_key_file2 not in self.antex_data_2.get('pco', {}):
                self.antenna_match_label.config(
                    text="[!] Select antenna from File 2", 
                    foreground="orange"
                )
                signals_to_display = signals1
            else:
                signals2 = get_signals(self.antex_data_2, selected_key_file2)
                signals_to_display = signals1.intersection(signals2)
                
                if not signals_to_display:
                    self.antenna_match_label.config(
                        text="[!] No common signals - using File 1 signals only", 
                        foreground="orange"
                    )
                    signals_to_display = signals1
                else:
                    self._update_antenna_match()

        formatted_signals = []
        system_map = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS', 'J': 'QZSS', 'S': 'SBAS'}
        
        for sig_code in sorted(list(signals_to_display)):
            system_char = sig_code[0]
            system_name = system_map.get(system_char, "UNK")
            formatted_signals.append(f"{system_name} {sig_code}")

        # Add ionosphere-free combinations only if BOTH required signals are in common set
        has_g01 = 'G01' in signals_to_display; has_g02 = 'G02' in signals_to_display; has_g05 = 'G05' in signals_to_display
        if has_g01 and has_g02 and "GPS IF-LC L1/L2" not in formatted_signals: formatted_signals.append("GPS IF-LC L1/L2") 
        if has_g01 and has_g05 and "GPS IF-LC L1/L5" not in formatted_signals: formatted_signals.append("GPS IF-LC L1/L5") 
        has_e01 = 'E01' in signals_to_display; has_e5a = 'E05' in signals_to_display
        if has_e01 and has_e5a: formatted_signals.append("Galileo IF-LC E1/E5a")

        self._populate_signal_widgets(formatted_signals)

    def _populate_signal_widgets(self, formatted_signals):
        """Populate the signal ScrollableFrame with Radiobuttons or Checkbuttons."""
        # Clear existing widgets
        for widget in self.signal_container.winfo_children():
            widget.destroy()
        
        self.signal_vars = {}
        self.current_signals = formatted_signals
        
        if not formatted_signals:
            return
        
        # Update info label for filtering behavior
        common_label = "Showing COMMON signals (Intersection of File 1 & 2).\n(G=GPS, R=GLO, E=GAL, C=BDS)"
        if self.antex_mode.get() == "single_file":
            common_label = "Showing all signals from selected file.\n(G=GPS, R=GLO, E=GAL, C=BDS)"
        self.signal_info_label.config(text=common_label)
        
        is_multi = self.multifreq_var.get()
        mode_text = "(Single Select Mode - Pick one)" if not is_multi else "(Multi Select Mode - Pick multiple)"
        self.mode_label.config(text=mode_text)
        
        if is_multi:
            # Multi-frequency mode: Checkbuttons
            for sig_str in formatted_signals:
                var = tk.BooleanVar(value=False)
                var.trace_add("write", self._update_insights)
                self.signal_vars[sig_str] = var
                cb = ttk.Checkbutton(self.signal_container, text=sig_str, variable=var)
                cb.pack(anchor="w", padx=5, pady=1)
        else:
            # Single-frequency mode: Radiobuttons
            for sig_str in formatted_signals:
                rb = ttk.Radiobutton(self.signal_container, text=sig_str, 
                                     variable=self.selected_signal_var, value=sig_str)
                rb.pack(anchor="w", padx=5, pady=1)
            # Auto-select first signal
            if formatted_signals and not self.selected_signal_var.get():
                self.selected_signal_var.set(formatted_signals[0])

    def _populate_signals_from_file2(self):
        """Populate signal list from File 2 (reference file) - used in folder mode."""
        if not self.antex_data_2:
            return
        
        available_antennas = list(self.antex_data_2.get('pco', {}).keys())
        if not available_antennas:
            return
        
        first_antenna = available_antennas[0]
        signals = set()
        ant_pco_data = self.antex_data_2['pco'].get(first_antenna, {})
        for system_data in ant_pco_data.values():
            signals.update(system_data.keys())
        
        formatted_signals = []
        system_map = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BDS', 'J': 'QZSS', 'S': 'SBAS'}
        for sig_code in sorted(list(signals)):
            system_char = sig_code[0]
            system_name = system_map.get(system_char, "UNK")
            formatted_signals.append(f"{system_name} {sig_code}")
        
        has_g01 = 'G01' in signals; has_g02 = 'G02' in signals; has_g05 = 'G05' in signals
        if has_g01 and has_g02: formatted_signals.append("GPS IF-LC L1/L2") 
        if has_g01 and has_g05: formatted_signals.append("GPS IF-LC L1/L5")
        has_e01 = 'E01' in signals; has_e5a = 'E05' in signals
        if has_e01 and has_e5a: formatted_signals.append("Galileo IF-LC E1/E5a")
        
        self._populate_signal_widgets(formatted_signals)

    def _on_pos_mode_change(self, *args):
        new_mode = self.pos_type.get()
        if new_mode == self.last_pos_mode:
            self.update_position_mode()
            return
        try:
            c1 = self.coord1_var.get().strip(); c2 = self.coord2_var.get().strip(); c3 = self.coord3_var.get().strip()
            if not c1 or not c2 or not c3:
                self.last_pos_mode = new_mode
                self.update_position_mode()
                return
            val1, val2, val3 = float(c1), float(c2), float(c3)

            if self.last_pos_mode == "Ellipsoidal" and new_mode == "ECEF":
                xyz = ell_to_ecef(np.deg2rad(val1), np.deg2rad(val2), val3)
                self.coord1_var.set(f"{xyz[0]:.4f}"); self.coord2_var.set(f"{xyz[1]:.4f}"); self.coord3_var.set(f"{xyz[2]:.4f}")
            elif self.last_pos_mode == "ECEF" and new_mode == "Ellipsoidal":
                lat_rad, lon_rad, h = ecef_to_ell(val1, val2, val3)
                self.coord1_var.set(f"{np.rad2deg(lat_rad):.8f}"); self.coord2_var.set(f"{np.rad2deg(lon_rad):.8f}"); self.coord3_var.set(f"{h:.4f}")
            elif self.last_pos_mode == "ECEF" and new_mode == "Map":
                lat_rad, lon_rad, h = ecef_to_ell(val1, val2, val3)
                self.coord1_var.set(f"{np.rad2deg(lat_rad):.8f}"); self.coord2_var.set(f"{np.rad2deg(lon_rad):.8f}"); self.coord3_var.set(f"{h:.4f}")
        except ValueError: pass
        self.last_pos_mode = new_mode
        self.update_position_mode()

    def _update_pos_labels_only(self):
        mode = self.pos_type.get()
        if mode == "ECEF": l1, l2, l3 = "X [m]:", "Y [m]:", "Z [m]:"
        elif mode in ("Ellipsoidal", "Map"): l1, l2, l3 = "Lat [deg]:", "Lon [deg]:", "H [m]:"
        else: l1, l2, l3 = "Coord 1:", "Coord 2:", "Coord 3:"
        self.coord1_label.config(text=l1); self.coord2_label.config(text=l2); self.coord3_label.config(text=l3)

    def _on_y_axis_mode_change(self, event=None):
        """Show/hide custom Y-axis min/max fields based on selection."""
        if self.y_axis_mode_var.get() == "Custom":
            self.y_axis_custom_frame.pack(side="left", padx=(5, 0))
        else:
            self.y_axis_custom_frame.pack_forget()

    def update_position_mode(self):
        mode = self.pos_type.get()
        if mode == "Area":
            point_state = "disabled"; area_state = "normal"; l1, l2, l3 = "---", "---", "---"
        else:
            point_state = "normal"  # Always editable, including Map mode
            area_state = "disabled"
            if mode == "ECEF": l1, l2, l3 = "X [m]:", "Y [m]:", "Z [m]:"
            else: l1, l2, l3 = "Lat [deg]:", "Lon [deg]:", "H [m]:"
            if mode == "Map": self._launch_map_selector()
        self.coord1_label.config(text=l1); self.coord2_label.config(text=l2); self.coord3_label.config(text=l3)
        self.coord1_entry.config(state=point_state); self.coord2_entry.config(state=point_state); self.coord3_entry.config(state=point_state)
        self.lat_min_entry.config(state=area_state); self.lat_max_entry.config(state=area_state)
        self.lon_min_entry.config(state=area_state); self.lon_max_entry.config(state=area_state); self.grid_step_entry.config(state=area_state)

    def _parse_interval_input(self):
        raw_val = self.interval_var.get().strip()
        if ":" in raw_val:
            try:
                parts = raw_val.split(":")
                if len(parts) != 2: raise ValueError
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError: raise ValueError(f"Invalid HH:MM format: '{raw_val}'")
        else:
            try: return int(raw_val)
            except ValueError: raise ValueError(f"Invalid interval number: '{raw_val}'")

    def _build_config_from_gui(self, analysis_type: str):
        try:
            interval_minutes = self._parse_interval_input()
            config = {
                'analysis_mode': self.antex_mode.get(),
                'orbit_type': self.orbit_type_var.get(),
                'weighting': self.weight_var.get(),
                'tropo_model': self.tropo_mapping_var.get(),
                'gradient_active': self.gradient_var.get(),
                'antenna_type': self.antenna_type_var.get().strip(), 
                'sampling_rate_sec': int(self.samplerate_var.get()),
                'interval_min': interval_minutes, 
                'orbit_dir': os.path.join(project_root, 'data', 'orbit'),
                'elevation_mask_active': self.elevation_mask_var.get(),
                'azimuth_mask_active': self.azimuth_mask_var.get(),
                'azimuth_mask_values': self.azimuth_masks_list.copy(),
                'obstruction_mask_active': self.obstruction_mask_var.get(),
                'obstruction_mask_file': self.obstruction_mask_path.get().strip(),
                'computation_mode': self.computation_mode_var.get(),
            }
            # LoS-specific config
            if config['computation_mode'] == 'los':
                los_path = self.los_obs_path.get().strip()
                rinex_path = self.rinex_obs_path.get().strip()

                if not los_path and not rinex_path:
                    messagebox.showerror(
                        "LoS Error",
                        "No observation file selected!\n\n"
                        "Please provide either:\n"
                        "  • A satellite observation CSV file, or\n"
                        "  • A RINEX observation file (auto-converted)"
                    )
                    return None

                if los_path:
                    config['los_obs_file'] = los_path
                if rinex_path:
                    config['rinex_obs_file'] = rinex_path

                try:
                    config['heading_deg'] = float(self.heading_deg_var.get())
                except ValueError:
                    config['heading_deg'] = 0.0
                try:
                    config['pitch_deg'] = float(self.pitch_deg_var.get())
                except ValueError:
                    config['pitch_deg'] = 0.0
                try:
                    config['roll_deg'] = float(self.roll_deg_var.get())
                except ValueError:
                    config['roll_deg'] = 0.0
                # Optional ATT/KIN files
                att_path = self.att_file_path.get().strip()
                if att_path:
                    config['att_file'] = att_path
                kin_path = self.kin_file_path.get().strip()
                if kin_path:
                    config['kin_file'] = kin_path
                # Optional resample interval
                resample_str = self.los_resample_var.get().strip()
                if resample_str:
                    try:
                        config['los_resample_sec'] = float(resample_str)
                    except ValueError:
                        pass
            if config['elevation_mask_active']: config['elevation_mask_angle'] = float(self.elevation_angle_var.get())
            if config['analysis_mode'] in ('two_files', 'Datei'):
                config['antex_file_1'] = self.antex1_path.get(); config['antex_file_2'] = self.antex2_path.get()
            elif config['analysis_mode'] == 'single_file':
                config['antex_file_1'] = self.antex1_path.get()
            elif config['analysis_mode'] in ('whole_folder', 'Ordner'):
                config['folder_path'] = self.antex1_path.get()
                if self.antex2_path.get(): config['antex_file_2'] = self.antex2_path.get()

            # Only access antex_data if both are loaded
            if self.antex_data_1 and self.antex_data_2:
                # Use the user's actual File 2 dropdown selection, not auto-matching
                if hasattr(self, 'antenna_type_var_file2') and self.antenna_type_var_file2.get():
                    # User has made a selection in File 2 dropdown
                    config['_antenna_type_file2'] = self.antenna_type_var_file2.get()
                else:
                    # Fallback: If no File 2 dropdown exists, use first antenna from File 2
                    if self.antex_data_2.get('metadata'):
                        first_key = list(self.antex_data_2['metadata'].keys())[0]
                        config['_antenna_type_file2'] = first_key
                        print(f"  [INFO] No File 2 selection, using first antenna: '{first_key}'")

            pos_mode = self.pos_type.get()
            if pos_mode != "Area":
                c1 = float(self.coord1_var.get()); c2 = float(self.coord2_var.get()); c3 = float(self.coord3_var.get())
                if pos_mode in ('Ellipsoidal', 'Map'): config['position'] = ell_to_ecef(np.deg2rad(c1), np.deg2rad(c2), c3); config['latitude'] = c1; config['longitude'] = c2; config['height'] = c3
                elif pos_mode == 'ECEF': config['position'] = np.array([c1, c2, c3]); config['x_coordinate'] = c1; config['y_coordinate'] = c2; config['z_coordinate'] = c3
                config['position_type'] = pos_mode
            else:
                config['position'] = None; config['position_type'] = pos_mode
                config['lat_min'] = float(self.lat_min_var.get()); config['lat_max'] = float(self.lat_max_var.get())
                config['lon_min'] = float(self.lon_min_var.get()); config['lon_max'] = float(self.lon_max_var.get())
                config['global_grid_step'] = float(self.grid_step_var.get())

                config['global_grid_step'] = float(self.grid_step_var.get())
 
            # Read signal selection from ScrollableFrame variables
            parsed_signals = self._selected_signal_codes()

            if not parsed_signals:
                messagebox.showerror("Signal Error", "No signal selected!\n\nPlease select at least one signal before running the analysis.")
                return None
            config['signals'] = parsed_signals

            if analysis_type == 'timeline':
                config['start_date'] = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
                config['end_date'] = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
                # Also include start_time/end_time for validation (use the date with time)
                def normalize_time(time_str):
                    parts = time_str.strip().split(":")
                    if len(parts) == 2:
                        h, m = parts[0].zfill(2), parts[1].zfill(2)
                        return f"{h}:{m}"
                    return time_str
                start_time_norm = normalize_time(self.start_time_var.get())
                end_time_norm = normalize_time(self.end_time_var.get())
                config['start_time'] = datetime.strptime(f"{self.start_date_var.get()} {start_time_norm}", "%Y-%m-%d %H:%M")
                config['end_time'] = datetime.strptime(f"{self.end_date_var.get()} {end_time_norm}", "%Y-%m-%d %H:%M")
            else:
                # Normalize time format (handle single-digit hours like "0:00" -> "00:00")
                def normalize_time(time_str):
                    parts = time_str.strip().split(":")
                    if len(parts) == 2:
                        h, m = parts[0].zfill(2), parts[1].zfill(2)
                        return f"{h}:{m}"
                    return time_str
                start_time_norm = normalize_time(self.start_time_var.get())
                end_time_norm = normalize_time(self.end_time_var.get())
                config['start_time'] = datetime.strptime(f"{self.start_date_var.get()} {start_time_norm}", "%Y-%m-%d %H:%M")
                config['end_time'] = datetime.strptime(f"{self.end_date_var.get()} {end_time_norm}", "%Y-%m-%d %H:%M")
            return config
        except ValueError as ve: messagebox.showerror("Input Error", f"Error parsing input: {ve}"); return None
        except Exception as e: messagebox.showerror("Config Error", f"Failed to build config: {e}"); return None

    def _load_processing_file(self):
        configs_dir = os.path.join(project_root, 'configs')
        if not os.path.isdir(configs_dir):
            configs_dir = project_root
        filepath = filedialog.askopenfilename(title="Select a processing file", initialdir=configs_dir, filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if not filepath: return
        try:
            config = read_config_file(filepath)
            self._populate_gui_from_config(config)
            messagebox.showinfo("Success", f"Loaded: {os.path.basename(filepath)}")
        except Exception as e: messagebox.showerror("Load Error", f"{e}")

    def _show_create_config_dialog(self):
        """Opens a dialog to create a new config file with a structured template."""
        dialog = tk.Toplevel(self)
        dialog.title("Create New Config File")
        dialog.geometry("650x850")  # Increased height for more settings
        dialog.resizable(True, True)
        
        # Create scrollable frame
        canvas = tk.Canvas(dialog, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=15)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        dialog.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store entry variables
        entries = {}
        
        def add_section(parent, title):
            frame = ttk.LabelFrame(parent, text=title, padding=10)
            frame.pack(fill="x", pady=5)
            return frame
        
        def add_field(parent, label, default="", width=50):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=25, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            entry = ttk.Entry(row, textvariable=var, width=width)
            entry.pack(side="left", fill="x", expand=True)
            return var
        
        # --- Analysis Mode --- (N1: Prefill with current GUI values)
        mode_frame = add_section(scrollable_frame, "Analysis Mode")
        entries['analysis_mode'] = add_field(mode_frame, "analysis_mode:", self.antex_mode.get())
        ttk.Label(mode_frame, text="Options: two_files, single_file, whole_folder", foreground="gray").pack(anchor="w")
        
        # --- ANTEX Files ---
        antex_frame = add_section(scrollable_frame, "ANTEX Files")
        entries['antex_file_1'] = add_field(antex_frame, "antex_file_1:", self.antex1_path.get() or "data/antex/your_file1.atx")
        # C1: File 2 row (hideable for single-file mode)
        file2_row = ttk.Frame(antex_frame)
        file2_row.pack(fill="x", pady=2)
        ttk.Label(file2_row, text="antex_file_2:", width=25, anchor="w").pack(side="left")
        entries['antex_file_2'] = tk.StringVar(value=self.antex2_path.get() or "data/antex/your_file2.atx")
        ttk.Entry(file2_row, textvariable=entries['antex_file_2'], width=50).pack(side="left", fill="x", expand=True)
        entries['antenna_type'] = add_field(antex_frame, "antenna_type:", self.antenna_type_var.get() if hasattr(self, 'antenna_type_var') else "")
        ttk.Label(antex_frame, text="Antenna type must match exactly what appears in the ANTEX file", foreground="gray").pack(anchor="w")
        
        # C1: Toggle File 2 visibility based on analysis mode
        def _toggle_file2_visibility(*args):
            if entries['analysis_mode'].get().strip() == 'single_file':
                file2_row.pack_forget()
            else:
                # Re-insert after antex_file_1 row
                file2_row.pack(fill="x", pady=2, before=file2_row.master.winfo_children()[2])
        entries['analysis_mode'].trace_add('write', _toggle_file2_visibility)
        _toggle_file2_visibility()  # Apply initial state
        
        # --- Date/Time Settings ---
        time_frame = add_section(scrollable_frame, "Date/Time Settings")
        entries['start_date'] = add_field(time_frame, "start_date (YYYY-MM-DD):", self.start_date_var.get() if hasattr(self, 'start_date_var') else "2024-01-01")
        entries['start_time'] = add_field(time_frame, "start_time (HH:MM):", self.start_time_var.get() if hasattr(self, 'start_time_var') else "00:00")
        entries['end_date'] = add_field(time_frame, "end_date (YYYY-MM-DD):", self.end_date_var.get() if hasattr(self, 'end_date_var') else "2024-01-01")
        entries['end_time'] = add_field(time_frame, "end_time (HH:MM):", self.end_time_var.get() if hasattr(self, 'end_time_var') else "23:59")
        entries['sampling_rate_sec'] = add_field(time_frame, "sampling_rate_sec:", "30")  # Default 30 seconds
        entries['interval_min'] = add_field(time_frame, "interval_min:", self.interval_var.get() if hasattr(self, 'interval_var') else "1440")
        
        # --- Processing Options ---
        proc_frame = add_section(scrollable_frame, "Processing Options")
        entries['orbit_type'] = add_field(proc_frame, "orbit_type:", self.orbit_type_var.get() if hasattr(self, 'orbit_type_var') else "final")
        ttk.Label(proc_frame, text="Options: final, broadcast", foreground="gray").pack(anchor="w")
        entries['weighting'] = add_field(proc_frame, "weighting:", self.weight_var.get() if hasattr(self, 'weight_var') else "sin")
        ttk.Label(proc_frame, text="Options: sin, sin2, unit", foreground="gray").pack(anchor="w")
        entries['tropo_model'] = add_field(proc_frame, "tropo_model:", self.tropo_mapping_var.get() if hasattr(self, 'tropo_mapping_var') else "GMF")
        ttk.Label(proc_frame, text="Options: 1/sin(Elevation), GMF, None", foreground="gray").pack(anchor="w")
        # Get selected signals
        # Get selected signals
        selected_signals = []
        if self.multifreq_var.get():
             for sig, var in self.signal_vars.items():
                 if var.get(): selected_signals.append(sig)
        else:
             sel = self.selected_signal_var.get()
             if sel: selected_signals.append(sel)
        if not selected_signals: selected_signals = ["G01"]
        entries['signals'] = add_field(proc_frame, "signals:", ",".join(selected_signals) if selected_signals else "G01")
        ttk.Label(proc_frame, text="Examples: G01, G02, E01, G01,G02 (comma separated)", foreground="gray").pack(anchor="w")
        entries['gradient_active'] = add_field(proc_frame, "gradient_active:", "True" if (hasattr(self, 'gradient_var') and self.gradient_var.get()) else "False")
        ttk.Label(proc_frame, text="Options: True, False (enables gradient model)", foreground="gray").pack(anchor="w")
        
        # --- Position Settings --- (C3: renamed, C4: [deg] notation)
        pos_frame = add_section(scrollable_frame, "Position")
        entries['position_type'] = add_field(pos_frame, "position_type:", self.pos_type.get() if hasattr(self, 'pos_type') else "Ellipsoidal")
        ttk.Label(pos_frame, text="Options: Ellipsoidal, ECEF, Area", foreground="gray").pack(anchor="w")
        entries['latitude'] = add_field(pos_frame, "latitude [deg] or X [m]:", self.coord1_var.get() if hasattr(self, 'coord1_var') else "52.385")
        entries['longitude'] = add_field(pos_frame, "longitude [deg] or Y [m]:", self.coord2_var.get() if hasattr(self, 'coord2_var') else "9.713")
        entries['height'] = add_field(pos_frame, "height [m] or Z [m]:", self.coord3_var.get() if hasattr(self, 'coord3_var') else "65.0")
        
        # --- Orbit Directory ---
        orbit_frame = add_section(scrollable_frame, "Orbit Directory")
        entries['orbit_dir'] = add_field(orbit_frame, "orbit_dir:", self.orbit_dir.get() if hasattr(self, 'orbit_dir') else "data/orbit")
        
        # --- Elevation Mask ---
        mask_frame = add_section(scrollable_frame, "Elevation Mask")
        # Get the actual elevation angle value (not the boolean checkbox state)
        elev_mask_default = "10"
        try:
            if hasattr(self, 'elevation_mask_var') and self.elevation_mask_var.get():
                elev_mask_default = str(self.elevation_angle_var.get())
        except Exception:
            pass
        entries['elevation_mask'] = add_field(mask_frame, "elevation_mask [deg]:", elev_mask_default)
        ttk.Label(mask_frame, text="Minimum elevation angle in degrees (0-90)", foreground="gray").pack(anchor="w")

        # Obstruction mask file (horizon profile from RINEX-Masker)
        obstruction_default = ""
        try:
            if hasattr(self, 'obstruction_mask_var') and self.obstruction_mask_var.get():
                obstruction_default = self.obstruction_mask_path.get().strip()
        except Exception:
            pass
        entries['obstruction_mask_file'] = add_field(mask_frame, "obstruction_mask_file:", obstruction_default)
        ttk.Label(mask_frame, text="Optional: mask exported by RINEX-Masker (leave empty for none)",
                  foreground="gray").pack(anchor="w")
        
        # --- Azimuth Masks ---
        azimuth_frame = add_section(scrollable_frame, "Azimuth Masks")
        # Get current azimuth masks from GUI (now includes per-range elevation)
        az_masks_str = ""
        if hasattr(self, 'azimuth_masks_list') and self.azimuth_masks_list:
            az_parts = []
            for mask in self.azimuth_masks_list:
                try:
                    # azimuth_masks_list stores tuples: (az_from, az_to, el_limit)
                    az_from, az_to, el_limit = mask[0], mask[1], mask[2]
                    az_parts.append(f"{az_from}-{az_to},{el_limit}")
                except (IndexError, TypeError):
                    pass
            az_masks_str = ";".join(az_parts)
        entries['azimuth_masks'] = add_field(azimuth_frame, "azimuth_masks [deg]:", az_masks_str)
        ttk.Label(azimuth_frame, text="Format: start1-end1,elev1;start2-end2,elev2 (e.g., 0-45,5;270-360,7.5)", foreground="gray").pack(anchor="w")
        
        # --- Global/Regional Bounds ---
        global_frame = add_section(scrollable_frame, "Global/Regional Analysis Bounds")
        entries['lat_min'] = add_field(global_frame, "lat_min [deg]:", self.lat_min_var.get() if hasattr(self, 'lat_min_var') else "-90")
        entries['lat_max'] = add_field(global_frame, "lat_max [deg]:", self.lat_max_var.get() if hasattr(self, 'lat_max_var') else "90")
        entries['lon_min'] = add_field(global_frame, "lon_min [deg]:", self.lon_min_var.get() if hasattr(self, 'lon_min_var') else "-180")
        entries['lon_max'] = add_field(global_frame, "lon_max [deg]:", self.lon_max_var.get() if hasattr(self, 'lon_max_var') else "180")
        entries['global_grid_step'] = add_field(global_frame, "global_grid_step [deg]:", self.grid_step_var.get() if hasattr(self, 'grid_step_var') else "10")
        ttk.Label(global_frame, text="For area-based/global analysis only", foreground="gray").pack(anchor="w")
        
        # --- Save Button ---
        def save_config():
            # Create configs folder if it doesn't exist
            config_dir = os.path.join(os.getcwd(), 'configs')
            os.makedirs(config_dir, exist_ok=True)
            
            # Ask for filename
            filepath = filedialog.asksaveasfilename(
                initialdir=config_dir,
                title="Save Config File",
                defaultextension=".txt",
                filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
            )
            if not filepath:
                return
            
            try:
                with open(filepath, 'w') as f:
                    f.write("# ============================================\n")
                    f.write("# PCC-Explorer Configuration File\n")
                    f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# ============================================\n\n")
                    
                    f.write("# --- Analysis Mode ---\n")
                    f.write(f"analysis_mode={entries['analysis_mode'].get()}\n\n")
                    
                    f.write("# --- ANTEX Files ---\n")
                    f.write(f"antex_file_1={entries['antex_file_1'].get()}\n")
                    # C1: Only write antex_file_2 if not single_file mode
                    if entries['analysis_mode'].get().strip() != 'single_file':
                        f.write(f"antex_file_2={entries['antex_file_2'].get()}\n")
                    f.write(f"antenna_type={entries['antenna_type'].get()}\n\n")
                    
                    f.write("# --- Date/Time Settings ---\n")
                    f.write(f"start_date={entries['start_date'].get()}\n")
                    f.write(f"start_time={entries['start_time'].get()}\n")
                    f.write(f"end_date={entries['end_date'].get()}\n")
                    f.write(f"end_time={entries['end_time'].get()}\n")
                    f.write(f"sampling_rate_sec={entries['sampling_rate_sec'].get()}\n")
                    f.write(f"interval_min={entries['interval_min'].get()}\n\n")
                    
                    f.write("# --- Processing Options ---\n")
                    f.write(f"orbit_type={entries['orbit_type'].get()}\n")
                    f.write(f"weighting={entries['weighting'].get()}\n")
                    f.write(f"tropo_model={entries['tropo_model'].get()}\n")
                    f.write(f"signals={entries['signals'].get()}\n")
                    f.write(f"gradient_active={entries['gradient_active'].get()}\n\n")
                    
                    f.write("# --- Position ---\n")
                    f.write(f"position_type={entries['position_type'].get()}\n")
                    f.write(f"latitude={entries['latitude'].get()}\n")
                    f.write(f"longitude={entries['longitude'].get()}\n")
                    f.write(f"height={entries['height'].get()}\n\n")
                    
                    f.write("# --- Orbit Directory ---\n")
                    f.write(f"orbit_dir={entries['orbit_dir'].get()}\n\n")
                    
                    f.write("# --- Elevation Mask ---\n")
                    f.write(f"elevation_mask={entries['elevation_mask'].get()}\n")
                    obstruction_file = entries['obstruction_mask_file'].get().strip()
                    f.write(f"obstruction_mask_file={obstruction_file}\n")
                    f.write(f"obstruction_mask_active={'true' if obstruction_file else 'false'}\n")
                    f.write("# Horizon profile exported by RINEX-Masker (pcc-mask-v1 JSON or text mask)\n\n")
                    
                    f.write("# --- Azimuth Masks ---\n")
                    f.write(f"azimuth_masks={entries['azimuth_masks'].get()}\n")
                    f.write("# Format: start1-end1,elev1;start2-end2,elev2 (e.g., 0-45,5;270-360,7.5)\n\n")
                    
                    f.write("# --- Global/Regional Analysis Bounds ---\n")
                    f.write(f"lat_min={entries['lat_min'].get()}\n")
                    f.write(f"lat_max={entries['lat_max'].get()}\n")
                    f.write(f"lon_min={entries['lon_min'].get()}\n")
                    f.write(f"lon_max={entries['lon_max'].get()}\n")
                    f.write(f"global_grid_step={entries['global_grid_step'].get()}\n")
                
                messagebox.showinfo("Success", f"Config file saved to:\n{filepath}")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save config: {e}")
        
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill="x", pady=15)
        ttk.Button(btn_frame, text="Save Config File", command=save_config, bootstyle="success").pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, bootstyle="secondary-outline").pack(side="right", padx=5)


    def _populate_gui_from_config(self, config):
        try:
            self.antex_mode.set(config.get('analysis_mode', 'two_files'))
            self.update_antex_mode()
            # Resolve file paths BEFORE setting path variables.
            # Setting path_var triggers trace callbacks that reset antenna/signal
            # state if the file doesn't exist (e.g. config from another machine).
            path_configs = [
                (1, self.antex1_path, config.get('antex_file_1', config.get('folder_path', ''))),
                (2, self.antex2_path, config.get('antex_file_2', '')),
            ]
            for file_num, path_var, raw_path in path_configs:
                if not raw_path:
                    continue
                raw_path = raw_path.replace('/', os.sep)
                # Resolve relative paths against project root
                if not os.path.isabs(raw_path):
                    resolved = os.path.join(project_root, raw_path)
                    if os.path.exists(resolved):
                        raw_path = resolved
                # Portable fallback: config from another machine — find same
                # filename in local data/antex/ folder
                if not os.path.isfile(raw_path):
                    basename = os.path.basename(raw_path)
                    local_antex = os.path.join(project_root, 'data', 'antex', basename)
                    if os.path.isfile(local_antex):
                        raw_path = local_antex
                        print(f"  [Config] Resolved '{basename}' to local path")
                # Only set path variable once we have a valid, resolved path
                path_var.set(raw_path)
                if os.path.isfile(raw_path):
                    self._update_antenna_selector(file_num, raw_path)
        except Exception as e:
            print(f"Warning: Error loading ANTEX settings: {e}")
        
        try:
            pos_type = config.get('position_type', 'Ellipsoidal')
            self.pos_type.set(pos_type)
            self.update_position_mode()
            if pos_type == 'Ellipsoidal':
                self.coord1_var.set(str(config.get('latitude', ''))); self.coord2_var.set(str(config.get('longitude', ''))); self.coord3_var.set(str(config.get('height', '')))
            elif pos_type == 'ECEF':
                self.coord1_var.set(str(config.get('x_coordinate', ''))); self.coord2_var.set(str(config.get('y_coordinate', ''))); self.coord3_var.set(str(config.get('z_coordinate', '')))
        except Exception as e:
            print(f"Warning: Error loading position settings: {e}")
        
        # Handle Date/Time loading (Robust to str or datetime)
        try:
            def _fmt_date(val, fmt):
                if isinstance(val, str): return val.split(" ")[0] # Assume YYYY-MM-DD
                return val.strftime(fmt)
                
            def _fmt_time(val, fmt):
                if isinstance(val, str): 
                    if " " in val: return val.split(" ")[1][:5]
                    return val[:5] # Force HH:MM
                return val.strftime(fmt)

            s_date = config.get('start_date', config.get('start_time'))
            s_time = config.get('start_time')
            e_date = config.get('end_date', config.get('end_time'))
            e_time = config.get('end_time')

            if not isinstance(s_date, (str, datetime)): s_date = datetime.now()
            if not isinstance(e_date, (str, datetime)): e_date = datetime.now()

            self.start_date_var.set(_fmt_date(s_date, '%Y-%m-%d'))
            self.start_time_var.set(_fmt_time(s_time if s_time else "00:00", '%H:%M'))
            self.end_date_var.set(_fmt_date(e_date, '%Y-%m-%d'))
            self.end_time_var.set(_fmt_time(e_time if e_time else "23:59", '%H:%M'))
            self.samplerate_var.set(str(int(config.get('sampling_rate_sec', 300))))
            
            # Handle interval_min: can be integer (minutes) or HH:MM string
            interval_raw = config.get('interval_min', 1440)
            if isinstance(interval_raw, str) and ':' in interval_raw:
                # Already in HH:MM format
                self.interval_var.set(interval_raw)
            else:
                mins = int(interval_raw)
                hh = mins // 60; mm = mins % 60
                self.interval_var.set(f"{hh:02d}:{mm:02d}")
        except Exception as e:
            print(f"Warning: Error loading date/time settings: {e}")

        try:
            target_ant = config.get('antenna_type', '')
            if target_ant and hasattr(self, 'antenna_combobox') and target_ant in self.antenna_combobox['values']:
                 self.antenna_type_var.set(target_ant)
                 self._on_antenna_selected()
        except Exception as e:
            print(f"Warning: Error loading antenna type: {e}")
        
        try:
            if 'orbit_type' in config: self.orbit_type_var.set(config['orbit_type'])
            if 'tropo_model' in config: self.tropo_mapping_var.set(config['tropo_model'])
            if 'gradient_active' in config: self.gradient_var.set(config['gradient_active'])
            if 'weighting' in config: self.weight_var.set(config['weighting'])
            if 'orbit_dir' in config and hasattr(self, 'orbit_dir'): self.orbit_dir.set(config['orbit_dir'])
        except Exception as e:
            print(f"Warning: Error loading processing options: {e}")
            
        try:
            # Handle elevation mask: support both old format (elevation_mask_active/angle)
            # and new format (elevation_mask as a single numeric value)
            if 'elevation_mask_active' in config:
                self.elevation_mask_var.set(config['elevation_mask_active'])
                self.elevation_angle_var.set(str(config.get('elevation_mask_angle', 5.0)))
            elif 'elevation_mask' in config:
                # New config format: elevation_mask=7 means mask is active with 7 deg
                try:
                    mask_val = float(config['elevation_mask'])
                    self.elevation_mask_var.set(True)
                    self.elevation_angle_var.set(str(mask_val))
                except (ValueError, TypeError):
                    self.elevation_mask_var.set(False)

            # Obstruction mask (horizon profile from RINEX-Masker)
            if 'obstruction_mask_file' in config and config['obstruction_mask_file']:
                self.obstruction_mask_path.set(config['obstruction_mask_file'])
                self.obstruction_mask_var.set(bool(config.get('obstruction_mask_active', True)))

            self.update_mask_widgets()
        except Exception as e:
            print(f"Warning: Error loading elevation mask: {e}")
            
        if 'signals' in config:
            self.update_idletasks()
            self.after(500, lambda: self._restore_signal_selection(config['signals']))

    def _restore_signal_selection(self, config_signals):
        if not self.antenna_type_var.get(): return
        
        config_signals_list = config_signals if isinstance(config_signals, list) else [s.strip() for s in config_signals.split(',')]
        
        # Decide if we need Multi mode
        if len(config_signals_list) > 1:
            self.multifreq_var.set(True)
        else:
            self.multifreq_var.set(False)

        # Refresh list first to correct mode widgets
        self._update_signal_list()

        # Apply selection
        # Map Config Codes (IF_...) to GUI Labels
        CODE_TO_LABEL = {
            "IF_G01_G02": "GPS IF-LC L1/L2",
            "IF_G01_G05": "GPS IF-LC L1/L5",
            "IF_R01_R02": "GLONASS IF-LC R1/R2",
            "IF_R01_R03": "GLONASS IF-LC G1/G3",
            "IF_E01_E05": "Galileo IF-LC E1/E5a",
            "IF_E01_E07": "Galileo IF-LC E1/E5b",
            "IF_E01_E08": "Galileo IF-LC E1/E5",
            "IF_E01_E06": "Galileo IF-LC E1/E6",
            "IF_C02_C07": "Beidou IF-LC B1/B2",
            "IF_C02_C06": "Beidou IF-LC B1/B3"
        }

        # Apply selection
        if self.multifreq_var.get():
             # Set Checkbuttons
             for sig_code in config_signals_list:
                 # Try direct match (e.g. "G01") or mapped match
                 sig_label = CODE_TO_LABEL.get(sig_code, sig_code)
                 
                 if sig_label in self.signal_vars:
                     self.signal_vars[sig_label].set(True)
                 elif sig_code in self.signal_vars: # Fallback
                     self.signal_vars[sig_code].set(True)
        else:
             # Set Radiobutton
             if config_signals_list:
                 raw_sig = config_signals_list[0]
                 sig_label = CODE_TO_LABEL.get(raw_sig, raw_sig)
                 self.selected_signal_var.set(sig_label)

    def browse_file(self, path_variable, file_num):
        initial_dir = os.path.join(project_root, 'data', 'antex')
        filename = filedialog.askopenfilename(title="Select ANTEX", initialdir=initial_dir, filetypes=(("ANTEX", "*.atx;*.ATX;*.atx2;*.ATX2"), ("All", "*.*")))
        if filename:
            path_variable.set(filename)
            self._update_antenna_selector(file_num, filename)

    def browse_folder(self, path_variable):
        initial_dir = os.path.join(project_root, 'data', 'antex')
        foldername = filedialog.askdirectory(title="Select Folder", initialdir=initial_dir)
        if foldername:
            path_variable.set(foldername)
            self.antenna_combobox.set(''); self.antenna_combobox.config(state="disabled")
            self._update_signal_list([]) # Clear signals
            # Disable list if needed, or _update_signal_list handles empty state

    def update_antex_mode(self):
        mode = self.antex_mode.get()
        if mode == "two_files":
            self.antex1_label.config(text="File 1:")
            self.antex2_label.grid()  # Show File 2 row
            self.antex2_entry.grid()
            self.antex2_browse.grid()
            self.antex1_browse.config(command=lambda: self.browse_file(self.antex1_path, 1))
            self.antex2_browse.config(command=lambda: self.browse_file(self.antex2_path, 2))
            self.antex2_entry.config(state="normal")
            self.antex2_browse.config(state="normal")
            if self.antex1_path.get(): self._update_antenna_selector(1, self.antex1_path.get())
            # Show File 2 antenna row
            self.file2_row.pack(fill="x", pady=2)
        elif mode == "single_file":
            self.antex1_label.config(text="ANTEX File:")
            # Hide File 2 row completely
            self.antex2_label.grid_remove()
            self.antex2_entry.grid_remove()
            self.antex2_browse.grid_remove()
            self.antex2_path.set("")  # Clear File 2 path
            self.antex_data_2 = None
            self.antex1_browse.config(command=lambda: self.browse_file(self.antex1_path, 1))
            if self.antex1_path.get(): self._update_antenna_selector(1, self.antex1_path.get())
            # Hide File 2 antenna row
            self.file2_row.pack_forget()
            self.antenna_type_var_file2.set("")
        elif mode == "whole_folder":
            self.antex1_label.config(text="Folder:")
            self.antex2_label.config(text="Reference File:")
            self.antex2_label.grid()  # Show File 2 row
            self.antex2_entry.grid()
            self.antex2_browse.grid()
            self.antex1_browse.config(command=lambda: self.browse_folder(self.antex1_path))
            self.antex2_browse.config(command=lambda: self.browse_file(self.antex2_path, 2))
            # In folder mode, use antenna and signals from reference file (File 2)
            self.antenna_combobox.set(''); self.antenna_combobox.config(state="disabled")
            # If reference file is loaded (antex_data_2), populate signals from it
            if self.antex_data_2:
                self._populate_signals_from_file2()
            else:
                self._update_signal_list(["(Load Reference File)"])
            # Hide File 2 antenna combobox for folder mode
            self.antenna_combobox_file2.grid_remove()

    def update_frequency_mode(self):
        """Called when the frequency mode toggle is switched."""
        self._update_signal_list()

    def _update_signal_list(self, signals=None):
        """Populates scrollable frame with Radiobuttons/Checkbuttons."""
        if signals is not None:
             self.current_signals = signals
        else:
             signals = self.current_signals

        # Clear existing widgets
        for widget in self.signal_container.winfo_children():
            widget.destroy()
        
        # Clear previous selection so user must explicitly pick a signal
        self.selected_signal_var.set("")

        if not signals:
            ttk.Label(self.signal_container, text="No signals available", font=("Helvetica", 9, "italic")).pack(anchor="w", padx=5, pady=5)
            return

        is_multi = self.multifreq_var.get()
        
        # Update mode label
        if is_multi:
            self.mode_label.config(text="(Multi-Select Mode - Pick any)", foreground="#5cb85c") 
        else:
            self.mode_label.config(text="(Single Select Mode - Pick one)", foreground="#5bc0de") 

        self.signal_vars = {} 

        for sig in signals:
            frame = ttk.Frame(self.signal_container)
            frame.pack(fill="x", pady=1)
            
            if is_multi:
                # Checkbutton
                var = tk.BooleanVar(value=False)
                var.trace_add("write", self._update_insights)
                self.signal_vars[sig] = var
                cb = ttk.Checkbutton(frame, text=sig, variable=var)
                cb.pack(side="left", padx=5)
            else:
                # Radiobutton
                rb = ttk.Radiobutton(frame, text=sig, variable=self.selected_signal_var, value=sig)
                rb.pack(side="left", padx=5)
        
        # Single mode: Do NOT auto-select any signal - user must explicitly choose
        # (auto-selecting the first signal would start analyses the user did not choose)

    def _on_antenna_selected(self, event=None):
        """Handler for antenna selection - populates signal list with intersection support."""
        if not self.antex_data_1: return
        ant_key1 = self.antenna_type_var.get()
        if not ant_key1: return
        
        # Update match status since antenna changed
        self._update_antenna_match()

        # Helper to extract signals
        def extract_signals(antex_data, key):
            pco = antex_data.get('pco', {}).get(key, {})
            freqs = set()
            for sys in pco:
                freqs.update(pco[sys].keys())
            return freqs

        signals1 = extract_signals(self.antex_data_1, ant_key1)
        final_signals = signals1
        
        # Check Mode: If Two Files, intersect with File 2
        # (Only if File 2 is loaded and an antenna is selected)
        mode = self.antex_mode.get()
        info_text = "Showing signals from selected antenna."
        
        if mode == 'two_files' and self.antex_data_2:
             ant_key2 = self.antenna_type_var_file2.get()
             if ant_key2:
                 signals2 = extract_signals(self.antex_data_2, ant_key2)
                 intersection = signals1.intersection(signals2)
                 if intersection:
                     final_signals = intersection
                     info_text = "Showing COMMON signals (Intersection of File 1 & 2)."
                 else:
                     info_text = "No common signals found! Showing File 1 signals."
        
        # Update signal info label
        if hasattr(self, 'signal_info_label'):
             self.signal_info_label.config(text=f"{info_text} (G=GPS, R=GLO, E=GAL, C=BDS)")
        
        sorted_signals = sorted(list(final_signals))
        
        # Add IF-LC combinations
        # Check against result set
        res_set = set(sorted_signals)
        
        # GPS
        if 'G01' in res_set and 'G02' in res_set: sorted_signals.append("GPS IF-LC L1/L2")
        if 'G01' in res_set and 'G05' in res_set: sorted_signals.append("GPS IF-LC L1/L5")
        
        # GLONASS (R01/R02/R03)
        if 'R01' in res_set and 'R02' in res_set: sorted_signals.append("GLONASS IF-LC R1/R2")
        # For legacy antex, signals might be named G1/G2 under GLONASS system.
        # But our parser uses R01 etc.
        if 'R01' in res_set and 'R03' in res_set: sorted_signals.append("GLONASS IF-LC G1/G3")

        # Galileo 
        if 'E01' in res_set and 'E05' in res_set: sorted_signals.append("Galileo IF-LC E1/E5a")
        if 'E01' in res_set and 'E07' in res_set: sorted_signals.append("Galileo IF-LC E1/E5b")
        if 'E01' in res_set and 'E08' in res_set: sorted_signals.append("Galileo IF-LC E1/E5")
        if 'E01' in res_set and 'E06' in res_set: sorted_signals.append("Galileo IF-LC E1/E6")

        # Beidou 
        # B1I=C02, B2I=C07, B3I=C06
        if 'C02' in res_set and 'C07' in res_set: sorted_signals.append("Beidou IF-LC B1/B2") 
        if 'C02' in res_set and 'C06' in res_set: sorted_signals.append("Beidou IF-LC B1/B3")
        
        self._update_signal_list(sorted_signals)

    def _run_analysis(self, analysis_func, label, config):
        if not config: return
        
        # Check for antenna type mismatch and warn user
        if self.antex_data_1 and self.antex_data_2 and config.get('analysis_mode') == 'two_files':
            selected_key_file1 = self.antenna_type_var.get()
            selected_key_file2 = self.antenna_type_var_file2.get()
            meta1 = self.antex_data_1.get('metadata', {})
            meta2 = self.antex_data_2.get('metadata', {})
            
            if selected_key_file1 and selected_key_file2:
                if selected_key_file1 in meta1 and selected_key_file2 in meta2:
                    type1 = meta1.get(selected_key_file1, {}).get('type', '').strip()
                    type2 = meta2.get(selected_key_file2, {}).get('type', '').strip()
                    
                    # Compare the SELECTED antenna types, not any type in the file
                    if type1 != type2:
                        # Different antenna types - show warning popup
                        warning_msg = (
                            f"WARNING: Different antenna types detected!\n\n"
                            f"File 1: {type1}\n"
                            f"File 2: {type2}\n\n"
                            f"Comparing different antenna types may produce unexpected results.\n\n"
                            f"Do you want to continue anyway?"
                        )
                        if not messagebox.askyesno("Antenna Type Mismatch", warning_msg, icon='warning'):
                            return  # User cancelled
        
        gui_mode = config['analysis_mode']; val_mode = 'folder' if gui_mode == 'whole_folder' else label
        if label != 'global':
             valid, err = ConfigValidator.validate_config(config, val_mode)
             if not valid: messagebox.showerror("Error", err); return
        def task(cb): return analysis_func(config, cb)
        def complete(res):
            self.title("PCC-Explorer")
            # Figures the analysis wanted to draw are built HERE, on the
            # main thread. Drawing them inside the worker thread left an orphaned
            # Tk figure manager, and the next plt.show() below died with
            # "RuntimeError: main thread is not in main loop".
            self._draw_pending_plots(config)
            if res is None: messagebox.showerror("Error", "No results returned."); return
            if label == 'single':
                # Unpack 7 values (epoch_results is new for LoS per-epoch dropdown)
                r, p, M, az, el, sat_ids, epoch_results = res
                if r is not None:
                    self.analysis_results = {"m_matrix": M, "azimuth": az, "elevation": el, "satellite_ids": sat_ids}

                    colors = self.color_schemes.get(self.color_scheme_var.get(), None)
                    # Set appropriate plot title based on analysis mode
                    if config.get('analysis_mode') == 'single_file':
                        plot_title = "Absolute PCC Impact on Geodetic Parameters"
                    else:
                        plot_title = "ΔPCC Impact on Geodetic Parameters"
                    # Apply display name mapping to param labels
                    display_params = [PARAM_DISPLAY_NAMES.get(pn, pn) for pn in p]
                    plot_results_bar_chart(display_params, r, colors=colors, title=plot_title)
                    # Show notification about saved results
                    results_folder = os.path.join(project_root, 'results')
                    messagebox.showinfo("Analysis Complete", f"Results have been saved to:\n{results_folder}")

                    # --- Epoch dropdown for LoS results ---
                    if epoch_results and len(epoch_results) > 0:
                        self._epoch_results_data = {
                            'combined': r,
                            'param_names': p,
                            'display_params': display_params,
                            'epochs': epoch_results,  # [(dt, vec, n_sats), ...]
                            'colors': colors,
                            'plot_title': plot_title
                        }
                        self._show_epoch_dropdown()
            elif label == 'timeline':
                d, rm, p = res
                if d and rm is not None:
                    # Save results and config file
                    output_dir = os.path.join(project_root, 'results')
                    antenna1 = config.get('file1_antenna', self.antenna_type_var.get() if hasattr(self, 'antenna_type_var') else None)
                    antenna2 = config.get('file2_antenna', self.antenna_type_var_file2.get() if hasattr(self, 'antenna_type_var_file2') else None)
                    if antenna1 and antenna2 and antenna1 != antenna2:
                        antenna_display = f"{antenna1} vs {antenna2}"
                    else:
                        antenna_display = antenna1
                    antenna_name = antenna1
                    config_info = {
                        'antenna1': antenna1,
                        'antenna2': antenna2,
                        'analysis_mode': config.get('analysis_mode', 'two_files'),
                        'signals': config.get('signals', []),
                        'location': f"Lat: {config.get('latitude', 'N/A')}, Lon: {config.get('longitude', 'N/A')}, H: {config.get('height', 'N/A')}"
                    }
                    saved_path = save_timeline_results_to_txt(output_dir, d, rm, p, antenna_name, config_info)
                    
                    # Save config file for time series (matching single analysis behavior)
                    try:
                        from src.data_io import save_results_to_txt
                        configs_dir = os.path.join(project_root, 'configs')
                        os.makedirs(configs_dir, exist_ok=True)
                        safe_name = antenna_name.replace(' ', '_') if antenna_name else 'results'
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        config_filename = f"TimeSeries_{safe_name}_config_{timestamp}.txt"
                        config_filepath = os.path.join(configs_dir, config_filename)
                        with open(config_filepath, 'w') as cf:
                            cf.write("# PCC-Explorer Time Series Configuration\n")
                            cf.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            for key, value in config.items():
                                if isinstance(value, datetime):
                                    cf.write(f"{key}={value.strftime('%Y-%m-%d %H:%M')}\n")
                                elif isinstance(value, (str, int, float, bool, list)):
                                    cf.write(f"{key}={value}\n")
                        print(f"Config saved to: {config_filepath}")
                    except Exception as e:
                        print(f"Warning: Could not save time series config: {e}")
                    
                    if saved_path:
                        messagebox.showinfo("Time Series Results Saved", f"Consolidated results exported to:\n{saved_path}")
                    
                    # Now show the plot
                    colors = self.color_schemes.get(self.color_scheme_var.get(), None)
                    y_axis_mode = self.y_axis_mode_var.get()
                    y_limits = None
                    if y_axis_mode == "Custom":
                        try:
                            y_min = float(self.y_axis_min_var.get())
                            y_max = float(self.y_axis_max_var.get())
                            y_limits = (y_min, y_max)
                        except ValueError:
                            y_limits = None
                    station_coords = None
                    try:
                        lat = float(self.lat_entry.get()) if hasattr(self, 'lat_entry') else None
                        lon = float(self.lon_entry.get()) if hasattr(self, 'lon_entry') else None
                        height = float(self.height_entry.get()) if hasattr(self, 'height_entry') else None
                        station_coords = (lat, lon, height) if lat is not None else None
                    except (ValueError, AttributeError):
                        station_coords = None
                    plot_timeline(d, rm, p, colors=colors, y_axis_mode=y_axis_mode, y_limits=y_limits,
                                  station_coords=station_coords, antenna_name=antenna_display)
            elif label == 'folder':
                # Folder mode: res is (results_dict, param_names)
                results_dict, param_names = res
                if results_dict and param_names:
                    colors = self.color_schemes.get(self.color_scheme_var.get(), None)
                    # Create summary plot - average results across all files
                    import numpy as np_local
                    all_results = [v for v in results_dict.values() if v is not None and not np_local.any(np_local.isnan(v))]
                    if all_results:
                        avg_results = np_local.mean(all_results, axis=0)
                        display_params = [PARAM_DISPLAY_NAMES.get(pn, pn) for pn in param_names]
                        plot_results_bar_chart(display_params, avg_results, colors=colors, 
                                              title=f"Folder Analysis: Average PCC Impact ({len(all_results)} files)")
                        messagebox.showinfo("Folder Analysis Complete", 
                                           f"Processed {len(results_dict)} files.\n"
                                           f"Results with data: {len(all_results)}")
                    else:
                        messagebox.showwarning("Folder Analysis", "No valid results to plot.\nAll files skipped or returned NaN.")
            elif label == 'global':
                # Clear single-analysis results for global analysis
                self.analysis_results = None
                lons, lats, grid, params = res
                if grid is not None:
                    # Get colorbar limits from GUI
                    colorbar_limits = None
                    if self.colorbar_mode_var.get() == "Custom":
                        try:
                            cbar_min = float(self.colorbar_min_var.get())
                            cbar_max = float(self.colorbar_max_var.get())
                            colorbar_limits = (cbar_min, cbar_max)
                        except ValueError:
                            colorbar_limits = None  # Fall back to auto if invalid
                    
                    # Plot ALL components, not just Up
                    for idx, param_name in enumerate(params):
                        display_name = GLOBAL_DISPLAY_NAMES.get(param_name, param_name)
                        plot_world_map(lons, lats, grid[:, :, idx], display_name, config, show_tracks=True, colorbar_limits=colorbar_limits)
                    
                    # Save grid data to .txt files per component
                    try:
                        output_dir = os.path.join(project_root, 'results', 'global_grids')
                        os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Build comprehensive config filename with antenna name
                        antenna_name = config.get('antenna_type', 'unknown').replace(' ', '_')
                        config_filename = f"Global_Grid_{antenna_name}_config_{timestamp}.txt"
                        
                        for idx, param_name in enumerate(params):
                            filename = f"Global_Grid_{param_name}_{timestamp}.txt"
                            filepath = os.path.join(output_dir, filename)
                            with open(filepath, 'w') as f:
                                display_name = GLOBAL_DISPLAY_NAMES.get(param_name, param_name)
                                f.write(f"# Global Analysis Grid: {display_name}\n")
                                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"# Configuration file:  {config_filename}\n")
                                # Calculate step from grid spacing
                                lat_step = abs(lats[1] - lats[0]) if len(lats) > 1 else 0
                                lon_step = abs(lons[1] - lons[0]) if len(lons) > 1 else 0
                                f.write(f"# Lat range: {lats[0]:.1f}:{lat_step:.1f}:{lats[-1]:.1f} (min:step:max)\n")
                                f.write(f"# Lon range: {lons[0]:.1f}:{lon_step:.1f}:{lons[-1]:.1f} (min:step:max)\n")
                                f.write(f"# Grid dimensions: {grid.shape[0]} lat x {grid.shape[1]} lon\n")
                                f.write("#\n# Data (rows=lat, cols=lon):\n")
                                
                                # Write as matrix
                                for lat_idx in range(grid.shape[0]):
                                    row_values = [f"{grid[lat_idx, lon_idx, idx]:.4f}" for lon_idx in range(grid.shape[1])]
                                    f.write(' '.join(row_values) + '\n')
                        
                        # Save config file to configs/ folder (config_filename already built above)
                        configs_dir = os.path.join(project_root, 'configs')
                        os.makedirs(configs_dir, exist_ok=True)
                        config_filepath = os.path.join(configs_dir, config_filename)
                        with open(config_filepath, 'w') as cf:
                            cf.write("# PCC-Explorer Global Analysis Configuration\n")
                            cf.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            for key, value in config.items():
                                # Serialize datetime objects as ISO strings
                                if isinstance(value, datetime):
                                    cf.write(f"{key}={value.strftime('%Y-%m-%d %H:%M')}\n")
                                # Serialize numpy arrays (e.g. position)
                                elif isinstance(value, np.ndarray) and value.ndim == 1 and value.size <= 3:
                                    cf.write(f"{key}={','.join(f'{v:.6f}' for v in value)}\n")
                                # Skip complex objects (orbit data, antex data, etc.)
                                elif isinstance(value, (str, int, float, bool, list)):
                                    cf.write(f"{key}={value}\n")
                        
                        messagebox.showinfo("Global Grids Saved", 
                                          f"Analysis Complete!\n\n"
                                          f"1. Grid Data: Saved {len(params)} component files.\n"
                                          f"2. Configuration: Saved as '{config_filename}'\n\n"
                                          f"Location:\n{output_dir}")
                    except Exception as e:
                        print(f"Warning: Could not save global grids: {e}")
        def error(e): messagebox.showerror("Error", f"Analysis failed:\n{e}"); import traceback; traceback.print_exc()
        def on_cancel():
            self.title("PCC-Explorer")
            # Optional: messagebox.showinfo("Cancelled", "Analysis cancelled by user.")
            
        self.title(f"PCC-Explorer (Processing...)")
        # Map internal labels to user-friendly display names
        display_names = {'timeline': 'Time Series', 'single': 'Single', 'global': 'Global', 'folder': 'Folder'}
        display_label = display_names.get(label, label.title())
        run_with_progress(self, task, complete, error, on_cancel=on_cancel, dialog_title=f"Running {display_label} Analysis...", can_cancel=True)

    def run_single_analysis(self):
        # Redirect Area-based mode to Global Analysis
        if self.pos_type.get() == "Area":
            messagebox.showwarning("Position Mode", "Area-based calculation is selected.\n\nPlease use 'Run Global/Regional Analysis' button for grid-based analysis,\nor select a different position type (Ellipsoidal, ECEF, or Map).")
            return
        cfg = self._build_config_from_gui('single')
        if cfg and cfg.get('analysis_mode') == 'whole_folder': self._run_analysis(run_folder_pipeline, 'folder', cfg)
        elif cfg: self._run_analysis(run_processing_pipeline, 'single', cfg)

    def run_timeline_analysis(self):
        # Block Time Series for Area-based mode
        if self.pos_type.get() == "Area":
            messagebox.showerror("Not Supported",
                "Time Series analysis is not available for Area-based (Global/Regional) calculations.\n\n"
                "Please select a specific position (Ellipsoidal, ECEF, or Map) "
                "or use 'Run Global/Regional Analysis' instead.")
            return
        cfg = self._build_config_from_gui('timeline')
        if cfg and not self._confirm_timeline_has_a_series(cfg):
            return
        self._run_analysis(run_timeline_pipeline, 'timeline', cfg)

    def _timeline_period_count(self, cfg):
        """
        How many points a time series with this configuration would produce.

        The same arithmetic as run_timeline_pipeline, so the warning cannot say
        one thing while the run does another.
        """
        try:
            start_date = cfg['start_date'].date()
            end_date = cfg['end_date'].date()
            interval_minutes = cfg.get('interval_min', 1440) or 1440
            if interval_minutes < 1440:
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date + timedelta(days=1),
                                          datetime.min.time())
                return int((end_dt - start_dt).total_seconds() / (interval_minutes * 60))
            return (end_date - start_date).days + 1
        except Exception:
            return None

    def _confirm_timeline_has_a_series(self, cfg):
        """
        Warn before a time series that cannot show anything changing over time.

        A time series can end up showing a single point. There are two
        different reasons for that, and only one of them is something the
        user can put right.

        In GRID mode it is the defaults: Start date and End date both default to
        yesterday and the interval to 24:00, so one day at one point per day is
        exactly one point. The fields are editable, so saying which three settings
        are responsible is a useful answer.

        In LINE-OF-SIGHT mode those same fields are disabled and the panel says
        "not used in LoS mode", because the LoS path reads the whole observation
        file whatever the period: parse_satellite_observations() takes a path and
        no date range, and the only date-dependent input left is the gridded VMF
        troposphere coefficient. So every period repeats the same satellite
        geometry, and the grid-mode advice would point at fields this mode
        forbids. The per-epoch view in LoS mode is Run Single Analysis together
        with the Epoch Selection box, so say that instead.
        """
        periods = self._timeline_period_count(cfg)

        if cfg.get('computation_mode') == 'los':
            count = ""
            if periods is not None:
                count = ("These settings produce %d point%s.\n\n"
                         % (periods, "" if periods == 1 else "s"))
            return messagebox.askyesno(
                "Time series in Line-of-Sight mode",
                "In Line-of-Sight mode a time series repeats the same analysis "
                "for every period.\n\n"
                "The observations are read from the whole file, and the date and "
                "interval fields are disabled in this mode, so every point of "
                "the plot uses the same satellite geometry. Only the troposphere "
                "model can differ from one period to the next, and only when "
                "gridded VMF is switched on.\n\n"
                + count +
                "For results that change over time, use Run Single Analysis in "
                "Line-of-Sight mode and step through the epochs with the Epoch "
                "Selection box.\n\n"
                "Run anyway?")

        if periods is None or periods >= 2:
            return True

        interval = self.interval_var.get() if hasattr(self, 'interval_var') else "24:00"
        return messagebox.askyesno(
            "Time series with a single point",
            f"These settings produce {periods} point, so every panel of the plot "
            "will show one dot rather than a series.\n\n"
            f"Start date: {cfg['start_date'].date()}\n"
            f"End date:   {cfg['end_date'].date()}\n"
            f"Interval:   {interval}\n\n"
            "For a series over time, either set a later End date, or set a "
            "sub-daily interval such as 01:00 to get one point per hour of that "
            "day.\n\n"
            "Run anyway?")

    def run_global_analysis(self):
        cfg = self._build_config_from_gui('global')
        if cfg: self._run_analysis(run_global_pipeline, 'global', cfg)

    def show_skyplot(self):
        if self.analysis_results and "azimuth" in self.analysis_results:
            # Close previous skyplot windows before opening new one
            plt.close('all')
            # Get coordinates for title
            lat, lon, height = None, None, None
            try:
                pos_mode = self.pos_type.get()
                if pos_mode in ("Ellipsoidal", "Map"):
                    lat = float(self.coord1_var.get())
                    lon = float(self.coord2_var.get())
                    height = float(self.coord3_var.get())
                elif pos_mode == "ECEF":
                    # Convert XYZ to ellipsoidal for display
                    x = float(self.coord1_var.get())
                    y = float(self.coord2_var.get())
                    z = float(self.coord3_var.get())
                    lat_rad, lon_rad, height = ecef_to_ell(x, y, z)
                    lat = np.rad2deg(lat_rad)
                    lon = np.rad2deg(lon_rad)
            except (ValueError, TypeError):
                pass  # Coordinates not available
            sat_ids = self.analysis_results.get("satellite_ids", None)
            plot_skyplot(self.analysis_results["azimuth"], self.analysis_results["elevation"], 
                        masks=self.azimuth_masks_list, latitude=lat, longitude=lon, height=height,
                        satellite_ids=sat_ids)
        else:
            # Context-aware error messages for skyplot
            if self.pos_type.get() == "Area":
                messagebox.showinfo("Sky Plot Not Available",
                    "Sky Plot is not available for Global/Regional analysis.\n\n"
                    "Please run a Single Analysis for a specific location first.")
            else:
                messagebox.showinfo("No Data",
                    "No satellite data available.\n\n"
                    "Please run a Single Analysis first to generate the Sky Plot.")

    def _launch_map_selector(self):
        try:
            from matplotlib.widgets import Cursor
            win = tk.Toplevel(self); win.title("Select Position"); win.geometry("800x600")
            fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.stock_img(); ax.coastlines()
            
            # Add crosshair cursor
            cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
            win.cursor_keepalive = cursor # Keep reference
            
            canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            def onclick(event):
                if event.inaxes == ax:
                    lat_val = event.ydata
                    lon_val = event.xdata
                    self.coord1_var.set(f"{lat_val:.4f}")
                    self.coord2_var.set(f"{lon_val:.4f}")
                    self.coord3_var.set("0.0")
                    win.destroy()
                    # Fields remain editable so user can adjust lat, lon, and height
            fig.canvas.mpl_connect('button_press_event', onclick)
        except Exception as e: messagebox.showerror("Map Error", f"{e}"); self.pos_type.set("Ellipsoidal"); self.update_position_mode()

    def _update_antenna_selector(self, file_num, filename):
        if not filename or not os.path.exists(filename): return
        try:
            antex_data = read_antex_file(filename)
            if file_num == 1: self.antex_data_1 = antex_data
            elif file_num == 2: self.antex_data_2 = antex_data
            if file_num == 1 and self.antex_data_1:
                file1_keys = sorted(self.antex_data_1['metadata'].keys())
                self.antenna_combobox['values'] = file1_keys
                if self.antenna_type_var.get() not in file1_keys:
                    self.antenna_type_var.set(file1_keys[0] if file1_keys else "")
                self.antenna_combobox.config(state="readonly")
                self._on_antenna_selected()
        except Exception: pass

    def _show_epoch_dropdown(self):
        """Show epoch selection dropdown for LoS per-epoch results."""
        if not hasattr(self, '_epoch_results_data') or not self._epoch_results_data:
            return

        epoch_data = self._epoch_results_data
        epochs = epoch_data['epochs']  # [(dt, vec, n_sats), ...]

        # Build dropdown options
        options = ["All Epochs (Combined)"]
        first_dt = epochs[0][0]
        last_dt = epochs[-1][0]
        n_first = epochs[0][2]
        n_last = epochs[-1][2]
        options.append(f"First Epoch: {first_dt.strftime('%Y-%m-%d %H:%M:%S')} ({n_first} sats)")
        options.append(f"Last Epoch: {last_dt.strftime('%Y-%m-%d %H:%M:%S')} ({n_last} sats)")

        # Create a small popup window
        win = tk.Toplevel(self)
        win.title("Epoch Selection")
        win.geometry("480x180")
        win.resizable(False, False)

        frame = ttk.Frame(win, padding=15)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text="LoS Per-Epoch Results",
                  font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=(0, 5))
        # Say which run produced this box and what "Combined" means. Without
        # it, the box gives no clue which button raised it, and invites the
        # assumption that the single analysis shows the mean over all epochs.
        # It does not: it is one adjustment over all observations.
        ttk.Label(frame,
                  text=f"From Run Single Analysis in Line-of-Sight mode: "
                       f"{len(epochs)} epoch solutions available.",
                  font=('Helvetica', 9), wraplength=440,
                  justify='left').pack(anchor='w')
        ttk.Label(frame,
                  text="All Epochs (Combined) is one adjustment over the "
                       "observations of every epoch together, not the mean of "
                       "the single-epoch solutions.",
                  font=('Helvetica', 8), foreground='gray', wraplength=440,
                  justify='left').pack(anchor='w', pady=(2, 0))

        combo_var = tk.StringVar(value=options[0])
        combo = ttk.Combobox(frame, textvariable=combo_var, values=options,
                             state='readonly', width=60)
        combo.pack(pady=10, fill='x')

        def on_select(event=None):
            self._on_epoch_selected(combo_var.get())

        combo.bind('<<ComboboxSelected>>', on_select)

        # Show LoS Timeline button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Show LoS Timeline",
                   command=self._plot_los_timeline_from_data).pack(side='left')
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side='right')

        # Size to the content, with 480x180 as the minimum: the window is not
        # resizable, so a fixed size would push the buttons below the edge.
        win.update_idletasks()
        win.geometry("%dx%d" % (max(480, win.winfo_reqwidth()),
                                max(180, win.winfo_reqheight())))

    def _on_epoch_selected(self, selection: str):
        """Handle epoch dropdown selection change."""
        if not hasattr(self, '_epoch_results_data') or not self._epoch_results_data:
            return

        data = self._epoch_results_data
        epochs = data['epochs']

        if selection.startswith("All Epochs"):
            results = data['combined']
            title = data['plot_title']
        elif selection.startswith("First Epoch"):
            results = epochs[0][1]
            dt = epochs[0][0]
            title = f"{data['plot_title']}, First Epoch ({dt.strftime('%H:%M:%S')})"
        elif selection.startswith("Last Epoch"):
            results = epochs[-1][1]
            dt = epochs[-1][0]
            title = f"{data['plot_title']}, Last Epoch ({dt.strftime('%H:%M:%S')})"
        else:
            return

        from src.plotting import plot_results_bar_chart
        plot_results_bar_chart(data['display_params'], results,
                              colors=data['colors'], title=title)

    def _plot_los_timeline_from_data(self):
        """Plot LoS per-epoch results as a timeline."""
        if not hasattr(self, '_epoch_results_data') or not self._epoch_results_data:
            messagebox.showinfo("No Data", "No per-epoch results available.")
            return

        data = self._epoch_results_data
        epochs = data['epochs']  # [(dt, vec, n_sats), ...]

        from src.plotting import plot_los_timeline
        colors = data['colors']
        plot_los_timeline(epochs, data['param_names'], colors=colors)


if __name__ == "__main__":
    # --- HYBRID MODE CHECK ---
    # If the user passed arguments (like a config file), run CLI mode.
    # sys.argv[0] is the program name. sys.argv[1] would be the config file.
    if len(sys.argv) > 1:
        # Run in Command Line / Batch Mode
        try:
            run_cli_mode()
        except SystemExit:
            pass # Clean exit handled by argparse
        except Exception as e:
            print(f"CLI Error: {e}")
    else:
        # No arguments provided -> Run GUI Mode (Normal)
        if 'win' in sys.platform:
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except: pass
        app = App()
        # Only in GUI mode: a CLI run must never wait for a dialog.
        if mailing_list is not None:
            mailing_list.maybe_show(app, "PCC-Explorer")
        app.mainloop()