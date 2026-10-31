#!/usr/bin/env python3
"""
visualizer_app_v3_enhanced.py
STM32 STFT Live Visualizer & Multi-Mode Datalogger v3 (PyQt5/PyQt6 + pyqtgraph)
Includes dynamic theming and adjustable font sizes.
Run:
    python visualizer_app_v3_enhanced.py
"""

import datetime
import math
import os
import sys
from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from serial.tools import list_ports

from stft_backend_v3 import (
    DEFAULT_STOP_HEX,
    DEFAULT_SYNC_HEX,
    HAS_OPENPYXL,
    MODE_NAMES,
    PRECISION_NAMES,
    Frame,
    ProfilingData,
    SerialReader,
    StreamingCSVLogger,
    parse_hex_marker,
    to_float,
)

# Qt5 / Qt6 compatibility constants
try:
    DASH_LINE = QtCore.Qt.PenStyle.DashLine
    LEFT_DOCK = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
except AttributeError:
    DASH_LINE = QtCore.Qt.DashLine
    LEFT_DOCK = QtCore.Qt.LeftDockWidgetArea

FILL_DB = -140.0

DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #121417; color: #E2E8F0; font-family: 'Segoe UI', 'Inter', sans-serif; }
QTabWidget::pane { border: 1px solid #2D3748; background-color: #181B20; }
QTabBar::tab { background-color: #1A202C; color: #A0AEC0; padding: 8px 18px; border: 1px solid #2D3748; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: 600; }
QTabBar::tab:selected { background-color: #2B6CB0; color: #FFFFFF; }
QGroupBox { border: 1px solid #2D3748; border-radius: 5px; margin-top: 10px; padding-top: 10px; background-color: #181B20; font-weight: bold; color: #63B3ED; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #2D3748; color: #EDF2F7; border: 1px solid #4A5568; border-radius: 4px; padding: 6px 10px; font-weight: 600; }
QPushButton:hover { background-color: #3B475E; border-color: #63B3ED; }
QPushButton:checked { background-color: #2B6CB0; border-color: #63B3ED; color: #FFFFFF; }
QPushButton#stopPlotBtn:checked { background-color: #C53030; border-color: #FC8181; }
QPushButton#csvBtn:checked { background-color: #2F855A; border-color: #68D391; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #0F1115; color: #EDF2F7; border: 1px solid #4A5568; border-radius: 4px; padding: 4px 6px; }
QStatusBar { background-color: #0F1115; color: #A0AEC0; border-top: 1px solid #2D3748; }
"""

LIGHT_STYLESHEET = """
QMainWindow, QWidget { background-color: #F8FAFC; color: #1A202C; font-family: 'Segoe UI', 'Inter', sans-serif; }
QTabWidget::pane { border: 1px solid #E2E8F0; background-color: #FFFFFF; }
QTabBar::tab { background-color: #E2E8F0; color: #4A5568; padding: 8px 18px; border: 1px solid #CBD5E0; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: 600; }
QTabBar::tab:selected { background-color: #63B3ED; color: #FFFFFF; }
QGroupBox { border: 1px solid #E2E8F0; border-radius: 5px; margin-top: 10px; padding-top: 10px; background-color: #FFFFFF; font-weight: bold; color: #2B6CB0; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #E2E8F0; color: #2D3748; border: 1px solid #CBD5E0; border-radius: 4px; padding: 6px 10px; font-weight: 600; }
QPushButton:hover { background-color: #D6E4EF; border-color: #63B3ED; }
QPushButton:checked { background-color: #63B3ED; border-color: #2B6CB0; color: #FFFFFF; }
QPushButton#stopPlotBtn:checked { background-color: #E53E3E; border-color: #FEB2B2; }
QPushButton#csvBtn:checked { background-color: #48BB78; border-color: #9AE6B4; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #FFFFFF; color: #1A202C; border: 1px solid #CBD5E0; border-radius: 4px; padding: 4px 6px; }
QStatusBar { background-color: #FFFFFF; color: #4A5568; border-top: 1px solid #E2E8F0; }
"""

class STFTVisualizerV3(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STM32 STFT Datalogger & Visualizer v3")
        self.resize(1380, 860)
        self.settings = QtCore.QSettings("stm32-stft-v3", "visualizer")

        self.reader: Optional[SerialReader] = None
        self.logger = StreamingCSVLogger()
        self.plotting_stopped = False
        self.dirty = False

        self.num_bins = 257
        self.spec = None
        self.filled = 0
        self.last_db = None
        self.raw_adc_buf = np.zeros(2048, dtype=np.float32)
        self.has_raw_adc = False

        self.chunks = 0
        self.total_slices = 0
        self.skipped = 0
        self.active_precision = 0
        self.active_mode = 0
        self.total_exec_time_us = 0.0
        self.last_chunk_time_us = 0.0

        self.prof_slices = 0
        self.prof_chunks = 0
        self.total_mult_us = 0.0
        self.total_rfft_us = 0.0
        self.total_mag_us = 0.0

        self.snapshot_slices = []
        self.snapshot_raw = []
        self.snapshot_count = 0
        self.max_snapshot_slices = 8000

        self.lo, self.hi = -40.0, 100.0
        self.sim_phase = 0.0
        self.rng = np.random.default_rng()

        self._build_ui()
        self._load_settings()
        self._reset_buffers(self.num_bins)
        self._connect_signals()
        self.refresh_ports()
        
        # Apply initial theme and formatting
        self._apply_theme()
        self._on_colormap(self.cmap_combo.currentText())
        self._on_mode_changed(self.mode_combo.currentIndex())
        self._set_connected(False)

        self.sim_timer = QtCore.QTimer(self, interval=200)
        self.sim_timer.timeout.connect(self._sim_tick)
        self.ui_timer = QtCore.QTimer(self, interval=33)
        self.ui_timer.timeout.connect(self._refresh_plots)
        self.ui_timer.start()

    def _spin(self, lo, hi, val, suffix="", decimals=None, step=None):
        w = QtWidgets.QDoubleSpinBox() if decimals is not None else QtWidgets.QSpinBox()
        if decimals is not None:
            w.setDecimals(decimals)
        w.setRange(lo, hi)
        w.setValue(val)
        if suffix:
            w.setSuffix(suffix)
        if step:
            w.setSingleStep(step)
        w.setKeyboardTracking(False)
        return w

    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # TAB 1: VISUALIZER
        vis_tab = QtWidgets.QWidget()
        vis_lay = QtWidgets.QVBoxLayout(vis_tab)
        vis_lay.setContentsMargins(6, 6, 6, 6)

        self.summary_group = QtWidgets.QGroupBox("Execution Time & Stream Summary")
        sum_lay = QtWidgets.QHBoxLayout(self.summary_group)
        self.lbl_avg_slice = QtWidgets.QLabel("Avg Time / Slice: 0.00 µs")
        self.lbl_avg_chunk = QtWidgets.QLabel("Avg Library / Chunk: 0.00 µs")
        self.lbl_prec = QtWidgets.QLabel("Precision: F32 (float32)")
        self.lbl_slices = QtWidgets.QLabel("Captured: 0 slices (0 chunks)")
        self.lbl_csv = QtWidgets.QLabel("CSV: Idle")

        sum_lay.addWidget(self.lbl_avg_slice)
        sum_lay.addStretch(1)
        sum_lay.addWidget(self.lbl_avg_chunk)
        sum_lay.addStretch(1)
        sum_lay.addWidget(self.lbl_prec)
        sum_lay.addStretch(1)
        sum_lay.addWidget(self.lbl_slices)
        sum_lay.addStretch(1)
        sum_lay.addWidget(self.lbl_csv)
        vis_lay.addWidget(self.summary_group)

        self.prof_group = QtWidgets.QGroupBox("Mode 2: CMSIS-DSP Average Execution Time Breakdown")
        prof_lay = QtWidgets.QHBoxLayout(self.prof_group)
        self.lbl_prof_mult = QtWidgets.QLabel("arm_mult: 0.00 µs/slice")
        self.lbl_prof_rfft = QtWidgets.QLabel("arm_rfft: 0.00 µs/slice")
        self.lbl_prof_mag = QtWidgets.QLabel("arm_cmplx_mag: 0.00 µs/slice")
        self.lbl_prof_lib = QtWidgets.QLabel("Library Overhead: 0.00 µs/slice")

        for w in (self.lbl_prof_mult, self.lbl_prof_rfft, self.lbl_prof_mag, self.lbl_prof_lib):
            prof_lay.addWidget(w)
            prof_lay.addStretch(1)
        vis_lay.addWidget(self.prof_group)

        # Base pyqtgraph config
        pg.setConfigOption("background", "#0F1115")
        pg.setConfigOption("foreground", "#CBD5E0")
        self.glw = pg.GraphicsLayoutWidget()
        vis_lay.addWidget(self.glw, 1)

        self.spec_plot = self.glw.addPlot(row=0, col=0, title="STFT Spectrogram")
        self.img = pg.ImageItem()
        self.spec_plot.addItem(self.img)
        self.bar = pg.ColorBarItem(values=(self.lo, self.hi), interactive=False, colorMap=pg.colormap.get("viridis"), label="Magnitude (dB)")
        self.bar.setImageItem(self.img, insert_in=self.spec_plot)

        self.mag_plot = self.glw.addPlot(row=1, col=0, colspan=2, title="Magnitude (dB) — Latest Slice")
        self.mag_curve = self.mag_plot.plot(pen=pg.mkPen("#63B3ED", width=2))
        self.peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#F6AD55", style=DASH_LINE, width=1.5))
        self.mag_plot.addItem(self.peak_line)

        self.raw_plot = self.glw.addPlot(row=2, col=0, colspan=2, title="Raw ADC Signal (Time Domain)")
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen("#68D391", width=1.5))

        for p in (self.spec_plot, self.mag_plot, self.raw_plot):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.getAxis("bottom").setLabel("Frequency (Hz)" if p is self.mag_plot else "Time (s)")

        self.spec_plot.getAxis("left").setLabel("Frequency (Hz)")
        self.mag_plot.getAxis("left").setLabel("Magnitude (dB)")
        self.raw_plot.getAxis("left").setLabel("ADC Code")
        self.tabs.addTab(vis_tab, "Visualizer")

        # TAB 2: SETTINGS
        set_tab = QtWidgets.QWidget()
        set_lay = QtWidgets.QVBoxLayout(set_tab)
        set_lay.setContentsMargins(20, 20, 20, 20)

        g_csv = QtWidgets.QGroupBox("CSV & Multi-Sheet Export Settings")
        f_csv = QtWidgets.QFormLayout(g_csv)
        dir_row = QtWidgets.QHBoxLayout()
        self.csv_dir_edit = QtWidgets.QLineEdit(os.path.abspath("./logs"))
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        dir_row.addWidget(self.csv_dir_edit, 1)
        dir_row.addWidget(self.browse_btn)

        self.csv_name_edit = QtWidgets.QLineEdit("stft_session")
        self.auto_csv_chk = QtWidgets.QCheckBox("Automatically start writing to CSV on Connect / Simulation")
        self.auto_csv_chk.setChecked(True)
        self.xlsx_chk = QtWidgets.QCheckBox("Generate combined multi-sheet Excel workbook (.xlsx)")
        self.xlsx_chk.setChecked(HAS_OPENPYXL)
        self.xlsx_chk.setEnabled(HAS_OPENPYXL)
        f_csv.addRow("CSV Folder Path:", dir_row)
        f_csv.addRow("CSV Base Filename:", self.csv_name_edit)
        f_csv.addRow("", self.auto_csv_chk)
        f_csv.addRow("", self.xlsx_chk)
        set_lay.addWidget(g_csv)

        g_proto = QtWidgets.QGroupBox("Customizable Start & Stop Bytes")
        f_proto = QtWidgets.QFormLayout(g_proto)
        self.sync_edit = QtWidgets.QLineEdit(DEFAULT_SYNC_HEX)
        self.stop_edit = QtWidgets.QLineEdit(DEFAULT_STOP_HEX)
        self.test_proto_btn = QtWidgets.QPushButton("Validate Start / Stop Hex")
        f_proto.addRow("Start Sync Word (16-bit Hex):", self.sync_edit)
        f_proto.addRow("Stop Byte Marker (32-bit Hex):", self.stop_edit)
        f_proto.addRow("", self.test_proto_btn)
        set_lay.addWidget(g_proto)

        # Theme & Formatting Settings
        g_theme = QtWidgets.QGroupBox("Theme & Font Settings")
        f_theme = QtWidgets.QFormLayout(g_theme)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.font_spin = self._spin(8, 24, 10, " pt")
        f_theme.addRow("Application Theme:", self.theme_combo)
        f_theme.addRow("Base Font Size:", self.font_spin)
        set_lay.addWidget(g_theme)

        g_meta = QtWidgets.QGroupBox("STFT Configuration Metadata (Written to Metadata Sheet)")
        f_meta = QtWidgets.QFormLayout(g_meta)
        self.win_len_spin = self._spin(16, 4096, 256, " samples")
        self.overlap_spin = self._spin(0, 99, 70, " %")
        self.win_type_combo = QtWidgets.QComboBox()
        self.win_type_combo.addItems(["HANNING", "HAMMING", "BLACKMAN", "BARTLETT", "NO_WINDOW"])
        f_meta.addRow("Window Length (WIN_LEN):", self.win_len_spin)
        f_meta.addRow("Overlap Percentage:", self.overlap_spin)
        f_meta.addRow("Window Function:", self.win_type_combo)
        set_lay.addWidget(g_meta)
        set_lay.addStretch(1)

        self.tabs.addTab(set_tab, "Settings")

        # LEFT DOCK
        side = QtWidgets.QWidget()
        s_lay = QtWidgets.QVBoxLayout(side)

        g_mode = QtWidgets.QGroupBox("Mode & Precision")
        f_mode = QtWidgets.QFormLayout(g_mode)
        self.mode_combo = QtWidgets.QComboBox()
        for i in range(4): self.mode_combo.addItem(MODE_NAMES[i], i)
        self.prec_combo = QtWidgets.QComboBox()
        self.prec_combo.addItems(["Auto (From Packet Header)", "F32 (float32)", "Q15 (int16)", "Q31 (int32)"])
        f_mode.addRow("Mode:", self.mode_combo)
        f_mode.addRow("Precision:", self.prec_combo)
        s_lay.addWidget(g_mode)

        g_ser = QtWidgets.QGroupBox("Serial Connection")
        f_ser = QtWidgets.QFormLayout(g_ser)
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh Ports")
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["115200", "230400", "460800", "921600", "2000000"])
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.sim_btn = QtWidgets.QPushButton("Simulate STM32 Stream")
        self.sim_btn.setCheckable(True)
        f_ser.addRow("Port:", self.port_combo)
        f_ser.addRow(self.refresh_btn)
        f_ser.addRow("Baud:", self.baud_combo)
        f_ser.addRow(self.connect_btn)
        f_ser.addRow(self.sim_btn)
        s_lay.addWidget(g_ser)

        g_sig = QtWidgets.QGroupBox("Signal Scaling")
        f_sig = QtWidgets.QFormLayout(g_sig)
        self.fs_spin = self._spin(1, 50e6, 900, " Hz", decimals=0, step=100)
        self.hop_spin = self._spin(1, 8192, 77, " samples")
        self.chunk_spin = self._spin(1, 65536, 512, " samples")
        f_sig.addRow("Sample Rate (Fs):", self.fs_spin)
        f_sig.addRow("Hop Length:", self.hop_spin)
        f_sig.addRow("Buffer Len:", self.chunk_spin)
        s_lay.addWidget(g_sig)

        g_plt = QtWidgets.QGroupBox("Plot Switches & Display")
        f_plt = QtWidgets.QFormLayout(g_plt)
        self.show_mag_chk = QtWidgets.QCheckBox("Show Magnitude (dB) Plot")
        self.show_mag_chk.setChecked(True)
        self.show_raw_chk = QtWidgets.QCheckBox("Show Raw ADC Signal Plot")
        self.show_raw_chk.setChecked(True)
        self.cmap_combo = QtWidgets.QComboBox()
        for name in ("viridis", "plasma", "inferno", "magma", "cividis"): self.cmap_combo.addItem(name)
        self.hist_spin = self._spin(10, 5000, 120, " slices")
        self.auto_lvl_chk = QtWidgets.QCheckBox("Auto dB Color Scale")
        self.auto_lvl_chk.setChecked(True)
        f_plt.addRow(self.show_mag_chk)
        f_plt.addRow(self.show_raw_chk)
        f_plt.addRow("Colormap:", self.cmap_combo)
        f_plt.addRow("History:", self.hist_spin)
        f_plt.addRow(self.auto_lvl_chk)
        s_lay.addWidget(g_plt)

        g_ctrl = QtWidgets.QGroupBox("Streaming & CSV Controls")
        v_ctrl = QtWidgets.QVBoxLayout(g_ctrl)
        self.stop_plot_btn = QtWidgets.QPushButton("Stop Plotting")
        self.stop_plot_btn.setObjectName("stopPlotBtn")
        self.stop_plot_btn.setCheckable(True)
        self.csv_btn = QtWidgets.QPushButton("Start Saving to CSV")
        self.csv_btn.setObjectName("csvBtn")
        self.csv_btn.setCheckable(True)
        self.export_snap_btn = QtWidgets.QPushButton("Export Snapshot CSV…")
        self.clear_btn = QtWidgets.QPushButton("Clear Plots & Averages")

        for b in (self.stop_plot_btn, self.csv_btn, self.export_snap_btn, self.clear_btn):
            v_ctrl.addWidget(b)
        s_lay.addWidget(g_ctrl)
        s_lay.addStretch(1)

        self.dock = QtWidgets.QDockWidget("Controls", self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        self.dock.setWidget(scroll)
        self.addDockWidget(LEFT_DOCK, self.dock)

        menubar = self.menuBar()
        view_menu = menubar.addMenu("View")
        view_menu.addAction(self.dock.toggleViewAction())
        self.conn_lbl = QtWidgets.QLabel("Disconnected")
        self.hover_lbl = QtWidgets.QLabel("Hover over spectrogram for readout")
        self.statusBar().addWidget(self.conn_lbl, 1)
        self.statusBar().addPermanentWidget(self.hover_lbl)

    def _connect_signals(self):
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.toggle_connect)
        self.sim_btn.toggled.connect(self._on_sim_toggled)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.prec_combo.currentIndexChanged.connect(self._update_summary_labels)

        for w in (self.fs_spin, self.hop_spin, self.chunk_spin, self.hist_spin):
            w.valueChanged.connect(self._update_geometry)

        self.show_mag_chk.toggled.connect(self._update_plot_visibility)
        self.show_raw_chk.toggled.connect(self._update_plot_visibility)
        self.cmap_combo.currentTextChanged.connect(self._on_colormap)

        # Theme & Formatting signals
        self.theme_combo.currentIndexChanged.connect(self._apply_theme)
        self.font_spin.valueChanged.connect(self._apply_theme)

        self.stop_plot_btn.toggled.connect(self._on_stop_plotting_toggled)
        self.csv_btn.toggled.connect(self._on_csv_toggled)
        self.export_snap_btn.clicked.connect(self.export_snapshot_csv)
        self.clear_btn.clicked.connect(self.clear_all)

        self.browse_btn.clicked.connect(self._browse_folder)
        self.test_proto_btn.clicked.connect(self._validate_markers_ui)

        self._proxy = pg.SignalProxy(self.spec_plot.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_hover)

    def _apply_theme(self, *_):
        is_dark = self.theme_combo.currentIndex() == 0
        base_sheet = DARK_STYLESHEET if is_dark else LIGHT_STYLESHEET
        font_size = int(self.font_spin.value())
        
        # Apply global font size override
        font_rule = f"\nQWidget {{ font-size: {font_size}pt; }}"
        
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyleSheet(base_sheet + font_rule)

        # Dynamic pyqtgraph recoloring
        bg_color = "#0F1115" if is_dark else "#FFFFFF"
        fg_color = "#CBD5E0" if is_dark else "#1A202C"
        grid_alpha = 0.25 if is_dark else 0.15
        
        pg.setConfigOption("background", bg_color)
        pg.setConfigOption("foreground", fg_color)
        
        self.glw.setBackground(bg_color)
        for p in (self.spec_plot, self.mag_plot, self.raw_plot):
            p.getAxis('bottom').setPen(fg_color)
            p.getAxis('left').setPen(fg_color)
            p.getAxis('bottom').setTextPen(fg_color)
            p.getAxis('left').setTextPen(fg_color)
            p.showGrid(x=True, y=True, alpha=grid_alpha)
            
        self.bar.getAxis('right').setPen(fg_color)
        self.bar.getAxis('right').setTextPen(fg_color)
        
        # Update summary text colors for readability in Light Mode
        if is_dark:
            self.lbl_avg_slice.setStyleSheet("color: #68D391; font-weight: bold;")
            for w in (self.lbl_avg_chunk, self.lbl_prec, self.lbl_slices, self.lbl_csv):
                w.setStyleSheet("color: #90CDF4; font-weight: bold;")
            for w in (self.lbl_prof_mult, self.lbl_prof_rfft, self.lbl_prof_mag, self.lbl_prof_lib):
                w.setStyleSheet("color: #F6E05E; font-weight: bold;")
        else:
            self.lbl_avg_slice.setStyleSheet("color: #2F855A; font-weight: bold;")
            for w in (self.lbl_avg_chunk, self.lbl_prec, self.lbl_slices, self.lbl_csv):
                w.setStyleSheet("color: #2B6CB0; font-weight: bold;")
            for w in (self.lbl_prof_mult, self.lbl_prof_rfft, self.lbl_prof_mag, self.lbl_prof_lib):
                w.setStyleSheet("color: #B7791F; font-weight: bold;")

    def _on_mode_changed(self, mode_idx: int):
        self.active_mode = mode_idx
        is_prod = (mode_idx == 3)
        if is_prod and self.csv_btn.isChecked(): self.csv_btn.setChecked(False)
        self.csv_btn.setEnabled(not is_prod)
        self.export_snap_btn.setEnabled(not is_prod)
        self.summary_group.setVisible(not is_prod)
        self.prof_group.setVisible(mode_idx == 2)
        self._update_plot_visibility()

    def _update_plot_visibility(self):
        is_prod = (self.active_mode == 3)
        show_mag = self.show_mag_chk.isChecked() and not is_prod
        show_raw = (self.active_mode == 1) and self.show_raw_chk.isChecked() and not is_prod
        self.mag_plot.setVisible(show_mag)
        self.raw_plot.setVisible(show_raw)
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1 if show_mag else 0)
        self.glw.ci.layout.setRowStretchFactor(2, 1 if show_raw else 0)

    def _browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select CSV Folder", self.csv_dir_edit.text())
        if folder: self.csv_dir_edit.setText(folder)

    def _get_markers(self):
        return parse_hex_marker(self.sync_edit.text(), 2), parse_hex_marker(self.stop_edit.text(), 4)

    def _validate_markers_ui(self):
        try:
            sync_b, stop_b = self._get_markers()
            QtWidgets.QMessageBox.information(self, "Valid Protocol Bytes", f"Start Sync: {sync_b.hex().upper()}\nStop Byte: {stop_b.hex().upper()}")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Hex", str(exc))

    def _effective_precision(self, packet_prec: int) -> int:
        ui_idx = self.prec_combo.currentIndex()
        return packet_prec if ui_idx == 0 else (ui_idx - 1)

    def _build_metadata_rows(self) -> list:
        eff_prec = self._effective_precision(self.active_precision)
        fft_len = (self.num_bins - 1) * 2
        avg_slice_us = (self.total_exec_time_us / self.total_slices) if self.total_slices > 0 else 0.0
        avg_chunk_us = (self.total_exec_time_us / self.chunks) if self.chunks > 0 else 0.0

        rows = [
            ["Timestamp", datetime.datetime.now().isoformat(timespec="seconds"), ""],
            ["Operating_Mode", MODE_NAMES.get(self.active_mode, "Unknown"), f"mode_id={self.active_mode}"],
            ["Data_Precision", PRECISION_NAMES.get(eff_prec, "Unknown"), f"precision_id={eff_prec}"],
            ["Sample_Rate_Fs", f"{self.fs_spin.value():.1f}", "Hz"],
            ["FFT_Length", str(fft_len), f"{self.num_bins} bins"],
            ["Window_Length", str(self.win_len_spin.value()), "samples"],
            ["Hop_Length", str(self.hop_spin.value()), "samples"],
            ["Overlap_Percent", str(self.overlap_spin.value()), "%"],
            ["Window_Type", self.win_type_combo.currentText(), ""],
            ["Buffer_Length_Per_Call", str(self.chunk_spin.value()), "samples"],
            ["Total_Chunks_Processed", str(self.chunks), "chunks"],
            ["Total_Slices_Processed", str(self.total_slices), "slices"],
            ["Avg_Execution_Time_Per_Slice", f"{avg_slice_us:.3f}", "us / slice"],
            ["Avg_Library_Exec_Time_Per_Chunk", f"{avg_chunk_us:.3f}", "us / chunk"],
        ]

        if self.prof_slices > 0:
            avg_mult_slice = self.total_mult_us / self.prof_slices
            avg_rfft_slice = self.total_rfft_us / self.prof_slices
            avg_mag_slice = self.total_mag_us / self.prof_slices
            avg_mult_chunk = self.total_mult_us / max(self.prof_chunks, 1)
            avg_rfft_chunk = self.total_rfft_us / max(self.prof_chunks, 1)
            avg_mag_chunk = self.total_mag_us / max(self.prof_chunks, 1)
            rows.extend([
                ["Avg_arm_mult_Time_Per_Slice", f"{avg_mult_slice:.3f}", "us / slice"],
                ["Avg_arm_rfft_Time_Per_Slice", f"{avg_rfft_slice:.3f}", "us / slice"],
                ["Avg_arm_cmplx_mag_Time_Per_Slice", f"{avg_mag_slice:.3f}", "us / slice"],
                ["Avg_arm_mult_Time_Per_Chunk", f"{avg_mult_chunk:.3f}", "us / chunk"],
                ["Avg_arm_rfft_Time_Per_Chunk", f"{avg_rfft_chunk:.3f}", "us / chunk"],
                ["Avg_arm_cmplx_mag_Time_Per_Chunk", f"{avg_mag_chunk:.3f}", "us / chunk"],
            ])
        return rows

    def _on_csv_toggled(self, checked: bool):
        if checked:
            if self.active_mode == 3:
                self.csv_btn.setChecked(False)
                return
            try:
                self.logger.start(self.csv_dir_edit.text(), self.csv_name_edit.text(), self.num_bins, self.active_mode)
                self.csv_btn.setText("Stop Saving to CSV (Recording)")
                self.lbl_csv.setText(f"CSV: Writing ({os.path.basename(self.logger.stft_path)})")
                self.lbl_csv.setStyleSheet("color: #68D391; font-weight: bold;")
            except OSError as exc:
                QtWidgets.QMessageBox.warning(self, "CSV Error", str(exc))
                self.csv_btn.blockSignals(True)
                self.csv_btn.setChecked(False)
                self.csv_btn.blockSignals(False)
        else:
            xlsx_file = self.logger.stop(meta_rows=self._build_metadata_rows(), build_xlsx=self.xlsx_chk.isChecked())
            self.csv_btn.setText("Start Saving to CSV")
            saved_msg = f"CSV Closed ({self.logger.logged_slices} slices)"
            if xlsx_file: saved_msg += f" + {os.path.basename(xlsx_file)}"
            self.lbl_csv.setText(saved_msg)
            self.lbl_csv.setStyleSheet("color: #90CDF4; font-weight: bold;")
            self.statusBar().showMessage(f"Saved logs to {self.logger.base_dir}", 6000)

    def _on_stop_byte_received(self):
        if self.logger.active: self.csv_btn.setChecked(False)
        self.statusBar().showMessage("Stop Byte received from STM32 — CSV logging finalized and closed.", 10000)

    def _on_stop_plotting_toggled(self, stopped: bool):
        self.plotting_stopped = stopped
        self.stop_plot_btn.setText("Resume Plotting" if stopped else "Stop Plotting")
        self.statusBar().showMessage("Plotting paused" if stopped else "Plotting resumed", 3000)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        for p in ports: self.port_combo.addItem(f"{p.device} — {p.description}", p.device)
        if not ports: self.port_combo.addItem("No serial ports found", None)

    def toggle_connect(self):
        if self.reader is not None:
            self._disconnect()
            return
        port = self.port_combo.currentData()
        if not port:
            QtWidgets.QMessageBox.warning(self, "No Port", "Select a serial port first.")
            return
        try:
            sync_b, stop_b = self._get_markers()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Start/Stop Hex", str(exc))
            return

        self.sim_btn.setChecked(False)
        if self.auto_csv_chk.isChecked() and self.active_mode != 3:
            self.csv_btn.setChecked(True)

        self.reader = SerialReader(port, int(self.baud_combo.currentText()), sync_b, stop_b, self)
        self.reader.frame.connect(self.on_frame)
        self.reader.stop_marker.connect(self._on_stop_byte_received)
        self.reader.failed.connect(lambda m: QtWidgets.QMessageBox.warning(self, "Serial Error", m))
        self.reader.finished.connect(lambda: self._set_connected(False))
        self.reader.start()
        self._set_connected(True, f"Connected: {port}")

    def _disconnect(self):
        if self.reader is not None:
            self.reader.requestInterruption()
            self.reader.wait(2000)
            self.reader = None
        if self.logger.active: self.csv_btn.setChecked(False)
        self._set_connected(False)

    def _set_connected(self, connected: bool, status_text: Optional[str] = None):
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.port_combo.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)
        self.baud_combo.setEnabled(not connected)
        self.conn_lbl.setText(status_text or ("Simulating" if self.sim_btn.isChecked() else "Disconnected"))

    def _on_sim_toggled(self, on: bool):
        if on:
            self._disconnect()
            self._reset_buffers(257)
            if self.auto_csv_chk.isChecked() and self.active_mode != 3: self.csv_btn.setChecked(True)
            self.sim_timer.start()
            self._set_connected(False, "Simulating STM32 Stream")
        else:
            self.sim_timer.stop()
            if self.logger.active: self.csv_btn.setChecked(False)
            self._set_connected(False)

    def _sim_tick(self):
        bins, fft_len, slices = 257, 512, 6
        mode = self.mode_combo.currentIndex()
        prec_ui = self.prec_combo.currentIndex()
        prec = 0 if prec_ui == 0 else (prec_ui - 1)

        k = np.arange(bins)
        mag_f32 = np.empty((slices, bins), dtype=np.float32)
        for s in range(slices):
            self.sim_phase += 0.08
            k0 = 35 + 75 * (0.5 + 0.5 * math.sin(self.sim_phase))
            mag_f32[s] = 1.5e4 * np.exp(-((k - k0) / 2.0) ** 2) + self.rng.random(bins) * 20.0 + 1.0

        if prec == 1: payload = np.clip(mag_f32 * (32768.0 / fft_len), -32768, 32767).astype(np.dtype("<i2"))
        elif prec == 2: payload = np.clip(mag_f32 * (2147483648.0 / fft_len), -2147483648, 2147483647).astype(np.dtype("<i4"))
        else: payload = mag_f32

        total_us = float(1380.0 + self.rng.normal(0, 10.0))
        raw_adc = None
        prof = None

        if mode == 1:
            n_samp = self.chunk_spin.value()
            t = np.arange(n_samp, dtype=np.float32) / self.fs_spin.value()
            f_hz = (35 + 75 * (0.5 + 0.5 * math.sin(self.sim_phase))) * (self.fs_spin.value() / fft_len)
            raw_adc = np.clip(32768 + 24000 * np.sin(2 * np.pi * f_hz * t), 0, 65535).astype(np.uint16)
        elif mode == 2:
            prof = ProfilingData(mult_us=total_us * 0.15, rfft_us=total_us * 0.58, cmplx_mag_us=total_us * 0.20)

        self.on_frame(Frame(prec, mode, slices, bins, total_us, payload, raw_adc, prof))

    def _reset_buffers(self, bins: int):
        self.num_bins = bins
        hist = self.hist_spin.value()
        self.spec = np.full((hist, bins), FILL_DB, dtype=np.float32)
        self.filled = 0
        self.last_db = None
        self.has_raw_adc = False
        self.snapshot_slices.clear()
        self.snapshot_raw.clear()
        self.snapshot_count = 0
        self.img.setImage(self.spec, autoLevels=False)
        self._update_geometry()

    def clear_all(self):
        self.chunks = 0
        self.total_slices = 0
        self.total_exec_time_us = 0.0
        self.last_chunk_time_us = 0.0
        self.prof_slices = 0
        self.prof_chunks = 0
        self.total_mult_us = 0.0
        self.total_rfft_us = 0.0
        self.total_mag_us = 0.0
        self.raw_curve.setData([], [])
        self.mag_curve.setData([], [])
        self._reset_buffers(self.num_bins)
        self._update_summary_labels()

    def on_frame(self, f: Frame):
        if f.num_bins != self.num_bins: self._reset_buffers(f.num_bins)

        if self.active_mode != 3 and f.mode != self.active_mode:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(f.mode)
            self.mode_combo.blockSignals(False)
            self._on_mode_changed(f.mode)

        self.active_precision = f.precision
        override_prec = None if self.prec_combo.currentIndex() == 0 else (self.prec_combo.currentIndex() - 1)
        mag = to_float(f, override_precision=override_prec)

        self.chunks += 1
        self.total_slices += f.slices
        self.skipped = f.skipped
        self.last_chunk_time_us = f.time_us
        self.total_exec_time_us += f.time_us

        if f.profiling is not None:
            self.prof_chunks += 1
            self.prof_slices += f.slices
            self.total_mult_us += f.profiling.mult_us
            self.total_rfft_us += f.profiling.rfft_us
            self.total_mag_us += f.profiling.cmplx_mag_us

        if self.logger.active and self.active_mode != 3: self.logger.write_frame(f, mag)

        if self.active_mode != 3:
            self.snapshot_slices.append(mag)
            self.snapshot_count += f.slices
            while self.snapshot_count > self.max_snapshot_slices and len(self.snapshot_slices) > 1:
                self.snapshot_count -= self.snapshot_slices.pop(0).shape[0]
            if f.raw_adc is not None:
                self.snapshot_raw.append(f.raw_adc)
                if len(self.snapshot_raw) > 100: self.snapshot_raw.pop(0)

        if not self.plotting_stopped:
            db = 20.0 * np.log10(np.maximum(np.abs(mag), 1e-6))
            n = min(f.slices, self.spec.shape[0])
            self.spec = np.roll(self.spec, -n, axis=0)
            self.spec[-n:] = db[-n:]
            self.filled = min(self.spec.shape[0], self.filled + n)
            self.last_db = db[-1]

            if f.raw_adc is not None and len(f.raw_adc) > 0:
                self.has_raw_adc = True
                n_raw = len(f.raw_adc)
                if n_raw >= len(self.raw_adc_buf):
                    self.raw_adc_buf = f.raw_adc[-len(self.raw_adc_buf):].astype(np.float32)
                else:
                    self.raw_adc_buf = np.roll(self.raw_adc_buf, -n_raw)
                    self.raw_adc_buf[-n_raw:] = f.raw_adc.astype(np.float32)
        self.dirty = True

    def _geometry(self):
        fs = self.fs_spin.value()
        fft_len = (self.num_bins - 1) * 2
        return fs, self.hop_spin.value() / fs, fs / fft_len, self.hist_spin.value()

    def _update_geometry(self):
        fs, dt, df, hist = self._geometry()
        if self.spec is not None and hist != self.spec.shape[0]:
            new = np.full((hist, self.num_bins), FILL_DB, dtype=np.float32)
            n = min(hist, self.spec.shape[0])
            new[-n:] = self.spec[-n:]
            self.spec, self.filled = new, min(self.filled, hist)
            self.img.setImage(self.spec, autoLevels=False)
        self.img.setRect(QtCore.QRectF(-hist * dt, -0.5 * df, hist * dt, self.num_bins * df))
        self.spec_plot.setXRange(-hist * dt, 0, padding=0)
        self.spec_plot.setYRange(0, fs / 2, padding=0)
        self.mag_plot.setXRange(0, fs / 2, padding=0)
        self.dirty = True

    def _refresh_plots(self):
        if not self.dirty: return
        self.dirty = False

        if not self.plotting_stopped:
            self.img.setImage(self.spec, autoLevels=False)
            if self.auto_lvl_chk.isChecked() and self.filled > 0:
                valid = self.spec[-self.filled:]
                lo, hi = np.percentile(valid, [2, 99.5])
                self.lo = 0.7 * self.lo + 0.3 * float(lo)
                self.hi = max(self.lo + 6.0, 0.7 * self.hi + 0.3 * float(hi))
                self.bar.setLevels((self.lo, self.hi))

            if self.last_db is not None and self.mag_plot.isVisible():
                _, _, df, _ = self._geometry()
                freqs = np.arange(self.num_bins) * df
                self.mag_curve.setData(freqs, self.last_db)
                k = int(np.argmax(self.last_db[1:])) + 1
                self.peak_line.setValue(freqs[k])
                self.mag_plot.setTitle(f"Magnitude (dB) — Peak: {freqs[k]:.1f} Hz ({self.last_db[k]:.1f} dB)")

            if self.has_raw_adc and self.raw_plot.isVisible():
                fs = self.fs_spin.value()
                t_axis = np.arange(len(self.raw_adc_buf)) / fs
                self.raw_curve.setData(t_axis, self.raw_adc_buf)

        self._update_summary_labels()

    def _update_summary_labels(self):
        eff_prec = self._effective_precision(self.active_precision)
        avg_slice_us = (self.total_exec_time_us / self.total_slices) if self.total_slices > 0 else 0.0
        avg_chunk_us = (self.total_exec_time_us / self.chunks) if self.chunks > 0 else 0.0

        self.lbl_avg_slice.setText(f"Avg Time / Slice: {avg_slice_us:.2f} µs")
        self.lbl_avg_chunk.setText(f"Avg Library / Chunk: {avg_chunk_us:.1f} µs")
        self.lbl_prec.setText(f"Precision: {PRECISION_NAMES.get(eff_prec, 'Unknown')}")
        self.lbl_slices.setText(f"Captured: {self.total_slices} slices ({self.chunks} chunks)")

        if self.prof_slices > 0 and self.prof_group.isVisible():
            m_s = self.total_mult_us / self.prof_slices
            r_s = self.total_rfft_us / self.prof_slices
            g_s = self.total_mag_us / self.prof_slices
            ovh_s = max(0.0, avg_slice_us - (m_s + r_s + g_s))

            self.lbl_prof_mult.setText(f"arm_mult: {m_s:.2f} µs/slice")
            self.lbl_prof_rfft.setText(f"arm_rfft: {r_s:.2f} µs/slice")
            self.lbl_prof_mag.setText(f"arm_cmplx_mag: {g_s:.2f} µs/slice")
            self.lbl_prof_lib.setText(f"Windowing/Buffer Overhead: {ovh_s:.2f} µs/slice")

    def _on_colormap(self, name: str):
        try: self.bar.setColorMap(pg.colormap.get(name))
        except Exception: pass

    def _on_mouse_hover(self, evt):
        pos = evt[0]
        if not self.spec_plot.sceneBoundingRect().contains(pos) or self.spec is None: return
        pt = self.spec_plot.vb.mapSceneToView(pos)
        _, dt, df, hist = self._geometry()
        col = int(math.floor(pt.x() / dt)) + hist
        b = int(round(pt.y() / df))
        if 0 <= col < hist and 0 <= b < self.num_bins:
            self.hover_lbl.setText(f"t={pt.x():.3f} s | f={b * df:.1f} Hz | {self.spec[col, b]:.1f} dB")

    def export_snapshot_csv(self):
        if not self.snapshot_slices:
            QtWidgets.QMessageBox.information(self, "No Data", "No STFT slices available to export yet.")
            return
        default_file = os.path.join(self.csv_dir_edit.text(), f"{self.csv_name_edit.text()}_snapshot.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Snapshot CSV", default_file, "CSV (*.csv)")
        if not path: return

        stem = os.path.splitext(path)[0]
        data = np.vstack(self.snapshot_slices)
        idx = np.arange(data.shape[0])[:, None]
        header = "Slice_Index," + ",".join(f"Bin_{i}" for i in range(data.shape[1]))
        np.savetxt(path, np.hstack([idx, data]), delimiter=",", header=header, comments="", fmt=["%d"] + ["%.6g"] * data.shape[1])

        self.logger.meta_path = f"{stem}_metadata.csv"
        self.logger.write_metadata(self._build_metadata_rows())

        if self.snapshot_raw:
            raw_all = np.concatenate(self.snapshot_raw)
            np.savetxt(f"{stem}_raw_adc.csv", np.column_stack([np.arange(len(raw_all)), raw_all]), delimiter=",", header="Sample_Index,Raw_ADC_Value", comments="", fmt="%d")
        self.statusBar().showMessage(f"Exported snapshot & metadata to {path}", 6000)

    def _load_settings(self):
        s = self.settings
        self.fs_spin.setValue(float(s.value("fs", 900)))
        self.hop_spin.setValue(int(s.value("hop", 77)))
        self.chunk_spin.setValue(int(s.value("chunk", 512)))
        self.csv_dir_edit.setText(str(s.value("csv_dir", os.path.abspath("./logs"))))
        self.csv_name_edit.setText(str(s.value("csv_name", "stft_session")))
        self.sync_edit.setText(str(s.value("sync_hex", DEFAULT_SYNC_HEX)))
        self.stop_edit.setText(str(s.value("stop_hex", DEFAULT_STOP_HEX)))
        self.theme_combo.setCurrentIndex(int(s.value("theme", 0)))
        self.font_spin.setValue(int(s.value("font_size", 10)))

    def closeEvent(self, event):
        s = self.settings
        s.setValue("fs", self.fs_spin.value())
        s.setValue("hop", self.hop_spin.value())
        s.setValue("chunk", self.chunk_spin.value())
        s.setValue("csv_dir", self.csv_dir_edit.text())
        s.setValue("csv_name", self.csv_name_edit.text())
        s.setValue("sync_hex", self.sync_edit.text())
        s.setValue("stop_hex", self.stop_edit.text())
        s.setValue("theme", self.theme_combo.currentIndex())
        s.setValue("font_size", self.font_spin.value())
        self.sim_timer.stop()
        if self.logger.active: self.logger.stop(self._build_metadata_rows(), build_xlsx=self.xlsx_chk.isChecked())
        self._disconnect()
        super().closeEvent(event)

def main():
    pg.setConfigOptions(antialias=True, imageAxisOrder="col-major")
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = STFTVisualizerV3()
    win.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())

if __name__ == "__main__":
    main()