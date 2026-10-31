#!/usr/bin/env python3
"""
stft_backend_v3.py
Modular backend for STM32 STFT Visualizer & Datalogger v3:
  - Packet protocol definitions & self-resynchronizing binary parser
  - F32 / Q15 / Q31 fixed-point normalization
  - Streaming CSV + Multi-sheet Excel (.xlsx) logger
  - Background QThread serial reader
"""

import csv
import datetime
import math
import os
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np
import serial
from pyqtgraph.Qt import QtCore

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# --------------------------------------------------------------------------
# Protocol Constants & Packet Formats (Little-Endian)
# --------------------------------------------------------------------------
BASE_HDR_FMT = "<HBBIIf"
BASE_HDR_SIZE = struct.calcsize(BASE_HDR_FMT)      # 16 bytes

MODE1_EXT_FMT = "<I"                               # uint32_t raw_num_samples
MODE1_EXT_SIZE = struct.calcsize(MODE1_EXT_FMT)    # 4 bytes

MODE2_EXT_FMT = "<fff"                             # float mult_us, rfft_us, cmplx_mag_us
MODE2_EXT_SIZE = struct.calcsize(MODE2_EXT_FMT)    # 12 bytes

DEFAULT_SYNC_HEX = "A5A5"
DEFAULT_STOP_HEX = "A9A9A9A9"

DTYPES = {
    0: np.dtype("<f4"),  # F32 (float32)
    1: np.dtype("<i2"),  # Q15 (int16)
    2: np.dtype("<i4"),  # Q31 (int32)
}

PRECISION_NAMES = {
    0: "F32 (float32)",
    1: "Q15 (int16)",
    2: "Q31 (int32)",
}

MODE_NAMES = {
    0: "Mode 0: Standard",
    1: "Mode 1: Raw Signal + STFT",
    2: "Mode 2: Profiling / Analysis",
    3: "Mode 3: Production (View Only)",
}

MAX_SLICES = 256
MIN_BINS, MAX_BINS = 17, 2049  # FFT sizes 32 .. 4096


@dataclass
class ProfilingData:
    mult_us: float = 0.0
    rfft_us: float = 0.0
    cmplx_mag_us: float = 0.0


@dataclass
class Frame:
    precision: int
    mode: int
    slices: int
    num_bins: int
    time_us: float
    data: np.ndarray                         # shape: (slices, num_bins)
    raw_adc: Optional[np.ndarray] = None     # shape: (raw_num_samples,)
    profiling: Optional[ProfilingData] = None
    skipped: int = 0


def parse_hex_marker(hex_str: str, expected_bytes: int) -> bytes:
    """
    Convert user hex string (e.g. '0xA5A5' or 'A9A9A9A9') into little-endian wire bytes
    matching STM32 uint16_t / uint32_t transmission order.
    """
    clean = hex_str.strip().lower().replace("0x", "").replace(" ", "")
    if len(clean) != expected_bytes * 2:
        raise ValueError(
            f"Expected {expected_bytes} bytes ({expected_bytes * 2} hex characters), got '{hex_str}'."
        )
    raw = bytes.fromhex(clean)
    return raw[::-1]  # Little-endian wire order


def to_float(frame: Frame, override_precision: Optional[int] = None) -> np.ndarray:
    """
    Convert F32, Q15, or Q31 STFT array to float32 magnitudes.
    CMSIS-DSP RFFT Q15/Q31 outputs are scaled down by 1 / FFT_LEN internally.
    """
    prec = frame.precision if override_precision is None else override_precision
    x = frame.data.astype(np.float32)
    fft_len = (frame.num_bins - 1) * 2
    if prec == 1:    # Q15
        x *= 2*fft_len / 32768.0
    elif prec == 2:  # Q31
        x *= 2*fft_len / 2147483648.0
    return x


# --------------------------------------------------------------------------
# Binary Stream Parser
# --------------------------------------------------------------------------
class FrameParser:
    """Self-resynchronizing parser supporting Modes 0, 1, 2, and 3."""

    def __init__(self, sync_bytes: bytes = b"\xA5\xA5", stop_bytes: bytes = b"\xA9\xA9\xA9\xA9"):
        self.buf = bytearray()
        self.skipped = 0
        self.sync_bytes = sync_bytes
        self.stop_bytes = stop_bytes

    @staticmethod
    def _valid_header(prec: int, mode: int, slices: int, nbins: int, t_us: float) -> bool:
        if prec not in DTYPES or mode not in (0, 1, 2, 3):
            return False
        if not (1 <= slices <= MAX_SLICES):
            return False
        if not (MIN_BINS <= nbins <= MAX_BINS):
            return False
        if (nbins - 1) & (nbins - 2):  # (nbins - 1) must be a power of 2
            return False
        return math.isfinite(t_us) and 0.0 <= t_us < 1e9

    def feed(self, data: bytes):
        buf = self.buf
        buf += data
        events = []
        sync = self.sync_bytes
        stop = self.stop_bytes
        keep_tail = max(len(sync), len(stop)) - 1

        while True:
            idx_sync = buf.find(sync)
            idx_stop = buf.find(stop) if stop else -1
            cands = [i for i in (idx_sync, idx_stop) if i >= 0]

            if not cands:
                if len(buf) > keep_tail:
                    self.skipped += len(buf) - keep_tail
                    del buf[:-keep_tail]
                break

            start = min(cands)
            if start > 0:
                self.skipped += start
                del buf[:start]

            if stop and buf.startswith(stop):
                events.append(("stop", None))
                del buf[:len(stop)]
                continue

            if len(buf) < BASE_HDR_SIZE:
                break

            _, prec, mode, slices, nbins, t_us = struct.unpack_from(BASE_HDR_FMT, buf, 0)
            if not self._valid_header(prec, mode, slices, nbins, t_us):
                self.skipped += 1
                del buf[:1]
                continue

            dt = DTYPES[prec]
            stft_bytes = slices * nbins * dt.itemsize
            offset = BASE_HDR_SIZE
            raw_adc_arr = None
            prof_data = None

            if mode == 1:
                if len(buf) < offset + MODE1_EXT_SIZE:
                    break
                (raw_count,) = struct.unpack_from(MODE1_EXT_FMT, buf, offset)
                if raw_count > 65536:
                    self.skipped += 1
                    del buf[:1]
                    continue
                offset += MODE1_EXT_SIZE
                raw_bytes = raw_count * 2  # uint16_t samples
                if len(buf) < offset + raw_bytes + stft_bytes:
                    break
                raw_adc_arr = np.frombuffer(bytes(buf[offset:offset + raw_bytes]), dtype=np.dtype("<u2")).copy()
                offset += raw_bytes

            elif mode == 2:
                if len(buf) < offset + MODE2_EXT_SIZE:
                    break
                mult_us, rfft_us, mag_us = struct.unpack_from(MODE2_EXT_FMT, buf, offset)
                if not (math.isfinite(mult_us) and math.isfinite(rfft_us) and math.isfinite(mag_us)):
                    self.skipped += 1
                    del buf[:1]
                    continue
                offset += MODE2_EXT_SIZE
                if len(buf) < offset + stft_bytes:
                    break
                prof_data = ProfilingData(mult_us=mult_us, rfft_us=rfft_us, cmplx_mag_us=mag_us)

            else:
                if len(buf) < offset + stft_bytes:
                    break

            total_len = offset + stft_bytes
            stft_payload = np.frombuffer(bytes(buf[offset:total_len]), dtype=dt).copy()
            del buf[:total_len]

            events.append((
                "frame",
                Frame(
                    precision=prec,
                    mode=mode,
                    slices=slices,
                    num_bins=nbins,
                    time_us=t_us,
                    data=stft_payload.reshape(slices, nbins),
                    raw_adc=raw_adc_arr,
                    profiling=prof_data,
                    skipped=self.skipped,
                ),
            ))
        return events


# --------------------------------------------------------------------------
# Streaming CSV & Multi-Sheet Workbook Logger
# --------------------------------------------------------------------------
class StreamingCSVLogger:
    """
    Writes incoming STFT slices, Raw ADC samples, and Metadata directly to disk
    so memory consumption stays flat during long streaming sessions.
    """

    def __init__(self):
        self.active = False
        self.base_dir = ""
        self.base_name = ""
        self.stft_path = ""
        self.raw_path = ""
        self.meta_path = ""
        self.xlsx_path = ""

        self._stft_file = None
        self._stft_writer = None
        self._raw_file = None
        self._raw_writer = None

        self.logged_slices = 0
        self.logged_raw_samples = 0
        self.logged_chunks = 0

    def start(self, directory: str, filename_stem: str, num_bins: int, mode: int):
        self.stop()
        os.makedirs(directory, exist_ok=True)
        stem = os.path.splitext(filename_stem.strip())[0] or "stft_capture"
        self.base_dir = directory
        self.base_name = stem

        self.stft_path = os.path.join(directory, f"{stem}_stft.csv")
        self.raw_path = os.path.join(directory, f"{stem}_raw_adc.csv")
        self.meta_path = os.path.join(directory, f"{stem}_metadata.csv")
        self.xlsx_path = os.path.join(directory, f"{stem}.xlsx")

        self._stft_file = open(self.stft_path, mode="w", newline="", encoding="utf-8")
        self._stft_writer = csv.writer(self._stft_file)
        header = ["Slice_Index", "Chunk_Index", "Slice_Exec_Time_us"] + [f"Bin_{i}" for i in range(num_bins)]
        self._stft_writer.writerow(header)

        if mode == 1:
            self._raw_file = open(self.raw_path, mode="w", newline="", encoding="utf-8")
            self._raw_writer = csv.writer(self._raw_file)
            self._raw_writer.writerow(["Sample_Index", "Chunk_Index", "Raw_ADC_Value"])

        self.logged_slices = 0
        self.logged_raw_samples = 0
        self.logged_chunks = 0
        self.active = True

    def write_frame(self, frame: Frame, mag_float: np.ndarray):
        if not self.active or self._stft_writer is None:
            return

        slice_time_us = frame.time_us / max(frame.slices, 1)
        for s in range(frame.slices):
            row = [self.logged_slices, self.logged_chunks, f"{slice_time_us:.3f}"] + [
                f"{val:.6g}" for val in mag_float[s]
            ]
            self._stft_writer.writerow(row)
            self.logged_slices += 1

        if frame.raw_adc is not None:
            if self._raw_file is None:
                self._raw_file = open(self.raw_path, mode="w", newline="", encoding="utf-8")
                self._raw_writer = csv.writer(self._raw_file)
                self._raw_writer.writerow(["Sample_Index", "Chunk_Index", "Raw_ADC_Value"])
            for val in frame.raw_adc:
                self._raw_writer.writerow([self.logged_raw_samples, self.logged_chunks, int(val)])
                self.logged_raw_samples += 1

        self.logged_chunks += 1
        if self.logged_chunks % 10 == 0:
            self._stft_file.flush()
            if self._raw_file:
                self._raw_file.flush()

    def write_metadata(self, meta_rows: list):
        if not self.meta_path:
            return
        with open(self.meta_path, mode="w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(["Parameter", "Value", "Units / Notes"])
            for row in meta_rows:
                writer.writerow(row)

    def stop(self, meta_rows: Optional[list] = None, build_xlsx: bool = False) -> Optional[str]:
        if self._stft_file:
            try:
                self._stft_file.flush()
                self._stft_file.close()
            except Exception:
                pass
            self._stft_file = None
            self._stft_writer = None

        if self._raw_file:
            try:
                self._raw_file.flush()
                self._raw_file.close()
            except Exception:
                pass
            self._raw_file = None
            self._raw_writer = None

        self.active = False
        if meta_rows is not None and self.meta_path:
            self.write_metadata(meta_rows)

        if build_xlsx and HAS_OPENPYXL:
            return self.export_to_excel()
        return None

    def export_to_excel(self) -> Optional[str]:
        """Combine Metadata, STFT Data, and Raw ADC CSVs into a multi-sheet .xlsx file."""
        if not HAS_OPENPYXL or not self.stft_path or not os.path.exists(self.stft_path):
            return None
        wb = openpyxl.Workbook()

        # Sheet 1: Metadata
        ws_meta = wb.active
        ws_meta.title = "Metadata"
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    ws_meta.append(row)

        # Sheet 2: STFT_Data
        ws_stft = wb.create_sheet(title="STFT_Data")
        with open(self.stft_path, "r", encoding="utf-8") as f:
            for idx, row in enumerate(csv.reader(f)):
                if idx == 0:
                    ws_stft.append(row)
                else:
                    ws_stft.append([int(row[0]), int(row[1]), float(row[2])] + [float(v) for v in row[3:]])

        # Sheet 3: Raw_ADC (if recorded in Mode 1)
        if self.raw_path and os.path.exists(self.raw_path) and self.logged_raw_samples > 0:
            ws_raw = wb.create_sheet(title="Raw_ADC")
            with open(self.raw_path, "r", encoding="utf-8") as f:
                for idx, row in enumerate(csv.reader(f)):
                    if idx == 0:
                        ws_raw.append(row)
                    else:
                        ws_raw.append([int(row[0]), int(row[1]), int(row[2])])

        wb.save(self.xlsx_path)
        return self.xlsx_path


# --------------------------------------------------------------------------
# Background Serial Reader Thread
# --------------------------------------------------------------------------
class SerialReader(QtCore.QThread):
    frame = QtCore.Signal(object)
    stop_marker = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, port: str, baud: int, sync_bytes: bytes, stop_bytes: bytes, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self.sync_bytes = sync_bytes
        self.stop_bytes = stop_bytes

    def run(self):
        parser = FrameParser(self.sync_bytes, self.stop_bytes)
        try:
            with serial.Serial(self.port, self.baud, timeout=0.05) as ser:
                ser.reset_input_buffer()
                while not self.isInterruptionRequested():
                    n = ser.in_waiting
                    data = ser.read(n if n > 0 else 1)
                    if not data:
                        continue
                    for kind, payload in parser.feed(data):
                        if kind == "frame":
                            self.frame.emit(payload)
                        elif kind == "stop":
                            self.stop_marker.emit()
        except (serial.SerialException, OSError) as exc:
            self.failed.emit(str(exc))