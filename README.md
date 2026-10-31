# PC Datalogger

A live spectrogram viewer and logger for the STFT output streamed by the firmware over USB CDC.

| File | Purpose |
|---|---|
| `stft_datalogger_app.py` | GUI application (run this) |
| `datalogger_backend.py` | Packet parser, F32/Q15/Q31 conversion, serial reader, CSV/XLSX logger. The app imports it by this name, so keep both files in the same folder. |

## Install and run
```
pip install numpy pyserial pyqtgraph PyQt5 openpyxl
python stft_datalogger_app.py
```

1. Select the board's USB serial port and click **Connect**.
2. Logging to CSV starts automatically. On the firmware's stop marker it writes `<name>_stft.csv`, `<name>_metadata.csv` and `<name>.xlsx`.
3. Enter the STFT settings (sample rate, window length and type, overlap) in the app. They are saved to the metadata sheet.

The `.xlsx` file is the input to `../cmsis-stft_matlab/MATLAB_STFT_Verify.m`.
