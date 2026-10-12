# Validation

- All 7 notebook code cells executed successfully; genuine outputs are saved.
- Notebook format validated with nbformat.
- Python solution/support modules pass syntax compilation.
- Report and Arabic explanation are included.
- Generated C model array verified byte-for-byte against the trained TFLite file.
- Ten real mock images quantized using model metadata and evaluated on desktop.
- GitHub Actions builds ESP32-S3 firmware on push; see Actions for the final result.
- No real-board runtime measurements or Wokwi observations are claimed.

## GitHub firmware build

**PASSED**: [Build run](https://github.com/ggdh17934/AIoT-2025--TP7/actions/runs/38262008881).

The included firmware and pinned platform configuration compiled successfully on GitHub's Linux runner.
This proves compilation; board execution, sensor/LCD behavior and runtime timing remain separate observations.
