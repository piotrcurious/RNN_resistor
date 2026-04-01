# RNN/GRU Resistor Error Estimator v2.3

This project implements an advanced resistance measurement system for Arduino Uno and ESP32. It uses Deep Learning (RNN/GRU) to correct errors caused by ADC non-linearity, supply noise, and environmental factors.

## Key Features
- **Multi-Platform Support**: Optimized RNN for Arduino Uno and GRU for ESP32.
- **Wide Range**: Dual-capacitor auto-ranging logic (1uF and 10nF) covers 10$\Omega$ to 10M$\Omega$.
- **Advanced ML**: Backpropagation Through Time (BPTT), RMSprop optimizer, and L2 Regularization.
- **Adaptive Estimation**: Logarithmic scaling for inputs/outputs and Self-Supervised Online Learning to track component drift.
- **Rich Interface**: Serial Calibration Mode, persistent weight storage (EEPROM/SPIFFS), and a Bootstrap-based Web Dashboard for ESP32.
- **Reliability**: Safety timeouts on all loops and Monte Carlo Dropout for real-time confidence estimation (ESP32).

## Quick Start
1. **Prepare Hardware**: Connect the RC circuit and voltage divider as described in the Hardware Setup section.
2. **Train (Optional)**: Run `python3 hardware_sim.py [uno|esp32]` to pre-train the model and export weights.
3. **Deploy**: Define `USE_PRETRAINED` in your `.ino` file and flash the device.
4. **Calibrate**: Open the Serial Monitor, send 'C' for Calibration Mode, and enter a ground-truth resistor value to fine-tune the model.

## Hardware Setup

### Arduino Uno
- **chargePin1 (13)**: 1uF range.
- **chargePin2 (12)**: 10nF range.
- **analogPin (A0)**: Measurement node.
- **dividerPin (A1)**: 1:10 voltage divider.
- **tempPin (A2)**: Thermistor.

### ESP32
- **chargePin1 (GPIO25)**: 1uF range.
- **chargePin2 (GPIO26)**: 10nF range.
- **analogPin (GPIO34)**: Measurement node.
- **dividerPin (GPIO35)**: 1:2 voltage divider.
- **tempPin (GPIO32)**: Temperature sensor.

## Serial Commands
- `C`: Toggle **Calibration Mode**.
- `S`: **Save** weights to EEPROM/SPIFFS.
- `R`: **Reset** weights and optimizer state.
- `D`: Output **Diagnostics** (rolling MSE).

## Simulation
A high-fidelity Python simulator (`hardware_sim.py`) is provided to model physics, noise, and TCR effects.
```bash
pip install numpy matplotlib
python3 hardware_sim.py [uno|esp32] [--plot]
```
