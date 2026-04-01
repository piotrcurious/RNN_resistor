# RNN/GRU Resistor Error Estimator

This project uses a Recurrent Neural Network (RNN) or Gated Recurrent Unit (GRU) to estimate and correct measurement errors when measuring resistance using an Arduino or ESP32.

## How it Works
The system measures the time constant ($\tau = RC$) of an RC circuit by charging a capacitor through an unknown resistor. Traditionally, $R = \tau / C$. However, factors like ADC non-linearity, supply voltage noise, and imperfect capacitor discharge introduce errors.

A many-to-one RNN (on Arduino Uno) or GRU (on ESP32) processes a sequence of 10 charging pulses. By training on known resistance values (Calibration Mode), the model learns to compensate for these systematic and environmental errors.

## Hardware Setup

### Arduino Uno
- **chargePin (13)**: Connected to the resistor.
- **analogPin (A0)**: Connected between the resistor and capacitor (measurement node).
- **dividerPin (A1)**: Connected to a 1:10 voltage divider from the 5V supply (to monitor $V_{supply}$).
- **Capacitor**: 1 uF (connected from measurement node to GND).

### ESP32
- **chargePin (GPIO25)**: Connected to the resistor.
- **analogPin (GPIO34)**: Measurement node.
- **dividerPin (GPIO35)**: Connected to a 1:2 voltage divider from the 3.3V supply.
- **Capacitor**: 1 uF.

## Software Features
- **Sequential BPTT**: Backpropagation Through Time trained on 10-pulse sequences.
- **Adaptive Optimizer**: RMSprop with Gradient Clipping and L2 Regularization.
- **Log Scaling**: Logarithmic normalization for a wide dynamic range (100 ohms to 1M ohms).
- **Weight Persistence**: Weights are saved/loaded from EEPROM (Uno) or SPIFFS (ESP32).
- **Residual Input**: Measures voltage before each pulse to account for incomplete discharge.
- **GRU Cells (ESP32)**: Uses Gated Recurrent Units for more robust sequence learning.

## Serial Commands
- `C`: Toggle **Calibration Mode**. When ON, you can enter the "Ground Truth" resistance value in the Serial monitor to train the model.
- `S`: **Save** current weights to persistent storage.
- `R`: **Reset** weights and optimizer state to default.
- `D`: Output **Diagnostics** (average weight magnitude).

## Simulator
A Python-based hardware simulator is provided in `hardware_sim.py` to test the machine learning logic against a virtualized RC circuit with noise and ADC non-linearity.

### Running the Simulator
1. Ensure you have Python 3 and `numpy` installed:
   ```bash
   pip install numpy
   ```
2. Run the simulator:
   ```bash
   python3 hardware_sim.py
   ```
The simulator will train a virtual RNN/GRU and display the error reduction achieved.
