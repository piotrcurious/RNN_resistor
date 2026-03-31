// Arduino code for measuring resistance with capacitor and RNN
// Sequential processing version - Corrected Backprop + Calibration Mode + Optimized SGD
// Consolidate and finalize production version

#include <math.h>

// --- Pin Definitions ---
#define analogPin 0   // RC measurement node
#define chargePin 13  // Charging digital pin
#define dividerPin 1 // Supply voltage divider node

// --- Measurement Constants ---
#define threshold 648 // ~63.2% of 1023 (Target voltage for tau measurement)
#define knownCapacitor 0.000001 // 1 uF capacitor

// --- RNN Hyperparameters ---
#define hiddenSize 8
#define inputSize 3  // Pulse duration, pulse index, supply voltage
#define outputSize 1
#define learningRate 0.01
#define momentum 0.9
#define gradClip 1.0

// --- Normalization Factors ---
#define T_MAX 100000.0 // Normalization for pulse duration (us)
#define V_REF 5.0      // Normalization for supply voltage
#define R_MAX 100000.0 // Normalization for predicted resistance

// --- RNN Weights and Biases ---
float Wxh[hiddenSize][inputSize];
float Whh[hiddenSize][hiddenSize];
float Why[outputSize][hiddenSize];
float bh[hiddenSize];
float by[outputSize];

// --- Optimizer State (Momentum Buffers) ---
float vWxh[hiddenSize][inputSize];
float vWhh[hiddenSize][hiddenSize];
float vWhy[outputSize][hiddenSize];
float vbh[hiddenSize];
float vby[outputSize];

// --- RNN Dynamic States ---
float h[hiddenSize];
float h_prev[hiddenSize];
float y[outputSize];

// --- RNN Inputs and Training Targets ---
float x[inputSize];
float t[outputSize];

// --- Sequence History for Backpropagation Through Time (BPTT) ---
float h_history[11][hiddenSize];
float x_history[11][inputSize];

// --- Application State ---
bool calibrationMode = false;

void setup() {
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  Serial.begin(9600);

  // Use floating analog pin for better randomness
  randomSeed(analogRead(5));
  rnnInit();

  Serial.println(F("RNN Resistor Estimator v1.0 Ready."));
  Serial.println(F("Send 'C' to toggle Calibration Mode."));
}

void loop() {
  // Handle user commands
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'C' || c == 'c') {
      calibrationMode = !calibrationMode;
      Serial.print(F("Calibration Mode: "));
      Serial.println(calibrationMode ? F("ON") : F("OFF"));
      if (calibrationMode) Serial.println(F("Ready to train. Please provide known R when prompted."));
    }
  }

  unsigned long startTime;
  unsigned long pulseDuration;
  int pulseCount = 0;

  // 1. Discharge the capacitor
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  while (analogRead(analogPin) > 5) { delay(1); }

  // 2. Measure supply voltage (using internal 1.1V ref for absolute accuracy)
  pinMode(chargePin, INPUT);
  analogReference(INTERNAL);
  analogRead(dividerPin); // Stabilize ADC MUX
  delay(10);
  float supplyVoltage = analogRead(dividerPin) * (1.1 / 1023.0) * 11.0;
  analogReference(DEFAULT);
  analogRead(analogPin); // Clear MUX for next measurement

  // 3. Reset RNN state for a new sequence
  rnnReset();

  // 4. Perform sequential measurements
  pinMode(chargePin, OUTPUT);
  while (pulseCount < 10) {
    digitalWrite(chargePin, HIGH);
    startTime = micros();
    while(analogRead(analogPin) < threshold) {
        // Optional timeout here
    }
    pulseDuration = micros() - startTime;

    // Normalize and prepare inputs
    x[0] = (float)pulseDuration / T_MAX;
    x[1] = (float)pulseCount / 10.0;
    x[2] = supplyVoltage / V_REF;

    // Record history for BPTT
    for(int i=0; i<inputSize; i++) x_history[pulseCount][i] = x[i];
    for(int i=0; i<hiddenSize; i++) h_history[pulseCount][i] = h_prev[i];

    rnnForward();

    // Advance hidden state
    for(int i=0; i<hiddenSize; i++) {
        h_prev[i] = h[i];
        h_history[pulseCount+1][i] = h[i];
    }
    pulseCount++;

    digitalWrite(chargePin, LOW);
    while (analogRead(analogPin) > 5) { delay(1); }
  }

  // 5. Output Prediction
  float predictedR = y[0] * R_MAX;

  // Traditional calculation for user reference
  float avgD = 0;
  for(int i=0; i<10; i++) avgD += x_history[i][0] * T_MAX;
  float calculatedR = (avgD/10.0 / 1000000.0) / knownCapacitor;

  Serial.print(F("Traditional Calc: "));
  Serial.print(calculatedR);
  Serial.print(F(" ohms | RNN Estimation: "));
  Serial.print(predictedR);
  Serial.println(F(" ohms"));

  // 6. Handle Training in Calibration Mode
  if (calibrationMode) {
    Serial.println(F("Enter GROUND TRUTH R (ohms):"));
    while (Serial.available() == 0) { delay(10); }
    float knownR = Serial.parseFloat();
    if (knownR > 0) {
        t[0] = knownR / R_MAX;
        Serial.print(F("Training RNN on target: "));
        Serial.println(knownR);
        rnnBackwardAndUpdate(10);
        Serial.println(F("Weights updated."));
    }
  }

  delay(2000);
}

// --- RNN Logic ---

void rnnForward() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = bh[i];
    for (int j = 0; j < inputSize; j++) h[i] += Wxh[i][j] * x[j];
    for (int k = 0; k < hiddenSize; k++) h[i] += Whh[i][k] * h_prev[k];
    h[i] = tanhf(h[i]);
  }
  for (int i = 0; i < outputSize; i++) {
    y[i] = by[i];
    for (int j = 0; j < hiddenSize; j++) y[i] += Why[i][j] * h[j];
  }
}

float clip(float val) {
    if (val > gradClip) return gradClip;
    if (val < -gradClip) return -gradClip;
    return val;
}

void rnnBackwardAndUpdate(int steps) {
    float dy[outputSize];
    float dh[hiddenSize];
    float dh_next[hiddenSize];
    for(int i=0; i<hiddenSize; i++) dh_next[i] = 0;

    // Gradient Accumulators
    float dWxh[hiddenSize][inputSize] = {0};
    float dWhh[hiddenSize][hiddenSize] = {0};
    float dWhy[outputSize][hiddenSize] = {0};
    float dbh[hiddenSize] = {0};
    float dby[outputSize] = {0};

    // Error at output (many-to-one architecture)
    for (int i = 0; i < outputSize; i++) {
        dy[i] = y[i] - t[i];
        dby[i] = dy[i];
        for (int j = 0; j < hiddenSize; j++) dWhy[i][j] = dy[i] * h_history[steps][j];
    }

    // Backprop Through Time (BPTT) loop
    for (int t_step = steps - 1; t_step >= 0; t_step--) {
        for (int i = 0; i < hiddenSize; i++) {
            dh[i] = dh_next[i];
            if (t_step == steps - 1) {
                for (int j = 0; j < outputSize; j++) dh[i] += Why[j][i] * dy[j];
            }
            // Tanh derivative applied to total error flowing to this hidden unit
            float dtanh = (1.0 - h_history[t_step+1][i] * h_history[t_step+1][i]);
            dh[i] *= dtanh;

            dbh[i] += dh[i];
            for (int j = 0; j < inputSize; j++) dWxh[i][j] += dh[i] * x_history[t_step][j];
            for (int k = 0; k < hiddenSize; k++) dWhh[i][k] += dh[i] * h_history[t_step][k];
        }
        // Flow error to previous time step
        for(int i=0; i<hiddenSize; i++) {
            dh_next[i] = 0;
            for(int j=0; j<hiddenSize; j++) dh_next[i] += Whh[j][i] * dh[j];
        }
    }

    // Optimizer Update: SGD with Momentum and Gradient Clipping
    for (int i = 0; i < hiddenSize; i++) {
        vbh[i] = momentum * vbh[i] - learningRate * clip(dbh[i]);
        bh[i] += vbh[i];
        for (int j = 0; j < inputSize; j++) {
            vWxh[i][j] = momentum * vWxh[i][j] - learningRate * clip(dWxh[i][j]);
            Wxh[i][j] += vWxh[i][j];
        }
        for (int k = 0; k < hiddenSize; k++) {
            vWhh[i][k] = momentum * vWhh[i][k] - learningRate * clip(dWhh[i][k]);
            Whh[i][k] += vWhh[i][k];
        }
    }
    for (int i = 0; i < outputSize; i++) {
        vby[i] = momentum * vby[i] - learningRate * clip(dby[i]);
        by[i] += vby[i];
        for (int j = 0; j < hiddenSize; j++) {
            vWhy[i][j] = momentum * vWhy[i][j] - learningRate * clip(dWhy[i][j]);
            Why[i][j] += vWhy[i][j];
        }
    }
}

void rnnInit() {
  float scale = 0.1; // Small random initialization
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] = 0; vbh[i] = 0;
    for (int j = 0; j < inputSize; j++) {
        Wxh[i][j] = (random(-1000, 1000) / 1000.0) * scale;
        vWxh[i][j] = 0;
    }
    for (int k = 0; k < hiddenSize; k++) {
        Whh[i][k] = (random(-1000, 1000) / 1000.0) * scale;
        vWhh[i][k] = 0;
    }
  }
  for (int i = 0; i < outputSize; i++) {
    by[i] = 0; vby[i] = 0;
    for (int j = 0; j < hiddenSize; j++) {
        Why[i][j] = (random(-1000, 1000) / 1000.0) * scale;
        vWhy[i][j] = 0;
    }
  }
}

void rnnReset() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = 0;
    h_prev[i] = 0;
    for(int t=0; t<11; t++) h_history[t][i] = 0;
  }
}
