// Arduino code for measuring resistance with capacitor and RNN
// Sequential processing version - Corrected Backprop + Calibration Mode

#include <math.h>

#define analogPin 0
#define chargePin 13
#define dividerPin 1
#define threshold 648

// RNN parameters
#define hiddenSize 8
#define inputSize 3
#define outputSize 1
#define learningRate 0.01

// Normalization factors
#define T_MAX 100000.0
#define V_REF 5.0
#define R_MAX 100000.0

// RNN weights and biases
float Wxh[hiddenSize][inputSize];
float Whh[hiddenSize][hiddenSize];
float Why[outputSize][hiddenSize];
float bh[hiddenSize];
float by[outputSize];

// RNN states
float h[hiddenSize];
float h_prev[hiddenSize];
float y[outputSize];

// RNN inputs and targets
float x[inputSize];
float t[outputSize];

// History for BPTT
float h_history[11][hiddenSize];
float x_history[11][inputSize];

bool calibrationMode = false;

void setup() {
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  Serial.begin(9600);

  randomSeed(analogRead(5));
  rnnInit();

  Serial.println("RNN Resistor Estimator Ready.");
  Serial.println("Send 'C' to toggle Calibration Mode.");
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'C' || c == 'c') {
      calibrationMode = !calibrationMode;
      Serial.print("Calibration Mode: ");
      Serial.println(calibrationMode ? "ON" : "OFF");
      if (calibrationMode) Serial.println("Send known resistance value in ohms when prompted.");
    }
  }

  unsigned long startTime;
  unsigned long pulseDuration;
  int pulseCount = 0;

  // Discharge
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  while (analogRead(analogPin) > 5) {}

  // Measure supply voltage
  pinMode(chargePin, INPUT);
  analogReference(INTERNAL);
  analogRead(dividerPin);
  delay(10);
  float supplyVoltage = analogRead(dividerPin) * (1.1 / 1023.0) * 11.0;
  analogReference(DEFAULT);
  analogRead(analogPin);

  rnnReset();

  // Sequential pulses
  pinMode(chargePin, OUTPUT);
  while (pulseCount < 10) {
    digitalWrite(chargePin, HIGH);
    startTime = micros();
    while(analogRead(analogPin) < threshold) { }
    pulseDuration = micros() - startTime;

    x[0] = (float)pulseDuration / T_MAX;
    x[1] = (float)pulseCount / 10.0;
    x[2] = supplyVoltage / V_REF;

    for(int i=0; i<inputSize; i++) x_history[pulseCount][i] = x[i];
    for(int i=0; i<hiddenSize; i++) h_history[pulseCount][i] = h_prev[i];

    rnnForward();

    for(int i=0; i<hiddenSize; i++) {
        h_prev[i] = h[i];
        h_history[pulseCount+1][i] = h[i];
    }
    pulseCount++;

    digitalWrite(chargePin, LOW);
    while (analogRead(analogPin) > 5) {}
  }

  float predictedR = y[0] * R_MAX;

  // Basic calculation for reference
  float avgD = 0;
  for(int i=0; i<10; i++) avgD += x_history[i][0] * T_MAX;
  float calculatedR = (avgD/10.0 / 1000000.0) / 0.000001;

  Serial.print("Calc R: ");
  Serial.print(calculatedR);
  Serial.print(" | RNN R: ");
  Serial.print(predictedR);
  Serial.println(" ohms");

  if (calibrationMode) {
    Serial.println("Enter known R:");
    while (Serial.available() == 0) {}
    float knownR = Serial.parseFloat();
    if (knownR > 0) {
        t[0] = knownR / R_MAX;
        Serial.print("Training on target: ");
        Serial.println(knownR);
        rnnBackwardAndUpdate(10);
    }
  } else {
    // In normal mode, we can still perform "self-supervised" update or skip
    // For now, let's skip training in normal mode to prevent divergence
    // rnnBackwardAndUpdate(10);
  }

  delay(2000);
}

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

void rnnBackwardAndUpdate(int steps) {
    float dy[outputSize];
    float dh[hiddenSize];
    float dh_next[hiddenSize];
    for(int i=0; i<hiddenSize; i++) dh_next[i] = 0;

    float dWxh[hiddenSize][inputSize] = {0};
    float dWhh[hiddenSize][hiddenSize] = {0};
    float dWhy[outputSize][hiddenSize] = {0};
    float dbh[hiddenSize] = {0};
    float dby[outputSize] = {0};

    for (int i = 0; i < outputSize; i++) {
        dy[i] = y[i] - t[i];
        dby[i] = dy[i];
        for (int j = 0; j < hiddenSize; j++) dWhy[i][j] = dy[i] * h_history[steps][j];
    }

    for (int t_step = steps - 1; t_step >= 0; t_step--) {
        for (int i = 0; i < hiddenSize; i++) {
            dh[i] = dh_next[i];
            if (t_step == steps - 1) {
                for (int j = 0; j < outputSize; j++) dh[i] += Why[j][i] * dy[j];
            }
            float dtanh = (1.0 - h_history[t_step+1][i] * h_history[t_step+1][i]);
            dh[i] *= dtanh;
            dbh[i] += dh[i];
            for (int j = 0; j < inputSize; j++) dWxh[i][j] += dh[i] * x_history[t_step][j];
            for (int k = 0; k < hiddenSize; k++) dWhh[i][k] += dh[i] * h_history[t_step][k];
        }
        for(int i=0; i<hiddenSize; i++) {
            dh_next[i] = 0;
            for(int j=0; j<hiddenSize; j++) dh_next[i] += Whh[j][i] * dh[j];
        }
    }

    for (int i = 0; i < hiddenSize; i++) {
        bh[i] -= learningRate * dbh[i];
        for (int j = 0; j < inputSize; j++) Wxh[i][j] -= learningRate * dWxh[i][j];
        for (int k = 0; k < hiddenSize; k++) Whh[i][k] -= learningRate * dWhh[i][k];
    }
    for (int i = 0; i < outputSize; i++) {
        by[i] -= learningRate * dby[i];
        for (int j = 0; j < hiddenSize; j++) Why[i][j] -= learningRate * dWhy[i][j];
    }
}

void rnnInit() {
  float scale = 0.1;
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] = 0;
    for (int j = 0; j < inputSize; j++) Wxh[i][j] = (random(-1000, 1000) / 1000.0) * scale;
    for (int k = 0; k < hiddenSize; k++) Whh[i][k] = (random(-1000, 1000) / 1000.0) * scale;
  }
  for (int i = 0; i < outputSize; i++) {
    by[i] = 0;
    for (int j = 0; j < hiddenSize; j++) Why[i][j] = (random(-1000, 1000) / 1000.0) * scale;
  }
}

void rnnReset() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = 0;
    h_prev[i] = 0;
    for(int t=0; t<11; t++) h_history[t][i] = 0;
  }
}
