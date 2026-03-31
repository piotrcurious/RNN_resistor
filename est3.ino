// Arduino code for measuring resistance with capacitor and RNN
// Improved with better normalization, linear output, and correct gradient scope

#include <math.h>

#define analogPin 0
#define chargePin 13
#define dividerPin 1
#define threshold 648 // ~63.2% of 1023 (10-bit ADC)

// RNN parameters
#define hiddenSize 8
#define inputSize 4
#define outputSize 1
#define learningRate 0.01

// Normalization factors
#define R_MAX 100000.0
#define V_REF 5.0

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

// Global Gradients
float dWxh[hiddenSize][inputSize];
float dWhh[hiddenSize][hiddenSize];
float dWhy[outputSize][hiddenSize];
float dbh[hiddenSize];
float dby[outputSize];

void setup() {
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  Serial.begin(9600);

  randomSeed(analogRead(5));
  rnnInit();
}

void loop() {
  float unknownResistor;
  float knownCapacitor = 0.000001; // 1 uF
  unsigned long startTime;
  unsigned long totalElapsedTime_us = 0;
  int pulseCount = 0;

  // Discharge the capacitor
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  while (analogRead(analogPin) > 5) {}

  // Measure supply voltage
  pinMode(chargePin, INPUT);
  analogReference(INTERNAL);
  analogRead(dividerPin); // Stabilize MUX
  delay(10);
  float dividerVoltage = analogRead(dividerPin) * (1.1 / 1023.0);
  float supplyVoltage = dividerVoltage * (10.0 + 1.0);

  // Switch back to default reference
  analogReference(DEFAULT);
  analogRead(analogPin); // Clear MUX

  // Charge the capacitor with discrete pulses
  pinMode(chargePin, OUTPUT);

  while (pulseCount < 10) {
    digitalWrite(chargePin, HIGH);
    startTime = micros();

    // Wait until threshold is reached
    while(analogRead(analogPin) < threshold) { }

    totalElapsedTime_us += (micros() - startTime);
    pulseCount++;

    // Discharge
    digitalWrite(chargePin, LOW);
    while (analogRead(analogPin) > 5) {}
  }

  // Calculate R from average time
  float averageTime_sec = (totalElapsedTime_us / 10.0) / 1000000.0;
  unknownResistor = averageTime_sec / knownCapacitor;

  // Set the input vector (Normalized)
  x[0] = unknownResistor / R_MAX;
  x[1] = pulseCount / 10.0;
  x[2] = 1.0;
  x[3] = supplyVoltage / V_REF;

  // Set the target (Normalized)
  // In a real scenario, you'd provide the actual known resistor value here
  t[0] = unknownResistor / R_MAX;

  rnnForward();

  Serial.print("Target R: ");
  Serial.print(t[0] * R_MAX);
  Serial.print(" | RNN Output: ");
  Serial.print(y[0] * R_MAX);
  Serial.println(" ohms");

  rnnBackward();
  rnnUpdate();
  // We keep h as h_prev for the next iteration if we want temporal recurrence,
  // but here each loop is a fresh measurement.
  rnnReset();

  delay(1000);
}

void rnnForward() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = bh[i];
    for (int j = 0; j < inputSize; j++) {
      h[i] += Wxh[i][j] * x[j];
    }
    for (int k = 0; k < hiddenSize; k++) {
      h[i] += Whh[i][k] * h_prev[k];
    }
    h[i] = tanhf(h[i]);
  }

  for (int i = 0; i < outputSize; i++) {
    y[i] = by[i];
    for (int j = 0; j < hiddenSize; j++) {
      y[i] += Why[i][j] * h[j];
    }
  }
}

void rnnBackward() {
  float dy[outputSize];
  for (int i = 0; i < outputSize; i++) {
    dy[i] = y[i] - t[i];
  }

  float dh[hiddenSize];
  for (int i = 0; i < hiddenSize; i++) {
    dh[i] = 0;
    for (int j = 0; j < outputSize; j++) {
      dh[i] += Why[j][i] * dy[j];
    }
    dh[i] *= (1.0 - h[i] * h[i]);
  }

  for (int i = 0; i < hiddenSize; i++) {
    dbh[i] = dh[i];
    for (int j = 0; j < inputSize; j++) {
      dWxh[i][j] = dh[i] * x[j];
    }
    for (int k = 0; k < hiddenSize; k++) {
      dWhh[i][k] = dh[i] * h_prev[k];
    }
  }

  for (int i = 0; i < outputSize; i++) {
    dby[i] = dy[i];
    for (int j = 0; j < hiddenSize; j++) {
      dWhy[i][j] = dy[i] * h[j];
    }
  }
}

void rnnUpdate() {
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] -= learningRate * dbh[i];
    for (int j = 0; j < inputSize; j++) {
      Wxh[i][j] -= learningRate * dWxh[i][j];
    }
    for (int k = 0; k < hiddenSize; k++) {
      Whh[i][k] -= learningRate * dWhh[i][k];
    }
  }

  for (int i = 0; i < outputSize; i++) {
    by[i] -= learningRate * dby[i];
    for (int j = 0; j < hiddenSize; j++) {
      Why[i][j] -= learningRate * dWhy[i][j];
    }
  }
}

void rnnInit() {
  float scale = sqrt(2.0 / (float)inputSize);
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] = 0;
    for (int j = 0; j < inputSize; j++) {
      Wxh[i][j] = (random(-1000, 1000) / 1000.0) * scale;
    }
    for (int k = 0; k < hiddenSize; k++) {
      Whh[i][k] = (random(-1000, 1000) / 1000.0) * scale;
    }
  }

  for (int i = 0; i < outputSize; i++) {
    by[i] = 0;
    for (int j = 0; j < hiddenSize; j++) {
      Why[i][j] = (random(-1000, 1000) / 1000.0) * scale;
    }
  }

  rnnReset();
}

void rnnReset() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = 0;
    h_prev[i] = 0;
  }
}
