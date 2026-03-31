// ESP32 code for measuring resistance with capacitor and RNN
// Sequential processing version - BPTT + Calibration Mode + Optimized SGD
// Optimized for ESP32 (12-bit ADC, 3.3V logic, Larger RAM)

#include <math.h>

// --- Pin Definitions ---
#define analogPin 34   // ADC1_CH6 (GPIO34) - RC measurement node
#define chargePin 25   // GPIO25 - Charging digital pin
#define dividerPin 35  // ADC1_CH7 (GPIO35) - Supply voltage divider node

// --- Measurement Constants ---
// ESP32 ADC is 12-bit (0-4095). 63.2% of 4095 is ~2588.
#define threshold 2588
#define knownCapacitor 0.000001 // 1 uF capacitor

// --- RNN Hyperparameters (Increased hidden size for ESP32) ---
#define hiddenSize 16
#define inputSize 3
#define outputSize 1
#define learningRate 0.01
#define momentum 0.9
#define gradClip 1.0

// --- Normalization Factors ---
#define T_MAX 100000.0 // us
#define V_REF 3.3      // ESP32 is 3.3V
#define R_MAX 100000.0 // ohms

// --- RNN Weights and Biases ---
float Wxh[hiddenSize][inputSize];
float Whh[hiddenSize][hiddenSize];
float Why[outputSize][hiddenSize];
float bh[hiddenSize];
float by[outputSize];

// --- Momentum Buffers ---
float vWxh[hiddenSize][inputSize];
float vWhh[hiddenSize][hiddenSize];
float vWhy[outputSize][hiddenSize];
float vbh[hiddenSize];
float vby[outputSize];

// --- RNN States ---
float h[hiddenSize];
float h_prev[hiddenSize];
float y[outputSize];

// --- History for BPTT ---
float h_history[11][hiddenSize];
float x_history[11][inputSize];

bool calibrationMode = false;

void setup() {
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  Serial.begin(115200); // Faster baud for ESP32

  // Use esp_random() for high-quality randomness
  randomSeed(esp_random());
  rnnInit();

  // Configure ADC for 12-bit resolution (default is 12)
  analogReadResolution(12);
  // Optional: Set attenuation for better range (default is 11dB, 0-3.9V range)
  // analogSetAttenuation(ADC_11db);

  Serial.println(F("ESP32 RNN Resistor Estimator v1.0 Ready."));
  Serial.println(F("Send 'C' to toggle Calibration Mode."));
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'C' || c == 'c') {
      calibrationMode = !calibrationMode;
      Serial.print(F("Calibration Mode: "));
      Serial.println(calibrationMode ? F("ON") : F("OFF"));
    }
  }

  unsigned long startTime;
  unsigned long pulseDuration;
  int pulseCount = 0;

  // 1. Discharge
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  while (analogRead(analogPin) > 20) { delay(1); } // Noise margin for 12-bit

  // 2. Measure supply voltage (ESP32 doesn't have internal 1.1V ref like AVR)
  // Assuming external divider provides V_supply / 2 to dividerPin
  float supplyVoltage = analogRead(dividerPin) * (3.3 / 4095.0) * 2.0;

  rnnReset();

  // 3. Sequential measurements
  pinMode(chargePin, OUTPUT);
  while (pulseCount < 10) {
    digitalWrite(chargePin, HIGH);
    startTime = micros();
    while(analogRead(analogPin) < threshold) { }
    pulseDuration = micros() - startTime;

    float x_in[inputSize];
    x_in[0] = (float)pulseDuration / T_MAX;
    x_in[1] = (float)pulseCount / 10.0;
    x_in[2] = supplyVoltage / V_REF;

    for(int i=0; i<inputSize; i++) x_history[pulseCount][i] = x_in[i];
    for(int i=0; i<hiddenSize; i++) h_history[pulseCount][i] = h_prev[i];

    rnnForward(x_in);

    for(int i=0; i<hiddenSize; i++) {
        h_prev[i] = h[i];
        h_history[pulseCount+1][i] = h[i];
    }
    pulseCount++;

    digitalWrite(chargePin, LOW);
    while (analogRead(analogPin) > 20) { delay(1); }
  }

  float predictedR = y[0] * R_MAX;

  float avgD = 0;
  for(int i=0; i<10; i++) avgD += x_history[i][0] * T_MAX;
  float calculatedR = (avgD/10.0 / 1000000.0) / knownCapacitor;

  Serial.print(F("Calc R: "));
  Serial.print(calculatedR);
  Serial.print(F(" ohms | RNN Est: "));
  Serial.print(predictedR);
  Serial.println(F(" ohms"));

  if (calibrationMode) {
    Serial.println(F("Enter Ground Truth R:"));
    while (Serial.available() == 0) { delay(10); }
    float knownR = Serial.parseFloat();
    if (knownR > 0) {
        float t_val[outputSize];
        t_val[0] = knownR / R_MAX;
        rnnBackwardAndUpdate(10, t_val);
        Serial.println(F("Weights updated."));
    }
  }

  delay(2000);
}

void rnnForward(float* x_in) {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = bh[i];
    for (int j = 0; j < inputSize; j++) h[i] += Wxh[i][j] * x_in[j];
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

void rnnBackwardAndUpdate(int steps, float* t_val) {
    float dy[outputSize];
    float dh[hiddenSize];
    float dh_next[hiddenSize];
    for(int i=0; i<hiddenSize; i++) dh_next[i] = 0;

    static float dWxh[hiddenSize][inputSize];
    static float dWhh[hiddenSize][hiddenSize];
    static float dWhy[outputSize][hiddenSize];
    static float dbh[hiddenSize];
    static float dby[outputSize];

    // Clear
    memset(dWxh, 0, sizeof(dWxh)); memset(dWhh, 0, sizeof(dWhh)); memset(dWhy, 0, sizeof(dWhy));
    memset(dbh, 0, sizeof(dbh)); memset(dby, 0, sizeof(dby));

    for (int i = 0; i < outputSize; i++) {
        dy[i] = y[i] - t_val[i];
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
  float scale = 0.1;
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] = 0; vbh[i] = 0;
    for (int j = 0; j < inputSize; j++) { Wxh[i][j] = (random(-1000, 1000) / 1000.0) * scale; vWxh[i][j] = 0; }
    for (int k = 0; k < hiddenSize; k++) { Whh[i][k] = (random(-1000, 1000) / 1000.0) * scale; vWhh[i][k] = 0; }
  }
  for (int i = 0; i < outputSize; i++) {
    by[i] = 0; vby[i] = 0;
    for (int j = 0; j < hiddenSize; j++) { Why[i][j] = (random(-1000, 1000) / 1000.0) * scale; vWhy[i][j] = 0; }
  }
}

void rnnReset() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = 0;
    h_prev[i] = 0;
    for(int t=0; t<11; t++) h_history[t][i] = 0;
  }
}
