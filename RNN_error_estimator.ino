// Arduino code for measuring resistance with capacitor and RNN
// Optimized for Arduino Uno (ATmega328P) - v2.2 (Fixed tanh & Scaling)

#include <math.h>
#include <EEPROM.h>

// #define USE_PRETRAINED
#ifdef USE_PRETRAINED
#include "pretrained_uno.h"
#endif

#define analogPin 0
#define chargePin1 13
#define chargePin2 12
#define dividerPin 1
#define tempPin 2

#define threshold 648
#define timeout_ms 2000

#define hiddenSize 6
#define inputSize 6
#define outputSize 1
#define learningRate 0.001
#define rho 0.9
#define epsilon 1e-8
#define gradClip 1.0
#define l2Reg 0.0001

#define LOG_R_MIN 1.0
#define LOG_R_MAX 7.0

float Wxh[hiddenSize][inputSize], Whh[hiddenSize][hiddenSize], Why[outputSize][hiddenSize], bh[hiddenSize], by[outputSize];
float sWxh[hiddenSize][inputSize], sWhh[hiddenSize][hiddenSize], sWhy[outputSize][hiddenSize], sbh[hiddenSize], sby[outputSize];
float h[hiddenSize], h_prev[hiddenSize], y[outputSize];

#define bpttSteps 6
float h_history[bpttSteps + 1][hiddenSize], x_history[bpttSteps][inputSize];
bool calibrationMode = false; float rollingMSE = 0; int activeRange = 0;

void setup() {
  pinMode(chargePin1, OUTPUT); pinMode(chargePin2, OUTPUT);
  digitalWrite(chargePin1, LOW); digitalWrite(chargePin2, LOW);
  Serial.begin(9600); randomSeed(analogRead(5));
  if (!loadWeights()) { rnnInit(); #ifdef USE_PRETRAINED
      loadPretrained(); #endif
  }
}

void loop() {
  if (Serial.available() > 0) { char c = Serial.read(); if (c == 'C' || c == 'c') calibrationMode = !calibrationMode; else if (c == 'S' || c == 's') saveWeights(); else if (c == 'R' || c == 'r') { rnnInit(); rollingMSE = 0; } }

  activeRange = 0; if (checkFastCharge(chargePin1)) activeRange = 1;
  int chargePin = (activeRange == 0) ? chargePin1 : chargePin2;

  unsigned long startTime, pulseDuration, loopStart; int pulseCount = 0;
  digitalWrite(chargePin1, LOW); digitalWrite(chargePin2, LOW);
  loopStart = millis();
  while (analogRead(analogPin) > 5) { if (millis() - loopStart > timeout_ms) return; }

  pinMode(chargePin1, INPUT); pinMode(chargePin2, INPUT);
  analogReference(INTERNAL); analogRead(dividerPin); delay(10);
  float supplyVoltage = analogRead(dividerPin) * (1.1 / 1023.0) * 11.0;
  float tempADC = analogRead(tempPin);
  analogReference(DEFAULT); analogRead(analogPin);

  rnnReset();
  pinMode(chargePin, OUTPUT);
  while (pulseCount < 10) {
    float residualADC = analogRead(analogPin);
    digitalWrite(chargePin, HIGH); startTime = micros(); loopStart = millis();
    while(analogRead(analogPin) < threshold) { if (millis() - loopStart > timeout_ms) { digitalWrite(chargePin, LOW); return; } }
    pulseDuration = micros() - startTime;
    float x_in[inputSize];
    x_in[0] = (log10((float)pulseDuration + 1.0)) / 6.0; x_in[1] = (float)pulseCount / 10.0; x_in[2] = supplyVoltage / 5.0; x_in[3] = residualADC / 1023.0; x_in[4] = tempADC / 1023.0; x_in[5] = (float)activeRange;
    if (pulseCount >= (10 - bpttSteps)) { int idx = pulseCount - (10 - bpttSteps); memcpy(x_history[idx], x_in, sizeof(x_in)); memcpy(h_history[idx], h_prev, sizeof(h_prev)); }
    rnnForward(x_in);
    memcpy(h_prev, h, sizeof(h));
    if (pulseCount >= (10 - bpttSteps)) memcpy(h_history[pulseCount - (10 - bpttSteps) + 1], h, sizeof(h));
    pulseCount++; digitalWrite(chargePin, LOW); loopStart = millis();
    while (analogRead(analogPin) > 5) { if (millis() - loopStart > timeout_ms) break; }
  }

  float predictedR = pow(10, y[0] * (LOG_R_MAX - LOG_R_MIN) + LOG_R_MIN);
  Serial.print(F("RNN: ")); Serial.print(predictedR); Serial.println(F(" ohms"));

  if (calibrationMode) {
    Serial.println(F("Target R:")); while (Serial.available() == 0) { delay(10); }
    float knownR = Serial.parseFloat();
    if (knownR > 0) { float target = (log10(knownR) - LOG_R_MIN) / (LOG_R_MAX - LOG_R_MIN); rollingMSE = 0.9 * rollingMSE + 0.1 * pow(y[0] - target, 2); rnnBackwardAndUpdate(bpttSteps, target); }
  }
  delay(2000);
}

float manual_tanh(float x) {
    float exp2x = exp(2.0 * x);
    return (exp2x - 1.0) / (exp2x + 1.0);
}

bool checkFastCharge(int pin) { pinMode(pin, OUTPUT); digitalWrite(pin, HIGH); unsigned long start = micros(); bool fast = false; while(analogRead(analogPin) < 200) { if (micros() - start > 1000) { fast = false; break; } fast = true; } digitalWrite(pin, LOW); while(analogRead(analogPin) > 5); return fast; }
void rnnForward(float* x_in) { for (int i = 0; i < hiddenSize; i++) { h[i] = bh[i]; for (int j = 0; j < inputSize; j++) h[i] += Wxh[i][j] * x_in[j]; for (int k = 0; k < hiddenSize; k++) h[i] += Whh[i][k] * h_prev[k]; h[i] = manual_tanh(h[i]); } for (int i = 0; i < outputSize; i++) { y[i] = by[i]; for (int j = 0; j < hiddenSize; j++) y[i] += Why[i][j] * h[j]; } }
void rnnBackwardAndUpdate(int steps, float target) {
    float dy = y[0] - target; float dh_next[hiddenSize]; memset(dh_next, 0, sizeof(dh_next));
    static float dWxh[hiddenSize][inputSize], dWhh[hiddenSize][hiddenSize], dWhy[outputSize][hiddenSize], dbh[hiddenSize], dby[outputSize];
    memset(dWxh, 0, sizeof(dWxh)); memset(dWhh, 0, sizeof(dWhh)); memset(dWhy, 0, sizeof(dWhy)); memset(dbh, 0, sizeof(dbh)); memset(dby, 0, sizeof(dby));
    dby[0] = dy; for (int j = 0; j < hiddenSize; j++) dWhy[0][j] = dy * h_history[steps][j];
    for (int t = steps - 1; t >= 0; t--) {
        float dh_curr[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            float grad_h = dh_next[i] + Why[0][i] * (t == steps - 1 ? dy : 0);
            dh_curr[i] = grad_h * (1.0 - h_history[t+1][i] * h_history[t+1][i]);
            dbh[i] += dh_curr[i];
            for (int j = 0; j < inputSize; j++) dWxh[i][j] += dh_curr[i] * x_history[t][j];
            for (int k = 0; k < hiddenSize; k++) dWhh[i][k] += dh_curr[i] * h_history[t][k];
        }
        for (int i = 0; i < hiddenSize; i++) { dh_next[i] = 0; for (int j = 0; j < hiddenSize; j++) dh_next[i] += Whh[j][i] * dh_curr[j]; }
    }
    auto upd = [&](float& w, float& g, float& s) { g = (g > 1.0 ? 1.0 : (g < -1.0 ? -1.0 : g)) + l2Reg * w; s = rho * s + (1 - rho) * g * g; w -= learningRate * g / (sqrt(s) + epsilon); };
    for (int i = 0; i < hiddenSize; i++) { upd(bh[i], dbh[i], sbh[i]); for (int j = 0; j < inputSize; j++) upd(Wxh[i][j], dWxh[i][j], sWxh[i][j]); for (int k = 0; k < hiddenSize; k++) upd(Whh[i][k], dWhh[i][k], sWhh[i][k]); upd(Why[0][i], dWhy[0][i], sWhy[0][i]); }
    upd(by[0], dby[0], sby[0]);
}
void rnnInit() { float sc = sqrt(2.0 / (inputSize + 2 * hiddenSize)); for (int i = 0; i < hiddenSize; i++) { bh[i] = 0; sbh[i] = 0; for (int j = 0; j < inputSize; j++) { Wxh[i][j] = (random(-1000, 1000) / 1000.0) * sc; sWxh[i][j] = 0; } for (int k = 0; k < hiddenSize; k++) { Whh[i][k] = (random(-1000, 1000) / 1000.0) * sc; sWhh[i][k] = 0; } Why[0][i] = (random(-1000, 1000) / 1000.0) * sc; sWhy[0][i] = 0; } by[0] = 0; sby[0] = 0; }
#ifdef USE_PRETRAINED
void loadPretrained() { memcpy(Wxh, pt_Wxh, sizeof(Wxh)); memcpy(Whh, pt_Whh, sizeof(Whh)); memcpy(Why, pt_Why, sizeof(Why)); memcpy(bh, pt_bh, sizeof(bh)); memcpy(by, pt_by, sizeof(by)); }
#endif
void rnnReset() { for (int i = 0; i < hiddenSize; i++) { h[i] = 0; h_prev[i] = 0; } memset(h_history, 0, sizeof(h_history)); }
void saveWeights() { int addr = 0; EEPROM.put(addr, Wxh); addr += sizeof(Wxh); EEPROM.put(addr, Whh); addr += sizeof(Whh); EEPROM.put(addr, Why); addr += sizeof(Why); EEPROM.put(addr, bh); addr += sizeof(bh); EEPROM.put(addr, by); addr += sizeof(by); }
bool loadWeights() { int addr = 0; EEPROM.get(addr, Wxh); if(isnan(Wxh[0][0]) || isinf(Wxh[0][0])) return false; addr += sizeof(Wxh); EEPROM.get(addr, Whh); addr += sizeof(Whh); EEPROM.get(addr, Why); addr += sizeof(Why); EEPROM.get(addr, bh); addr += sizeof(bh); EEPROM.get(addr, by); addr += sizeof(by); return true; }
