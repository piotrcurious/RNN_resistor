// ESP32 code for measuring resistance with capacitor and GRU
// Optimized for ESP32 - v2.2 (Scaling sync)

#include <math.h>
#include <FS.h>
#include <SPIFFS.h>
#include <WiFi.h>
#include <WebServer.h>

// #define USE_PRETRAINED
#ifdef USE_PRETRAINED
#include "pretrained_esp32.h"
#endif

#define analogPin 34
#define chargePin1 25
#define chargePin2 26
#define dividerPin 35
#define tempPin 32
#define threshold 2588
#define timeout_ms 2000

#define hiddenSize 16
#define inputSize 6
#define outputSize 1
#define learningRate 0.001
#define rho 0.9
#define epsilon 1e-8
#define gradClip 1.0
#define l2Reg 0.0001

#define LOG_R_MIN 1.0
#define LOG_R_MAX 7.0

float Wz[hiddenSize][inputSize+hiddenSize], Wr[hiddenSize][inputSize+hiddenSize], Wh[hiddenSize][inputSize+hiddenSize], bz[hiddenSize], br[hiddenSize], bh[hiddenSize], Why[outputSize][hiddenSize], by[outputSize];
float sWz[hiddenSize][inputSize+hiddenSize], sWr[hiddenSize][inputSize+hiddenSize], sWh[hiddenSize][inputSize+hiddenSize], sbz[hiddenSize], sbr[hiddenSize], sbh[hiddenSize], sWhy[outputSize][hiddenSize], sby[outputSize];
float h[hiddenSize], h_prev[hiddenSize], y[outputSize];
float h_history[11][hiddenSize], x_history[11][inputSize], z_history[11][hiddenSize], r_history[11][hiddenSize], h_tilde_history[11][hiddenSize];

bool calibrationMode = false; float rollingMSE = 0; int activeRange = 0;
WebServer server(80);

void setup() {
  pinMode(chargePin1, OUTPUT); pinMode(chargePin2, OUTPUT);
  digitalWrite(chargePin1, LOW); digitalWrite(chargePin2, LOW);
  Serial.begin(115200); randomSeed(esp_random());
  if (!SPIFFS.begin(true)) Serial.println(F("SPIFFS fail"));
  if (!loadWeights()) { gruInit(); #ifdef USE_PRETRAINED
      loadPretrained(); #endif
  }
  analogReadResolution(12); WiFi.softAP("ResistorGRU", "12345678");
  server.on("/", handleRoot); server.on("/cal", [](){ calibrationMode = !calibrationMode; server.send(200, "text/plain", calibrationMode?"ON":"OFF"); });
  server.begin();
}

void loop() {
  server.handleClient();
  if (Serial.available() > 0) { char c = Serial.read(); if (c == 'C' || c == 'c') calibrationMode = !calibrationMode; else if (c == 'S' || c == 's') saveWeights(); }
  activeRange = 0; if (checkFastCharge(chargePin1)) activeRange = 1;
  int chargePin = (activeRange == 0) ? chargePin1 : chargePin2;
  unsigned long startTime, pulseDuration, loopStart; int pulseCount = 0;
  digitalWrite(chargePin1, LOW); digitalWrite(chargePin2, LOW); loopStart = millis();
  while (analogRead(analogPin) > 20) { if (millis() - loopStart > timeout_ms) { server.handleClient(); return; } }
  float supplyVoltage = analogRead(dividerPin) * (3.3 / 4095.0) * 2.0; float tempADC = analogRead(tempPin);
  gruReset();
  while (pulseCount < 10) {
    float residualADC = analogRead(analogPin);
    digitalWrite(chargePin, HIGH); startTime = micros(); loopStart = millis();
    while(analogRead(analogPin) < threshold) { if (millis() - loopStart > timeout_ms) { digitalWrite(chargePin, LOW); return; } }
    pulseDuration = micros() - startTime;
    float x_in[inputSize];
    x_in[0] = (log10((float)pulseDuration + 1.0)) / 6.0; x_in[1] = (float)pulseCount / 10.0; x_in[2] = supplyVoltage / 5.0; x_in[3] = residualADC / 4095.0; x_in[4] = tempADC / 4095.0; x_in[5] = (float)activeRange;
    memcpy(x_history[pulseCount], x_in, sizeof(x_in)); memcpy(h_history[pulseCount], h_prev, sizeof(h_prev));
    gruForward(x_in, pulseCount);
    memcpy(h_prev, h, sizeof(h)); memcpy(h_history[pulseCount+1], h, sizeof(h));
    pulseCount++; digitalWrite(chargePin, LOW); loopStart = millis();
    while (analogRead(analogPin) > 20) { if (millis() - loopStart > timeout_ms) break; delay(1); }
    server.handleClient();
  }
  float predictedR = pow(10, y[0] * (LOG_R_MAX - LOG_R_MIN) + LOG_R_MIN);
  Serial.print(F("GRU: ")); Serial.print(predictedR); Serial.println(F(" ohms"));
  if (calibrationMode) {
    Serial.println(F("Target R:")); while (Serial.available() == 0) { server.handleClient(); delay(10); }
    float knownR = Serial.parseFloat();
    if (knownR > 0) { float target = (log10(knownR) - LOG_R_MIN) / (LOG_R_MAX - LOG_R_MIN); rollingMSE = 0.9 * rollingMSE + 0.1 * pow(y[0] - target, 2); gruBackwardAndUpdate(10, target); }
  }
  delay(1000);
}

bool checkFastCharge(int pin) { pinMode(pin, OUTPUT); digitalWrite(pin, HIGH); unsigned long start = micros(); bool fast = false; while(analogRead(analogPin) < 200) { if (micros() - start > 1000) { fast = false; break; } fast = true; } digitalWrite(pin, LOW); while(analogRead(analogPin) > 20); return fast; }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
void gruForward(float* x_in, int step) {
    for(int i=0; i<hiddenSize; i++) {
        float z_pre = bz[i], r_pre = br[i];
        for(int j=0; j<inputSize; j++) { z_pre += Wz[i][j] * x_in[j]; r_pre += Wr[i][j] * x_in[j]; }
        for(int j=0; j<hiddenSize; j++) { z_pre += Wz[i][inputSize+j] * h_prev[j]; r_pre += Wr[i][inputSize+j] * h_prev[j]; }
        z_history[step][i] = sigmoid(z_pre); r_history[step][i] = sigmoid(r_pre);
        float ht_pre = bh[i];
        for(int j=0; j<inputSize; j++) ht_pre += Wh[i][j] * x_in[j];
        for(int j=0; j<hiddenSize; j++) ht_pre += Wh[i][inputSize+j] * (r_history[step][j] * h_prev[j]);
        h_tilde_history[step][i] = tanhf(ht_pre);
        h[i] = (1.0 - z_history[step][i]) * h_prev[i] + z_history[step][i] * h_tilde_history[step][i];
    }
    for(int i=0; i<outputSize; i++) { y[i] = by[i]; for(int j=0; j<hiddenSize; j++) y[i] += Why[i][j] * h[j]; }
}
void gruBackwardAndUpdate(int steps, float target) {
    float dy = y[0] - target; float dh_next[hiddenSize]; memset(dh_next, 0, sizeof(dh_next));
    static float dWz[hiddenSize][inputSize+hiddenSize], dWr[hiddenSize][inputSize+hiddenSize], dWh[hiddenSize][inputSize+hiddenSize], dbz[hiddenSize], dbr[hiddenSize], dbh[hiddenSize], dWhy[outputSize][hiddenSize], dby[outputSize];
    memset(dWz, 0, sizeof(dWz)); memset(dWr, 0, sizeof(dWr)); memset(dWh, 0, sizeof(dWh)); memset(dbz, 0, sizeof(dbz)); memset(dbr, 0, sizeof(dbr)); memset(dbh, 0, sizeof(dbh)); memset(dWhy, 0, sizeof(dWhy)); memset(dby, 0, sizeof(dby));
    dby[0] = dy; for(int j=0; j<hiddenSize; j++) dWhy[0][j] = dy * h_history[steps][j];
    for(int t = steps - 1; t >= 0; t--) {
        float dh_curr[hiddenSize];
        for(int i=0; i<hiddenSize; i++) {
            float grad_h = dh_next[i] + Why[0][i] * (t == steps - 1 ? dy : 0);
            float z = z_history[t][i], r = r_history[t][i], ht = h_tilde_history[t][i], hp = h_history[t][i];
            float dht = grad_h * z * (1.0 - ht * ht); float dz = grad_h * (ht - hp) * z * (1.0 - z);
            dbh[i] += dht; dbz[i] += dz;
            for(int j=0; j<inputSize; j++) { dWh[i][j] += dht * x_history[t][j]; dWz[i][j] += dz * x_history[t][j]; }
            for(int j=0; j<hiddenSize; j++) { dWh[i][inputSize+j] += dht * (r_history[t][j] * hp); dWz[i][inputSize+j] += dz * hp; }
            float dr = 0; for(int j=0; j<hiddenSize; j++) dr += dht * Wh[i][inputSize+j] * hp;
            dr *= r * (1.0 - r); dbr[i] += dr;
            for(int j=0; j<inputSize; j++) dWr[i][j] += dr * x_history[t][j];
            for(int j=0; j<hiddenSize; j++) dWr[i][inputSize+j] += dr * hp;
            dh_curr[i] = grad_h;
        }
        for(int i=0; i<hiddenSize; i++) {
            dh_next[i] = 0;
            for(int j=0; j<hiddenSize; j++) {
                float dht_j = dh_curr[j] * z_history[t][j] * (1.0 - h_tilde_history[t][j] * h_tilde_history[t][j]);
                dh_next[i] += dh_curr[j] * (1.0 - z_history[t][j]) + dht_j * Wh[j][inputSize+i] * r_history[t][j];
                float dz_j = dh_curr[j] * (h_tilde_history[t][j] - h_history[t][j]) * z_history[t][j] * (1.0 - z_history[t][j]);
                dh_next[i] += dz_j * Wz[j][inputSize+i];
                float dr_j = (dht_j * Wh[j][inputSize+i] * h_history[t][j]) * r_history[t][j] * (1.0 - r_history[t][j]);
                dh_next[i] += dr_j * Wr[j][inputSize+i];
            }
        }
    }
    auto upd = [&](float& w, float& g, float& s) { g = (g>1?1:(g<-1?-1:g)) + l2Reg*w; s=rho*s+(1-rho)*g*g; w-=learningRate*g/(sqrt(s)+epsilon); };
    for(int i=0; i<hiddenSize; i++) {
        upd(bz[i], dbz[i], sbz[i]); upd(br[i], dbr[i], sbr[i]); upd(bh[i], dbh[i], sbh[i]);
        for(int j=0; j<(inputSize+hiddenSize); j++) { upd(Wz[i][j], dWz[i][j], sWz[i][j]); upd(Wr[i][j], dWr[i][j], sWr[i][j]); upd(Wh[i][j], dWh[i][j], sWh[i][j]); }
        upd(Why[0][i], dWhy[0][i], sWhy[0][i]);
    }
    upd(by[0], dby[0], sby[0]);
}
void gruInit() { float sc = sqrt(2.0 / (inputSize + 2*hiddenSize)); for(int i=0; i<hiddenSize; i++) { bz[i]=0; br[i]=0; bh[i]=0; sbz[i]=0; sbr[i]=0; sbh[i]=0; for(int j=0; j<(inputSize+hiddenSize); j++) { Wz[i][j]=(random(-1000,1000)/1000.0)*sc; Wr[i][j]=(random(-1000,1000)/1000.0)*sc; Wh[i][j]=(random(-1000,1000)/1000.0)*sc; } Why[0][i]=(random(-1000,1000)/1000.0)*sc; } by[0]=0; }
void gruReset() { for(int i=0; i<hiddenSize; i++) { h[i]=0; h_prev[i]=0; } }
void handleRoot() { String s = "<h1>Resistor GRU v2.2</h1><p>MSE: " + String(rollingMSE,6) + "</p>"; server.send(200, "text/html", s); }
void loadPretrained() { memcpy(Wz, pt_Wz, sizeof(Wz)); memcpy(Wr, pt_Wr, sizeof(Wr)); memcpy(Wh, pt_Wh, sizeof(Wh)); memcpy(bz, pt_bz, sizeof(bz)); memcpy(br, pt_br, sizeof(br)); memcpy(bh, pt_bh, sizeof(bh)); memcpy(Why, pt_Why, sizeof(Why)); memcpy(by, pt_by, sizeof(by)); }
void saveWeights() { File f = SPIFFS.open("/gru.bin", "w"); f.write((uint8_t*)Wz, sizeof(Wz)); f.write((uint8_t*)Wr, sizeof(Wr)); f.write((uint8_t*)Wh, sizeof(Wh)); f.write((uint8_t*)bz, sizeof(bz)); f.write((uint8_t*)br, sizeof(br)); f.write((uint8_t*)bh, sizeof(bh)); f.write((uint8_t*)Why, sizeof(Why)); f.write((uint8_t*)by, sizeof(by)); f.close(); }
bool loadWeights() { File f = SPIFFS.open("/gru.bin", "r"); if(!f) return false; f.read((uint8_t*)Wz, sizeof(Wz)); f.read((uint8_t*)Wr, sizeof(Wr)); f.read((uint8_t*)Wh, sizeof(Wh)); f.read((uint8_t*)bz, sizeof(bz)); f.read((uint8_t*)br, sizeof(br)); f.read((uint8_t*)bh, sizeof(bh)); f.read((uint8_t*)Why, sizeof(Why)); f.read((uint8_t*)by, sizeof(by)); f.close(); return true; }
