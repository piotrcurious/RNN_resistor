// Arduino code for measuring resistance with capacitor and RNN
// Fixed and Refactored

#define analogPin 0 
#define chargePin 13 
#define dividerPin 1 
#define threshold 648 // ~63.2% of 1023 (10-bit ADC)

// RNN parameters
#define hiddenSize 4 
#define inputSize 4 
#define outputSize 1 
#define learningRate 0.001 // Lowered learning rate for stability in regression

// RNN weights and biases 
float Wxh[hiddenSize][inputSize]; 
float Whh[hiddenSize][hiddenSize]; 
float Why[outputSize][hiddenSize]; 
float bh[hiddenSize]; 
float by[outputSize]; 

// RNN states
float h[hiddenSize]; 
float y[outputSize]; 

// RNN inputs and targets
float x[inputSize]; 
float t[outputSize]; 

// Global Gradients (FIXED SCOPE)
float dWxh[hiddenSize][inputSize]; 
float dWhh[hiddenSize][hiddenSize]; 
float dWhy[outputSize][hiddenSize]; 
float dbh[hiddenSize]; 
float dby[outputSize]; 

void setup() {
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  Serial.begin(9600);
  
  // Good practice to leave an analog pin floating for random seed
  randomSeed(analogRead(5)); 
  rnnInit(); 
}

void loop() {
  float unknownResistor; 
  float knownCapacitor = 0.000001; // 1 uF
  unsigned long startTime; 
  unsigned long totalElapsedTime = 0; // FIXED: Added accumulator
  int pulseCount = 0; 
  
  // Discharge the capacitor (FIXED: Added noise margin)
  pinMode(chargePin, OUTPUT); 
  digitalWrite(chargePin, LOW); 
  while (analogRead(analogPin) > 5) {} 

  // Measure supply voltage
  pinMode(chargePin, INPUT); 
  analogReference(INTERNAL); 
  analogRead(dividerPin); // FIXED: Dummy read to stabilize ADC MUX
  delay(10); 
  float dividerVoltage = analogRead(dividerPin) * (1.1 / 1023.0); 
  float supplyVoltage = dividerVoltage * (10.0 + 1.0); 
  
  // Switch back to default reference
  analogReference(DEFAULT); 
  analogRead(analogPin); // FIXED: Dummy read to clear MUX

  // Charge the capacitor with discrete pulses
  pinMode(chargePin, OUTPUT); 
  
  while (pulseCount < 10) {
    digitalWrite(chargePin, HIGH); 
    startTime = millis(); 
    
    // Wait until threshold is reached
    while(analogRead(analogPin) < threshold) { }
    
    // Accumulate total time
    totalElapsedTime += (millis() - startTime); 
    pulseCount++; 
    
    // Discharge for the next pulse
    digitalWrite(chargePin, LOW); 
    while (analogRead(analogPin) > 5) {} 
  }

  // FIXED MATH: Calculate R from average time. 
  // R = (Time in Seconds) / C. Total time is in ms, so divide by 1000.
  float averageTime_sec = (totalElapsedTime / 10.0) / 1000.0;
  unknownResistor = averageTime_sec / knownCapacitor;

  // Set the input vector for the RNN
  x[0] = analogRead(analogPin); // Final state
  x[1] = pulseCount; 
  x[2] = 1.0; 
  x[3] = supplyVoltage; 

  // Set the target vector for the RNN
  t[0] = unknownResistor; 

  // Execute RNN steps
  rnnForward();
  
  Serial.print("Target Calculated: ");
  Serial.print(t[0]);
  Serial.print(" ohms | RNN Output: ");
  Serial.print(y[0]);
  Serial.println(" ohms");

  rnnBackward();
  rnnUpdate();
  rnnReset();
  
  delay(1000); // Small delay to prevent serial spamming
}

// RNN forward pass function
void rnnForward() {
  // compute hidden state
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = bh[i]; 
    for (int j = 0; j < inputSize; j++) {
      h[i] += Wxh[i][j] * x[j]; 
    }
    for (int k = 0; k < hiddenSize; k++) {
      h[i] += Whh[i][k] * h[k]; 
    }
    h[i] = tanh(h[i]); // standard math.h tanh is sufficient
  }

  // compute output state
  for (int i = 0; i < outputSize; i++) {
    y[i] = by[i]; 
    for (int j = 0; j < hiddenSize; j++) {
      y[i] += Why[i][j] * h[j]; 
    }
    // FIXED: Removed tanh() here. Regression requires linear output.
  }
}

// RNN backward pass function
void rnnBackward() {
  // compute output error
  float dy[outputSize]; 
  for (int i = 0; i < outputSize; i++) {
    dy[i] = y[i] - t[i]; 
    // FIXED: Derivative of linear activation is 1, so no multiplication needed here.
  }

  // compute hidden error
  float dh[hiddenSize]; 
  for (int i = 0; i < hiddenSize; i++) {
    dh[i] = 0; 
    for (int j = 0; j < outputSize; j++) {
      dh[i] += Why[j][i] * dy[j]; 
    }
    dh[i] *= (1.0 - h[i] * h[i]); // derivative of tanh
  }

  // Calculate gradients (Variables are now global)
  for (int i = 0; i < hiddenSize; i++) {
    dbh[i] = dh[i]; 
    for (int j = 0; j < inputSize; j++) {
      dWxh[i][j] = dh[i] * x[j]; 
    }
    for (int k = 0; k < hiddenSize; k++) {
      dWhh[i][k] = dh[i] * h[k]; 
    }
  }

  for (int i = 0; i < outputSize; i++) {
    dby[i] = dy[i]; 
    for (int j = 0; j < hiddenSize; j++) {
      dWhy[i][j] = dy[i] * h[j]; 
    }
  }
}

// RNN update function
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

// RNN initialization function
void rnnInit() {
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] = random(-1000, 1000) / 1000.0; 
    for (int j = 0; j < inputSize; j++) {
      Wxh[i][j] = random(-1000, 1000) / 1000.0; 
    }
    for (int k = 0; k < hiddenSize; k++) {
      Whh[i][k] = random(-1000, 1000) / 1000.0; 
    }
  }

  for (int i = 0; i < outputSize; i++) {
    by[i] = random(-1000, 1000) / 1000.0; 
    for (int j = 0; j < hiddenSize; j++) {
      Why[i][j] = random(-1000, 1000) / 1000.0; 
    }
  }
}

// RNN reset function
void rnnReset() {
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = 0;
  }
}
