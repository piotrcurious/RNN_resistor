
// Arduino code for measuring resistance with capacitor and RNN
// Based on https://www.arduino.cc/en/Tutorial/CapacitanceMeter and https://www.circuitbasics.com/how-to-make-an-arduino-capacitance-meter/

#define analogPin 0 // analog pin for measuring voltage
#define chargePin 13 // digital pin for charging and discharging capacitor and input of supply voltage divider
#define resistorValue 10000.0F // known resistor value in ohms
#define threshold 648 // analog value for 63.2% of 5V
#define dividerPin 1 // analog pin for measuring supply voltage with voltage divider

// RNN parameters
#define hiddenSize 4 // number of hidden units
#define inputSize 4 // number of input features
#define outputSize 1 // number of output features
#define learningRate 0.01 // learning rate for gradient descent

// RNN weights and biases (randomly initialized)
float Wxh[hiddenSize][inputSize]; // input to hidden weights
float Whh[hiddenSize][hiddenSize]; // hidden to hidden weights
float Why[outputSize][hiddenSize]; // hidden to output weights
float bh[hiddenSize]; // hidden bias
float by[outputSize]; // output bias

// RNN states
float h[hiddenSize]; // hidden state
float y[outputSize]; // output state

// RNN inputs and targets
float x[inputSize]; // input vector
float t[outputSize]; // target vector

void setup() {
  pinMode(chargePin, OUTPUT);
  digitalWrite(chargePin, LOW);
  Serial.begin(9600);
  randomSeed(analogRead(5)); // use analog pin 5 as a random seed
  rnnInit(); // initialize RNN weights and biases randomly
}

void loop() {
  float unknownResistor; // unknown resistor value in ohms
  float knownCapacitor = 0.000001; // known capacitor value in farads (1 uF)
  unsigned long startTime; // start time of charging pulse
  unsigned long elapsedTime; // elapsed time of charging pulse
  int pulseCount = 0; // number of charging pulses
  
  // discharge the capacitor
  pinMode(chargePin, OUTPUT); // set charge pin as output
  digitalWrite(chargePin, LOW); // set charge pin to low
  while (analogRead(analogPin) > 0) {}

  // measure the supply voltage with voltage divider and internal reference voltage
  pinMode(chargePin, INPUT); // set charge pin as input to end discharge pulse and start input of supply voltage divider
  analogReference(INTERNAL); // use internal reference voltage (1.1V)
  delay(10); // wait for internal reference voltage to stabilize
  float dividerVoltage = analogRead(dividerPin) * (1.1 / 1023.0); // read voltage at divider pin
  float supplyVoltage = dividerVoltage * (10 + 1); // calculate supply voltage from divider ratio (1:10)
  
  // switch back to vss supply voltage reference after internal voltage reference measurement
  analogReference(DEFAULT); // use default reference voltage (5V)

  
  // charge the capacitor with discrete pulses and count how many pulses it takes to charge it above threshold
  pinMode(chargePin, OUTPUT); // set charge pin as output
  digitalWrite(chargePin, HIGH); // start charge pulse
  startTime = millis(); // record start time of charge pulse
  while (true) {
    x[0] = analogRead(analogPin); // read analog input before ending charge pulse
    if (x[0] >= threshold) { // check if voltage reaches 63.2% of supply voltage
      elapsedTime = millis() - startTime; // record elapsed time of charge pulse
      pulseCount++; // increment pulse count
      digitalWrite(chargePin, LOW); // end charge pulse
      pinMode(chargePin, INPUT); // start input of supply voltage divider
      while (analogRead(analogPin) > 0) {} // wait until capacitor is fully discharged
      pinMode(chargePin, OUTPUT); // end input of supply voltage divider and start charge pulse
      digitalWrite(chargePin, HIGH); // start another charge pulse
      startTime = millis(); // record start time of charge pulse
    }
    if (pulseCount >= 10) { // stop after 10 pulses
      break;
    }
  }

  // calculate the unknown resistor value from the average elapsed time and the known capacitor value
  unknownResistor = (elapsedTime / 10.0) / knownCapacitor / supplyVoltage * 1000.0;

  // set the input vector for the RNN
  x[1] = pulseCount; // pulse count
  x[2] = 1; // software trigger indicating new resistance measurement
  x[3] = supplyVoltage; // supply voltage

  // set the target vector for the RNN
  t[0] = unknownResistor; // unknown resistor value

  // forward pass of the RNN
  rnnForward();

  // print the RNN output and the target value
  Serial.print("RNN output: ");
  Serial.print(y[0]);
  Serial.println(" ohms");
  Serial.print("Target value: ");
  Serial.print(t[0]);
  Serial.println(" ohms");

  // backward pass of the RNN
  rnnBackward();

  // update the RNN weights and biases with gradient descent
  rnnUpdate();

  // reset the RNN hidden state to zero
  rnnReset();

}

// tanh activation function
float tanh(float x) {
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); // use the formula for hyperbolic tangent
}

// RNN forward pass function
void rnnForward() {
  // compute hidden state
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = bh[i]; // add hidden bias
    for (int j = 0; j < inputSize; j++) {
      h[i] += Wxh[i][j] * x[j]; // add input to hidden weights times input vector
    }
    for (int k = 0; k < hiddenSize; k++) {
      h[i] += Whh[i][k] * h[k]; // add hidden to hidden weights times previous hidden state
    }
    h[i] = tanh(h[i]); // apply tanh activation function
  }

  // compute output state
  for (int i = 0; i < outputSize; i++) {
    y[i] = by[i]; // add output bias
    for (int j = 0; j < hiddenSize; j++) {
      y[i] += Why[i][j] * h[j]; // add hidden to output weights times hidden state
    }
    y[i] = tanh(y[i]); // apply tanh activation function
  }
}

// RNN backward pass function
void rnnBackward() {
  // compute output error
  float dy[outputSize]; // output error vector
  for (int i = 0; i < outputSize; i++) {
    dy[i] = y[i] - t[i]; // subtract target vector from output vector
    dy[i] *= (1 - y[i] * y[i]); // multiply by derivative of tanh function
  }

  // compute hidden error
  float dh[hiddenSize]; // hidden error vector
  for (int i = 0; i < hiddenSize; i++) {
    dh[i] = 0; // initialize to zero
    for (int j = 0; j < outputSize; j++) {
      dh[i] += Why[j][i] * dy[j]; // add hidden to output weights times output error
    }
    dh[i] *= (1 - h[i] * h[i]); // multiply by derivative of tanh function
  }

  // compute gradients of weights and biases
  float dWxh[hiddenSize][inputSize]; // gradient of input to hidden weights
  float dWhh[hiddenSize][hiddenSize]; // gradient of hidden to hidden weights
  float dWhy[outputSize][hiddenSize]; // gradient of hidden to output weights
  float dbh[hiddenSize]; // gradient of hidden bias
  float dby[outputSize]; // gradient of output bias

  for (int i = 0; i < hiddenSize; i++) {
    dbh[i] = dh[i]; // gradient of hidden bias is equal to hidden error
    for (int j = 0; j < inputSize; j++) {
      dWxh[i][j] = dh[i] * x[j]; // gradient of input to hidden weights is equal to hidden error times input vector
    }
    for (int k = 0; k < hiddenSize; k++) {
      dWhh[i][k] = dh[i] * h[k]; // gradient of hidden to hidden weights is equal to hidden error times previous hidden state
    }
  }

  for (int i = 0; i < outputSize; i++) {
    dby[i] = dy[i]; // gradient of output bias is equal to output error
    for (int j = 0; j < hiddenSize; j++) {
      dWhy[i][j] = dy[i] * h[j]; // gradient of hidden to output weights is equal to output error times hidden state
    }
  }

}

// RNN update function
void rnnUpdate() {
  // update weights and biases with gradient descent and learning rate
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] -= learningRate * dbh[i]; // update hidden bias
    for (int j = 0; j < inputSize; j++) {
      Wxh[i][j] -= learningRate * dWxh[i][j]; // update input to hidden weights
    }
    for (int k = 0; k < hiddenSize; k++) {
      Whh[i][k] -= learningRate * dWhh[i][k]; // update hidden to hidden weights
    }
  }

  for (int i = 0; i < outputSize; i++) {
    by[i] -= learningRate * dby[i]; // update output bias
    for (int j = 0; j < hiddenSize; j++) {
      Why[i][j] -= learningRate * dWhy[i][j]; // update hidden to output weights
    }
  }

}

// RNN initialization function
void rnnInit() {
  // initialize weights and biases randomly between -1 and 1
  for (int i = 0; i < hiddenSize; i++) {
    bh[i] = random(-1000, 1000) / 1000.0; // initialize hidden bias
    for (int j = 0; j < inputSize; j++) {
      Wxh[i][j] = random(-1000, 1000) / 1000.0; // initialize input to hidden weights
    }
    for (int k = 0; k < hiddenSize; k++) {
      Whh[i][k] = random(-1000, 1000) / 1000.0; // initialize hidden to hidden weights
    }
  }

  for (int i = 0; i < outputSize; i++) {
    by[i] = random(-1000, 1000) / 1000.0; // initialize output bias
    for (int j = 0; j < hiddenSize; j++) {
      Why[i][j] = random(-1000, 1000) / 1000.0; // initialize hidden to output weights
    }
  }

}

// RNN reset function
void rnnReset() {
  // reset hidden state to zero
  for (int i = 0; i < hiddenSize; i++) {
    h[i] = 0;
  }

}
