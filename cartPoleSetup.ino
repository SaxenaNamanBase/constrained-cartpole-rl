// File to be used in case of Arduino Nano with changes needed to be made according to setup.

#define CLK 4  // Pin connected to CLK (A) on the encoder
#define DT 5   // Pin connected to DT (B) on the encoder
#define IN1 2  // IN1 on L298N
#define IN2 3  // IN2 on L298N

int counter = 0;      // Counter to store the encoder position
int currentStateCLK;
int lastStateCLK;

void setup() {
  // Set encoder pins as inputs
  pinMode(CLK, INPUT_PULLUP);
  pinMode(DT, INPUT_PULLUP);

  // Set motor control pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Setup Serial Communication
  Serial.begin(9600);

  // Read the initial state of CLK
  lastStateCLK = digitalRead(CLK);
}

void loop() {
  // Read the current state of CLK
  currentStateCLK = digitalRead(CLK);

  // If the state of CLK has changed, then there is movement
  if (currentStateCLK != lastStateCLK) {
    // If the DT state is different than the CLK state, then the encoder is rotating clockwise
    if (digitalRead(DT) != currentStateCLK) {
      counter++;
    } else {
      counter--;
    }

    // Send angle position on the serial output
    Serial.println(counter);
  }

  // Update lastStateCLK with the current state
  lastStateCLK = currentStateCLK;

  // Check for incoming serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'f') {  // 'f' for forward
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
    } else if (command == 'b') {  // 'b' for backward
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
    } else if (command == 's') {  // 's' for stop
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
    }
  }
}