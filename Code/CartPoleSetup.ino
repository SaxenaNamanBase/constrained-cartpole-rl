#define CLK 2  // Pin connected to CLK (A) on the encoder
#define DT 5  // Pin connected to DT (B) on the encoder
#define IN1 6  // IN1 on L298N
#define IN2 7  // IN2 on L298N

volatile long counter = 0;  // Counter to store the encoder position
volatile int lastStateCLK;
volatile bool valueChanged = false;

void setup() {
  // Set encoder pins as inputs
  pinMode(CLK, INPUT_PULLUP);
  pinMode(DT, INPUT_PULLUP);

  // Set motor control pins as outputs
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  // Setup Serial Communication
  Serial.begin(1000000);  // High baud rate for faster communication
  
  // Read the initial state of CLK
  lastStateCLK = digitalRead(CLK);
  attachInterrupt(digitalPinToInterrupt(CLK), updateEncoder, CHANGE);  //Runs whenever an interrupt is triggered
}

void loop() {
  // Check for incoming serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'f') {
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
    } else if (command == 'b') {
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
    } else if (command == 's') {
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
    }
  }
}

void updateEncoder() {

  // Read the current state of CLK
  int currentStateCLK = digitalRead(CLK);

  // If the state of CLK has changed, then there is movement
  if (currentStateCLK != lastStateCLK) {
    // If the DT state is different than the CLK state, then the encoder is rotating clockwise
    counter += (digitalRead(DT) != currentStateCLK) ? 1 : -1;
    // Send angle position on the serial output
    Serial.write((uint8_t*)(&counter), sizeof(counter));
  }
  lastStateCLK = currentStateCLK;
}