#include "SoftwareSerial.h"
#include "LobotServoController.h"

#define RxPin 8    //Define soft serial port
#define TxPin 9
#define LED  13
#define TOUCH 12   //Touch sensor

SoftwareSerial mySerial(RxPin, TxPin);
LobotServoController myController(mySerial);

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600);
  
  pinMode(LED, OUTPUT);
  pinMode(TOUCH, INPUT);
  digitalWrite(LED, LOW);
  //myController.runActionGroup(0, 1);
  delay(1500);
}


void run()
{
  boolean wasPressed = false;
  
  if (myController.isRunning())
    // skip rest of function if robot is moving
    return;

  // check touch sensor
  if (!digitalRead(TOUCH)) {
    // detected touch
    Serial.println("button");
    digitalWrite(LED, HIGH);
    wasPressed = true;
    delay(80);
  }
  else {
    // didn't detect touch
    digitalWrite(LED, LOW);
    wasPressed = false;
  }

  // wait for serial commands from Python
  if (Serial.available() > 0) {
    int numChars = Serial.parseInt();
    if (numChars <= 0) {
      Serial.println("error");
    }
    else {
      String s = "";
      for (int i = 0; i < numChars; i++) {
        int c = Serial.parseInt();
        s += String(c) + " ";
      }
      Serial.println(s);
      Serial.println("starting to write");
      
      // TODO: start writing the string s
    }
  }
  
}

void loop() {
  myController.receiveHandler();
  run();   
}
