#include "SoftwareSerial.h"
#include "LobotServoController.h"

#define LED   13
#define RxPin 8   //Define soft serial port
#define TxPin 9
#define SOUND A0
#define TOUCH 12   //Touch sensor

uint8_t result;
uint8_t volume;

SoftwareSerial mySerial(RxPin, TxPin);
LobotServoController myController(mySerial);

void setup() {
  // put your setup code here, to run once:
  mySerial.begin(9600);
  Serial.begin(9600);
  pinMode(LED, OUTPUT);
  pinMode(SOUND, INPUT);
  pinMode(TOUCH, INPUT);
  digitalWrite(LED, LOW);
  delay(1500);
  myController.runActionGroup(0, 1);
  delay(1000);
}

void loop()
{
  myController.receiveHandler();
  if (myController.isRunning() == false) {
    Serial.println("robot down");
  }

  if (!digitalRead(TOUCH)) {
    // detected touch
    Serial.println("button");
    digitalWrite(LED, HIGH);
    myController.runActionGroup(0, 1);
    delay(80);
  }
  
  //myController.runActionGroup(1, 1);
}
