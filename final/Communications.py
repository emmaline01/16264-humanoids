'''
Communicates with the Arduino via serial.
Python sends the Arduino integer commands 
(encodes characters as space separated ASCII values, prepended by the length of the string)
Arduino sends Python string commands.


Flow:
Python waits for Arduino to communicate it sensed a button press.
Arduino waits for Python to communicate the string to write.
Python waits for Arduino to receive the string to write.
(repeat)

'''

import time
import sys
import serial

def initComms(p):
    print("Port: ", p)

    print("Opening serial...")
    ser = serial.Serial(port=p, baudrate=9600)
    ser.flushInput()
    time.sleep(3)
    print("Serial opened.")
    return ser

def closeComms(ser):
    ser.close()
    time.sleep(0.2)
    print("Port closed")

# send a given command to the Arduino and receive immediate response
# command is a string
def sendCommandToArduino(ser, command):
    ser.write(command.encode('utf-8'))
    print("command sent!")

    while (ser.inWaiting() == 0):
        pass
    
    # s = ser.readline()
    # s = (s.decode('utf-8')).strip()
    # print("immediate response from Arduino: ", s)

# response is a specific string to look for that Arduino sends over
def waitForSpecificResponse(ser, response):
    s = ""
    while (s != response):
        if (s == "error"):
            break
        s = ser.readline()
        s = (s.decode('utf-8')).strip()
    
    print(s + " received! ")

# encode a string in space-separated integers to be sent to the Arduino
def encodeForArduino(s):
    if len(s) == 0:
        print("encoding empty string??")

    ans = str(len(s)) + " "
    for c in s:
        ans += str(ord(c)) + " "
    return ans.strip()


if __name__ == "__main__":
    ser = initComms("COM3")

    for i in range(2):
        print("waiting for button press on Arduino...")
        waitForSpecificResponse(ser, "button")
        
        toWrite = encodeForArduino("homework due in three days")
        sendCommandToArduino(ser, toWrite)
        waitForSpecificResponse(ser, "starting to write")
        print()
    
    closeComms(ser)