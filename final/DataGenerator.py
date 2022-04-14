# edited https://www.techgeekbuzz.com/how-to-play-and-record-audio-in-python/
import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
from SilenceRemove import main

word = "homework"
currFileNum = 11
totalRecordings = 1

def record_audio(filename):
    
    #frequency
    fs=32000  #frames per second  
    duration = 10  # seconds in integer
    
    print("Recording...")

    #start recording 
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)

    sd.wait()

    #write the data in filename and save it
    write(filename, fs, myrecording)

folder = "NewRecordings\\" + word + "\\"
if (not os.path.isdir(folder)):
    os.mkdir(folder)
folder = ""
for i in range(currFileNum, currFileNum + totalRecordings):
    filename = folder + "1_nohash_" + str(i) + ".wav"
    record_audio(filename)