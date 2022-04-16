'''
Records you saying a specified word "totalRecordings" number of times, and adds
the recordings to the existing SpeechCommands dataset.

'''

# edited https://www.techgeekbuzz.com/how-to-play-and-record-audio-in-python/
import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
from SilenceRemove import trimSilence

word = "two"
currFileNum = 35 # inclusive
totalRecordings = 13

def recordAudio(filename):
    #frequency
    fs=32000  #frames per second  
    duration = 2  # seconds in integer

    #start recording 
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)

    sd.wait()

    #write the data in filename and save it
    write(filename, fs, myrecording)


folder = "SpeechCommands\\speech_commands_v0.02\\" + word + "\\"
if (not os.path.isdir(folder)):
    os.mkdir(folder)

file = open("SpeechCommands\\speech_commands_v0.02\\testing_list.txt", "a")  # append mode
for i in range(currFileNum, currFileNum + totalRecordings):
    filename = "1_nohash_" + str(i) + ".wav"
    fullFilename = folder + filename

    print(str(i + 1 - currFileNum) + "/" + str(totalRecordings) + " Recording...")
    recordAudio(fullFilename)

    trimSilence(fullFilename, fullFilename)

    # add this file to training list
    file.write(word + "/" + filename + "\n")

file.close()