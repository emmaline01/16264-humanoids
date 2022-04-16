# edited https://www.techgeekbuzz.com/how-to-play-and-record-audio-in-python/
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def recordAudio(filename):
    #frequency
    fs=32000  #frames per second  
    duration = 10  # seconds in integer
    
    print("Recording...")

    #start recording 
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)

    sd.wait()

    #write the data in filename and save it
    write(filename, fs, myrecording)


if __name__ == "__main__":
    filename ="NewRecordings\\new_record.wav"
    recordAudio(filename)