# https://www.techgeekbuzz.com/how-to-play-and-record-audio-in-python/
import sounddevice as sd
from scipy.io.wavfile import write
import time

def timer(duration):
    while duration: 
        mins, secs = divmod(duration, 60) 
        timer = f"{mins} mins:{secs} seconds Left"
        print(timer, end=" \r") 
        time.sleep(1) 
        duration -= 1

def record_audio(filename):
    
    #frequency
    fs=44100  #frames per second  
    duration = 10  # seconds in integer
    
    print("Recording..........")

    #start recording 
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)

    timer(duration)    #call timer function
    sd.wait()

    #write the data in filename and save it
    write(filename, fs, myrecording)

filename ="NewRecordings\\new_record.wav"
record_audio(filename)