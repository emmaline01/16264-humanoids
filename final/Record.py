# edited https://www.techgeekbuzz.com/how-to-play-and-record-audio-in-python/
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename):
    
    #frequency
    fs=44100  #frames per second  
    duration = 1  # seconds in integer
    
    print("Recording..........")

    #start recording 
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)

    sd.wait()

    #write the data in filename and save it
    write(filename, fs, myrecording)

filename ="NewRecordings\\new_record.wav"
record_audio(filename)